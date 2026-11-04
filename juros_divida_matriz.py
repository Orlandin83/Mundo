"""Matriz fiscal do G20 e regressões de juros reais, sem indicador composto.

Execute: python juros_divida_matriz.py
A matriz usa escores Z: zero é a média dos países em cada indicador.

Inflação em 12 meses e juros no último mês comum: BIS, ABS e BCRA.

Fontes públicas, sem chave:
https://data.bis.org/topics/CBPOL
https://data.bis.org/topics/CPI
https://www.abs.gov.au/statistics/application-programming-interfaces-apis/data-api-user-guide
https://www.bcra.gob.ar/apis-banco-central/

O juro real é uma aproximação: taxa anual no fim do mês / inflação em 12 meses, pela
fórmula de Fisher. Não é o retorno realizado de uma aplicação em 12 meses.
As convenções de anualização e os instrumentos diferem entre países.
"""

# %% Bibliotecas e configuração
from datetime import datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_URL = "https://www.imf.org/external/datamapper/api/v1"
INDICADORES = {
    "divida_pib": "GGXWDG_NGDP",
    "saldo_primario_pib": "GGXONLB_NGDP",
}
# O DataMapper disponibiliza o saldo primário pelo Fiscal Monitor.
ALTERNATIVAS = {"GGXONLB_NGDP": "GGXONLB_G01_GDP_PT"}
ANO_REFERENCIA = datetime.now(ZoneInfo("America/Sao_Paulo")).year - 1
g20 = {
    "África do Sul": "ZAF",
    "Alemanha": "DEU",
    "Arábia Saudita": "SAU",
    "Argentina": "ARG",
    "Austrália": "AUS",
    "Brasil": "BRA",
    "Canadá": "CAN",
    "China": "CHN",
    "Coreia do Sul": "KOR",
    "Estados Unidos": "USA",
    "França": "FRA",
    "Índia": "IND",
    "Indonésia": "IDN",
    "Itália": "ITA",
    "Japão": "JPN",
    "México": "MEX",
    "Reino Unido": "GBR",
    "Rússia": "RUS",
    "Turquia": "TUR",
}


# %% Fontes e coleta de inflação e juros
BIS_URL = "https://stats.bis.org/api/v2/data/dataflow/BIS"
ABS_URL = "https://data.api.abs.gov.au/rest/data/ABS,CPI,2.0.0/3.10001.10.50.M"
BCRA_URL = "https://api.bcra.gob.ar/estadisticas/v4.0/Monetarias"
ISO2 = {
    "ZAF": "ZA", "DEU": "DE", "SAU": "SA", "ARG": "AR", "AUS": "AU",
    "BRA": "BR", "CAN": "CA", "CHN": "CN", "KOR": "KR", "USA": "US",
    "FRA": "FR", "IND": "IN", "IDN": "ID", "ITA": "IT", "JPN": "JP",
    "MEX": "MX", "GBR": "GB", "RUS": "RU", "TUR": "TR",
}
EURO = {"DEU", "FRA", "ITA"}


def ler_csv(sessao, url, **params):
    resposta = sessao.get(url, params=params, timeout=60)
    resposta.raise_for_status()
    tabela = pd.read_csv(StringIO(resposta.text), dtype={"REF_AREA": str})
    obrigatorias = {"TIME_PERIOD", "OBS_VALUE"}
    if not obrigatorias.issubset(tabela.columns):
        raise ValueError(f"Formato inesperado na resposta de {url}")
    tabela["OBS_VALUE"] = pd.to_numeric(tabela["OBS_VALUE"], errors="coerce")
    return tabela


def ultima_observacao(tabela, hoje, mensal=False):
    """Não usa a ordem da API nem avança sobre um mês sem observação."""
    tabela = tabela.copy()
    tabela["data"] = pd.to_datetime(tabela["TIME_PERIOD"], errors="coerce")
    if mensal:
        tabela["data"] = tabela["data"] + pd.offsets.MonthEnd(0)
    validas = tabela.dropna(subset=["data", "OBS_VALUE"])
    validas = validas.loc[validas["data"] <= hoje].sort_values("data")
    if validas.empty:
        return None
    return validas.iloc[-1]


def calcular_juro_real(juros, inflacao):
    """Entradas em porcentagem; saída em % a.a., sem arredondar os dados."""
    if pd.isna(juros) or pd.isna(inflacao) or inflacao <= -100:
        return float("nan")
    return 100 * ((1 + juros / 100) / (1 + inflacao / 100) - 1)


def preparar_mensal(tabela, hoje, diaria=False):
    """Uma observação válida por mês encerrado, sem preencher lacunas."""
    tabela = tabela.copy()
    tabela["data"] = pd.to_datetime(tabela["TIME_PERIOD"], errors="coerce")
    tabela["OBS_VALUE"] = pd.to_numeric(tabela["OBS_VALUE"], errors="coerce")
    tabela = tabela.dropna(subset=["data", "OBS_VALUE"])
    tabela["mes"] = tabela["data"].dt.to_period("M")
    tabela = tabela.loc[tabela["mes"] < hoje.to_period("M")]
    if not diaria:
        # Série mensal BIS: juros do último dia útil, rotulados pelo mês.
        tabela["data"] = tabela["mes"].dt.to_timestamp() + pd.offsets.MonthEnd(0)
    tabela = tabela.sort_values("data").drop_duplicates("mes", keep="last")
    if diaria:
        # Evita tratar uma série interrompida no início do mês como fechamento.
        fim_mes = tabela["mes"].dt.to_timestamp() + pd.offsets.MonthEnd(0)
        tabela = tabela.loc[(fim_mes - tabela["data"]).dt.days <= 7]
    return tabela.set_index("mes")


def ultimo_mes_comum(series):
    """Interseção efetiva: não basta o mínimo das datas mais recentes."""
    if not series:
        raise ValueError("Nenhuma série fornecida.")
    comuns = set(series[0].index)
    for serie in series[1:]:
        comuns.intersection_update(serie.index)
    if not comuns:
        raise ValueError("Não há mês com inflação e juros válidos para todos os países.")
    return max(comuns)


def consultar_juros_inflacao(paises):
    """Alinha todos os países ao último mês comum de inflação e juros.

    BIS mensal: juros de fim de período, não média mensal. Argentina:
    última TAMAR efetiva anual disponível dentro do mês (máximo 7 dias
    antes do fechamento). Inflação: variação em 12 meses naquele mês.
    São períodos de referência, não datas de publicação/vintages históricos.
    A busca começa em outubro/2024, início da TAMAR usada na comparação.
    """
    hoje = pd.Timestamp(datetime.now(ZoneInfo("America/Sao_Paulo")).date())
    inicio = "2024-10"
    fim = str(hoje.to_period("M") - 1)
    areas_cpi = "+".join(sorted({ISO2[iso] for iso in paises.values() if iso != "AUS"}))
    areas_juros = "+".join(sorted({
        "XM" if iso in EURO else ISO2[iso]
        for iso in paises.values() if iso != "ARG"
    }))
    with requests.Session() as sessao:
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        sessao.mount("https://", HTTPAdapter(max_retries=retry))
        cpi = ler_csv(
            sessao, f"{BIS_URL}/WS_LONG_CPI/1.0/M.{areas_cpi}.771",
            format="csv", startPeriod=inicio, endPeriod=fim,
        ) if areas_cpi else pd.DataFrame()
        juros = ler_csv(
            sessao, f"{BIS_URL}/WS_CBPOL/1.0/M.{areas_juros}",
            format="csv", startPeriod=inicio, endPeriod=fim,
        ) if areas_juros else pd.DataFrame()
        australia = None
        if "AUS" in paises.values():
            australia = ler_csv(
                sessao, ABS_URL, format="csvfilewithlabels",
                startPeriod=inicio, endPeriod=fim,
            )
        tamar = None
        if "ARG" in paises.values():
            resposta = sessao.get(BCRA_URL, params={"IdVariable": 45}, timeout=60)
            resposta.raise_for_status()
            cadastro = next((v for v in resposta.json().get("results", [])
                             if v["idVariable"] == 45), None)
            if cadastro is None or "efectivo anual" not in cadastro["unidadExpresion"].lower():
                raise ValueError("BCRA não retornou a TAMAR efetiva anual esperada (45).")
            registros, offset = [], 0
            while True:
                resposta = sessao.get(f"{BCRA_URL}/45", params={
                    "desde": inicio + "-01", "hasta": hoje.date().isoformat(),
                    "limit": 1000, "offset": offset,
                }, timeout=60)
                resposta.raise_for_status()
                pagina = resposta.json()
                detalhes = [obs for serie in pagina.get("results", [])
                            if serie["idVariable"] == 45 for obs in serie["detalle"]]
                registros.extend(detalhes)
                offset += len(detalhes)
                if not detalhes or offset >= pagina["metadata"]["resultset"]["count"]:
                    break
            tamar = pd.DataFrame(registros, columns=["fecha", "valor"]).rename(
                columns={"fecha": "TIME_PERIOD", "valor": "OBS_VALUE"}
            )

    series_cpi, series_juros = {}, {}
    for iso in paises.values():
        area = ISO2[iso]
        area_juros = "XM" if iso in EURO else area
        series_cpi[iso] = preparar_mensal(
            australia if iso == "AUS" else cpi.loc[cpi["REF_AREA"] == area], hoje,
        )
        series_juros[iso] = preparar_mensal(
            tamar if iso == "ARG" else juros.loc[juros["REF_AREA"] == area_juros],
            hoje, diaria=iso == "ARG",
        )
    mes = ultimo_mes_comum(list(series_cpi.values()) + list(series_juros.values()))
    if (hoje.to_period("M") - mes).n > 3:
        warnings.warn(f"O último mês comum está defasado: {mes}.", stacklevel=2)

    valores, fontes = [], []
    for pais, iso in paises.items():
        area = ISO2[iso]
        area_juros = "XM" if iso in EURO else area
        obs_cpi, obs_juros = series_cpi[iso].loc[mes], series_juros[iso].loc[mes]
        inflacao, taxa = float(obs_cpi["OBS_VALUE"]), float(obs_juros["OBS_VALUE"])
        valores.append({
            "pais": pais, "inflacao_12m": inflacao, "juros_nominais": taxa,
            "juro_real_ex_post": calcular_juro_real(taxa, inflacao),
        })
        fontes.append({
            "pais": pais, "iso3": iso, "periodo_comum": str(mes),
            "coletado_em": hoje.date().isoformat(),
            "data_inflacao_12m": obs_cpi["data"],
            "data_juros_nominais": obs_juros["data"],
            "instrumento_juros": "TAMAR privada, efetiva anual (proxy)" if iso == "ARG" else obs_juros.get("COMPILATION"),
            "juros_proxy": iso == "ARG",
            "serie_inflacao": "ABS CPI 3.10001.10.50.M" if iso == "AUS" else f"BIS WS_LONG_CPI M.{area}.771",
            "serie_juros": "BCRA Monetarias 45" if iso == "ARG" else f"BIS WS_CBPOL M.{area_juros}",
            "url_inflacao": ABS_URL if iso == "AUS" else f"{BIS_URL}/WS_LONG_CPI/1.0/M.{area}.771",
            "url_juros": f"{BCRA_URL}/45" if iso == "ARG" else f"{BIS_URL}/WS_CBPOL/1.0/M.{area_juros}",
            "status": "mesmo mês; juros de fim de período",
        })
    resultado = pd.DataFrame(valores)
    resultado.attrs["periodo_comum"] = str(mes)
    return resultado, pd.DataFrame(fontes)


# %% Consulta e organização dos dados
def consultar_api(sessao, recurso, **params):
    """Consulta a API pública do FMI, sem necessidade de chave."""
    resposta = sessao.get(f"{API_URL}/{recurso}", params=params, timeout=60)
    resposta.raise_for_status()
    return resposta.json()


def normalizar_z(serie):
    """Escore Z entre países disponíveis, com desvio populacional (ddof=0).

    Sem dispersão, atribui zero aos valores disponíveis; preserva ausentes.
    """
    desvio = serie.std(ddof=0)
    if desvio == 0:
        return serie.where(serie.isna(), 0.0)
    return (serie - serie.mean()) / desvio


def montar_dataframe(ano_referencia=ANO_REFERENCIA, incluir_juros=True):
    """Uma linha por país, valores em % do PIB (80 significa 80%).

    Dívida bruta no ano de referência; média aritmética do déficit primário
    dos anos disponíveis nos dez anos até esse ano, inclusive.
    Sem observações disponíveis, a média fica ausente (NaN).
    Déficit = -saldo primário: superávits entram como valores negativos.
    Anos encerrados ainda podem conter estimativas revisáveis do FMI.
    Escores Z são adimensionais.
    Juros e inflação usam o último mês comum a todos os países, independentemente
    do ano fiscal escolhido; datas/fontes em attrs['fontes_juros_inflacao'].
    Documentação: https://www.imf.org/external/datamapper/api/
    """
    if not isinstance(ano_referencia, int) or isinstance(ano_referencia, bool):
        raise ValueError("ano_referencia deve ser um ano inteiro.")
    if ano_referencia > ANO_REFERENCIA:
        raise ValueError("Use um ano encerrado para não incluir anos futuros.")

    anos = [str(ano) for ano in range(ano_referencia - 9, ano_referencia + 1)]
    series = {}
    fontes = {}
    with requests.Session() as sessao:
        catalogo = consultar_api(sessao, "indicators")["indicators"]
        for nome, solicitado in INDICADORES.items():
            codigo = solicitado
            if codigo not in catalogo:
                codigo = ALTERNATIVAS.get(solicitado, solicitado)
                if codigo not in catalogo:
                    raise ValueError(f"Indicador indisponível no DataMapper: {solicitado}")
                warnings.warn(
                    f"{solicitado} indisponível; usando {codigo} "
                    f"({catalogo[codigo].get('source', '')}).",
                    stacklevel=2,
                )
            dados = consultar_api(sessao, codigo, periods=",".join(anos))
            valores = dados.get("values", {}).get(codigo)
            if not valores:
                raise ValueError(f"O FMI não retornou valores para {codigo}.")
            # Reindexar impede que anos extras ou agregados entrem na média.
            series[nome] = (
                pd.DataFrame.from_dict(valores, orient="index")
                .reindex(index=list(g20.values()), columns=anos)
                .apply(pd.to_numeric, errors="coerce")
            )
            fontes[nome] = {
                "codigo_solicitado": solicitado,
                "codigo_usado": codigo,
                "fonte": catalogo[codigo].get("source"),
            }

    saldo = series["saldo_primario_pib"]
    resultado = pd.DataFrame({
        "pais": pd.Series({iso: pais for pais, iso in g20.items()}),
        "divida_pib": series["divida_pib"][str(ano_referencia)],
        "media_deficit_primario_pib": -saldo.mean(axis=1),
    }).reindex(list(g20.values())).reset_index(drop=True)
    resultado["z_divida_pib"] = normalizar_z(resultado["divida_pib"])
    resultado["z_deficit_primario"] = normalizar_z(
        resultado["media_deficit_primario_pib"]
    )
    resultado.attrs["fontes"] = fontes
    resultado.attrs["ano_fiscal"] = ano_referencia
    resultado.attrs["unidades"] = {
        "divida_pib": "% do PIB",
        "media_deficit_primario_pib": "% do PIB",
        "z_divida_pib": "adimensional",
        "z_deficit_primario": "adimensional",
    }
    if incluir_juros:
        dados_juros, fontes_juros = consultar_juros_inflacao(g20)
        metadados = resultado.attrs.copy()
        resultado = resultado.merge(dados_juros, on="pais", how="left", validate="one_to_one")
        resultado.attrs = metadados
        resultado.attrs["fontes_juros_inflacao"] = fontes_juros.to_dict("records")
        resultado.attrs["periodo_comum"] = dados_juros.attrs["periodo_comum"]
        resultado.attrs["unidades"].update({
            "inflacao_12m": "% em 12 meses",
            "juros_nominais": "% a.a., convenção da fonte",
            "juro_real_ex_post": "% a.a., aproximação com inflação passada",
        })
    colunas = {
        "pais": "país",
        "media_deficit_primario_pib": "déficit primário (média)",
        "divida_pib": "dívida/pib",
        "z_divida_pib": "z_divida_pib",
        "z_deficit_primario": "z_deficit_primario",
        "juro_real_ex_post": "juro real",
    }
    # Mantém o mesmo formato quando a consulta de juros é desativada.
    if not incluir_juros:
        resultado["juro_real_ex_post"] = float("nan")
    resultado = resultado[list(colunas)].rename(columns=colunas)
    resultado.attrs["unidades"] = {
        "déficit primário (média)": "% do PIB",
        "dívida/pib": "% do PIB",
        "z_divida_pib": "adimensional",
        "z_deficit_primario": "adimensional",
        "juro real": "% a.a., aproximação com inflação passada",
    }
    return resultado


# %% Regressões — padrão visual de superquarta.py
def gerar_graficos_regressao(df, arquivo=None):
    """Duas regressões OLS com intercepto; juro real é a variável dependente."""
    from matplotlib.transforms import Bbox

    if arquivo is None:
        arquivo = Path(__file__).with_name("juros_divida_matriz_regressoes.png")
    paineis = [
        ("déficit primário (média)", "Juro real × déficit primário médio", "Déficit primário médio (% do PIB)"),
        ("dívida/pib", "Juro real × dívida / PIB", "Dívida / PIB (%)"),
    ]
    fig, axes = plt.subplots(nrows=2, figsize=(17, 11))
    fig.subplots_adjust(left=0.075, right=0.77, bottom=0.17, top=0.87, hspace=0.65)
    periodo = df.attrs.get("periodo_comum", "não informado")
    ano = df.attrs.get("ano_fiscal", ANO_REFERENCIA)
    fig.suptitle("Juro real, dívida e déficit primário | G20", x=0.075, y=0.97,
                 ha="left", fontsize=22, fontweight="bold", color="darkblue")
    fig.text(0.075, 0.935,
             f"Juros e inflação: {periodo}  •  Dívida/PIB: {ano}  •  Déficit médio: {ano - 9}–{ano}",
             fontsize=11, color="dimgray")
    estatisticas = []
    for ax, (coluna, titulo, rotulo) in zip(axes, paineis):
        dados = df[["país", coluna, "juro real"]].replace([np.inf, -np.inf], np.nan).dropna()
        x, y = dados[coluna].to_numpy(dtype=float), dados["juro real"].to_numpy(dtype=float)
        if len(dados) < 3 or np.ptp(x) == 0:
            plt.close(fig)
            raise ValueError(f"Dados insuficientes para a regressão de {coluna}.")
        inclinacao, intercepto = np.polyfit(x, y, 1)
        previsto = intercepto + inclinacao * x
        sst = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum((y - previsto) ** 2) / sst if sst > 0 else float("nan")
        estatisticas.append({"variavel": coluna, "n": len(dados),
                             "inclinacao": inclinacao, "intercepto": intercepto, "r2": r2})
        xx = np.linspace(x.min(), x.max(), 200)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6, zorder=0)
        ax.plot(xx, intercepto + inclinacao * xx, color="black", alpha=0.7,
                linewidth=1.7, label="Regressão linear (OLS)")
        ax.scatter(x, y, color="darkblue", alpha=0.7, s=42, label="Países", zorder=3)
        brasil = dados.loc[dados["país"] == "Brasil"]
        ax.scatter(brasil[coluna], brasil["juro real"], color="lightblue",
                   edgecolor="darkblue", s=90, label="Brasil", zorder=4)
        ax.set_title(titulo, loc="left", fontweight="bold", fontsize=14, color="darkblue", pad=32)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=3, fontsize=8,
                  frameon=False, borderaxespad=0)
        ax.set_xlabel(rotulo, fontsize=10)
        ax.set_ylabel("Juro real (% a.a.)", fontsize=10)
        ax.margins(x=0.12, y=0.25)
        sinal = "+" if intercepto >= 0 else "−"
        texto = (f"y = {inclinacao:.2f}x {sinal} {abs(intercepto):.2f}\n"
                 f"R² = {r2:.3f}\nn = {len(dados)} países").replace(".", ",")
        ax.text(1.035, 0.65, texto, transform=ax.transAxes, va="center", fontsize=11,
                color="white", linespacing=1.6,
                bbox=dict(fc="darkblue", ec="none", boxstyle="round,pad=0.6", alpha=0.85))

        # Posiciona os nomes com base no tamanho renderizado, evitando sobreposição.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ocupados = [Bbox.from_bounds(px - 5, py - 5, 10, 10)
                    for px, py in ax.transData.transform(np.column_stack([x, y]))]
        deslocamentos = [(6, 7), (6, -13), (-6, 7), (-6, -13), (6, 20),
                         (6, -26), (-6, 20), (-6, -26), (24, 7), (-24, 7)]
        for _, linha in dados.iterrows():
            anotacao = ax.annotate(linha["país"], (linha[coluna], linha["juro real"]),
                                  xytext=(6, 7), textcoords="offset points", fontsize=8,
                                  color="darkblue", fontweight="bold" if linha["país"] == "Brasil" else "normal")
            melhor, menor = None, float("inf")
            for dx, dy in deslocamentos:
                anotacao.set_position((dx, dy))
                anotacao.set_ha("left" if dx > 0 else "right")
                caixa = anotacao.get_window_extent(renderer).expanded(1.05, 1.15)
                custo = sum(caixa.overlaps(outro) for outro in ocupados)
                custo += 100 * (not ax.bbox.contains(caixa.x0, caixa.y0)
                                or not ax.bbox.contains(caixa.x1, caixa.y1))
                if custo < menor:
                    melhor, menor = (dx, dy), custo
                if custo == 0:
                    break
            anotacao.set_position(melhor)
            anotacao.set_ha("left" if melhor[0] > 0 else "right")
            ocupados.append(anotacao.get_window_extent(renderer).expanded(1.05, 1.15))

    fig.text(0.075, 0.105, "Fonte: FMI (WEO / Fiscal Monitor), BIS, ABS e BCRA.", fontsize=9)
    fig.text(0.80, 0.105, "Elaborado por: Fabricio Orlandin, CFP®", ha="right", fontsize=9)
    fig.text(0.075, 0.079,
             "Argentina: usa TAMAR privada efetiva anual como proxy dos juros, em lugar da taxa básica; aplica-se a mesma fórmula de juro real.", fontsize=9)
    fig.text(0.075, 0.053,
             "Déficit primário médio: média dos anos disponíveis na janela de dez anos; déficit = −saldo primário.", fontsize=9)
    fig.text(0.075, 0.027,
             "Juro real = 100 × [(1 + juros/100) / (1 + inflação em 12 meses/100) − 1]. Regressões descritivas entre países, incluindo a Argentina.",
             fontsize=9, color="dimgray")
    fig.savefig(arquivo, dpi=180, facecolor="white")
    return fig, pd.DataFrame(estatisticas)


# %% Matriz fiscal — quatro quadrantes
def escalar_areas_juros(juros, area_min=45.0, area_max=450.0):
    """Área em pontos², linear entre os juros mínimo e máximo disponíveis.

    Preserva a ordem inclusive para juros negativos. Ausentes ficam NaN;
    quando todos os juros são iguais, usa a área intermediária.
    """
    juros = pd.to_numeric(juros, errors="coerce").replace([np.inf, -np.inf], np.nan)
    minimo, maximo = juros.min(), juros.max()
    if minimo == maximo:
        return juros.where(juros.isna(), (area_min + area_max) / 2)
    return area_min + (juros - minimo) / (maximo - minimo) * (area_max - area_min)


def gerar_grafico_matriz(df, arquivo=None):
    """Dispersão nos escores Z originais, sem deslocar os pontos.

    Os sinais comparam cada indicador à média do grupo, não a zero % do PIB.
    Valores sobre os eixos estão na média e não pertencem a um quadrante.
    """
    from matplotlib.transforms import Bbox
    from matplotlib.patches import Rectangle

    arquivo = arquivo or Path(__file__).with_name("juros_divida_matriz.png")
    xcol, ycol = "z_divida_pib", "z_deficit_primario"
    dados = df[["país", xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
    dados["juro real"] = pd.to_numeric(
        df.reindex(dados.index).get("juro real", pd.Series(index=dados.index, dtype=float)),
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    dados["area"] = escalar_areas_juros(dados["juro real"])
    if dados.empty:
        raise ValueError("Não há países com os dois indicadores normalizados disponíveis.")
    omitidos = len(df) - len(dados)
    limite = max(1.0, float(dados[[xcol, ycol]].abs().max().max()) * 1.30)
    # Faixas exclusivas para cabeçalho, legenda, eixos e notas de rodapé.
    fig, ax = plt.subplots(figsize=(14, 14), layout="none")
    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.29, top=0.80)
    ano = df.attrs.get("ano_fiscal", ANO_REFERENCIA)
    fig.suptitle("Matriz fiscal | G20", x=0.10, y=0.975, ha="left", va="top",
                 fontsize=23, fontweight="bold", color="darkblue")
    fig.text(0.10, 0.925,
             f"Dívida/PIB: {ano}  •  Déficit primário médio: {ano - 9}–{ano}  •  {len(dados)} países",
             fontsize=11, color="dimgray", va="top")
    quadrantes = [
        (-limite, 0, "#fff0d1", "I. Dívida < média\ne Déficit > média", 0.02, 0.98, "left", "top"),
        (0, 0, "#f7dddd", "II. Dívida > média\ne Déficit > média", 0.98, 0.98, "right", "top"),
        (0, -limite, "#e0eaf6", "III. Dívida > média\ne Déficit < média", 0.98, 0.02, "right", "bottom"),
        (-limite, -limite, "#dff0e5", "IV. Dívida < média\ne Déficit < média", 0.02, 0.02, "left", "bottom"),
    ]
    titulos = []
    for x, y, cor, titulo, tx, ty, ha, va in quadrantes:
        ax.add_patch(Rectangle((x, y), limite, limite, color=cor, zorder=0))
        titulos.append(ax.text(tx, ty, titulo, transform=ax.transAxes,
                               ha=ha, va=va, fontsize=10, color="#344054",
                               fontweight="bold"))
    ax.axhline(0, color="#475467", linewidth=1.4)
    ax.axvline(0, color="#475467", linewidth=1.4)
    ax.set(xlim=(-limite, limite), ylim=(-limite, limite),
           xlabel="Dívida / PIB normalizada (escore Z)",
           ylabel="Déficit primário médio normalizado (escore Z)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18, zorder=0)
    com_juros = dados.loc[dados["area"].notna()]
    sem_juros = dados.loc[dados["area"].isna()]
    from matplotlib.colors import to_rgba

    cores = [to_rgba("#f4c542", 0.95) if pais == "Brasil"
             else to_rgba("darkblue", 0.35) for pais in com_juros["país"]]
    ax.scatter(com_juros[xcol], com_juros[ycol], s=com_juros["area"], c=cores,
               edgecolor=to_rgba("#667085", 0.5), linewidth=0.8, zorder=3)
    ax.scatter(sem_juros[xcol], sem_juros[ycol], s=55, marker="x", color="#667085", zorder=3)
    periodo = df.attrs.get("periodo_comum", "não informado")
    fig.text(0.10, 0.89, f"Juro real (% a.a.) · {periodo}",
             fontsize=11, color="dimgray", va="top")
    if not sem_juros.empty:
        fig.text(0.10, 0.86, "× País sem juro real disponível", fontsize=9, color="dimgray", va="top")
    fig.text(0.50, 0.23,
             f"Dívida/PIB de {ano}; déficit primário = média de {ano - 9} a {ano}.",
             fontsize=10, fontweight="bold", color="darkblue", va="top", ha="center")
    fig.text(0.50, 0.205, "Quanto maior o círculo, maior o juro real do país.",
             fontsize=10, fontweight="bold", color="darkblue", va="top", ha="center")

    # Ajusta apenas os rótulos: as coordenadas dos países permanecem exatas.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ocupados = [t.get_window_extent(renderer) for t in titulos]
    for (px, py), area in zip(ax.transData.transform(dados[[xcol, ycol]].to_numpy()),
                              dados["area"].fillna(55)):
        raio = np.sqrt(area) / 2 * fig.dpi / 72 + 3
        ocupados.append(Bbox.from_bounds(px - raio, py - raio, 2 * raio, 2 * raio))
    for _, linha in dados.iterrows():
        rotulo = ax.annotate(linha["país"], (linha[xcol], linha[ycol]),
                             xytext=(8, 8), textcoords="offset points", fontsize=9,
                             color="darkblue", zorder=10,
                             bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                                       boxstyle="round,pad=0.15"),
                             fontweight="bold" if linha["país"] == "Brasil" else "normal",
                             arrowprops=dict(arrowstyle="-", color="#667085", lw=0.5))
        melhor, menor = (8, 8), float("inf")
        for distancia in (8, 20, 34, 50, 70):
            for dx, dy in ((distancia, distancia), (-distancia, distancia),
                           (distancia, -distancia), (-distancia, -distancia)):
                rotulo.set_position((dx, dy))
                rotulo.set_ha("left" if dx > 0 else "right")
                rotulo.update_positions(renderer)
                # Extensão do texto sem incluir a linha que aponta para o país.
                caixa = plt.Text.get_window_extent(rotulo, renderer).expanded(1.05, 1.15)
                custo = sum(caixa.overlaps(outro) for outro in ocupados)
                custo += 100 * (not ax.bbox.contains(caixa.x0, caixa.y0)
                                or not ax.bbox.contains(caixa.x1, caixa.y1))
                if custo < menor:
                    melhor, menor = (dx, dy), custo
            if menor == 0:
                break
        rotulo.set_position(melhor)
        rotulo.set_ha("left" if melhor[0] > 0 else "right")
        rotulo.update_positions(renderer)
        ocupados.append(plt.Text.get_window_extent(rotulo, renderer).expanded(1.05, 1.15))
    fig.text(0.50, 0.18, "Z = (valor − média dos países) / desvio-padrão populacional (ddof=0).", fontsize=10, va="top", ha="center")
    fig.text(0.50, 0.155, "(+) Acima da média  •  (−) Abaixo da média  •  Zero: média do grupo  •  Brasil em amarelo", fontsize=10, va="top", ha="center")
    fig.text(0.50, 0.13, "Déficit = −saldo primário. Z negativo não implica superávit nem dívida negativa, apenas que está abaixo da média.", fontsize=10, va="top", ha="center")
    fig.text(0.50, 0.10, "Fontes: FMI (WEO / Fiscal Monitor), BIS, ABS e BCRA. Argentina: juros via TAMAR (proxy).", fontsize=9, color="dimgray", va="top", ha="center")
    fig.text(0.70, 0.035, "Elaborado por: Fabricio Orlandin, CFP®", ha="right", fontsize=9, va="top")
    fig.savefig(arquivo, dpi=180, facecolor="white")
    return fig


# %% Executar (também disponível como célula no editor)
if __name__ == "__main__":
    # Se os juros falharem, mantém os países na matriz com marcadores em X.
    df = montar_dataframe(incluir_juros=False)
    print(df.drop(columns="juro real").to_string(index=False, float_format=lambda valor: f"{valor:.2f}"))
    try:
        dados_juros, df_fontes = consultar_juros_inflacao(g20)
        metadados = df.attrs.copy()
        df = df.drop(columns="juro real").merge(
            dados_juros[["pais", "juro_real_ex_post"]].rename(
                columns={"pais": "país", "juro_real_ex_post": "juro real"}),
            on="país", how="left", validate="one_to_one",
        )
        df.attrs = metadados
        df.attrs["periodo_comum"] = dados_juros.attrs["periodo_comum"]
        df.attrs["fontes_juros_inflacao"] = df_fontes.to_dict("records")
        fig_regressoes, df_regressoes = gerar_graficos_regressao(df)
        print(df_regressoes.to_string(index=False))
        print("Imagem salva em juros_divida_matriz_regressoes.png")
    except (requests.RequestException, ValueError, KeyError) as erro:
        warnings.warn(f"Consulta de juros ou regressões indisponíveis: {erro}", stacklevel=1)
    fig_matriz = gerar_grafico_matriz(df)
    print("Imagem salva em juros_divida_matriz.png")
    plt.show()


