#%%Bibliotecas
from bcb import sgs
import fredapi as fred
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
# %% Selic
selic0 = sgs.get({'SELIC': 432}, start='2010-01-01', end='2014-12-31')
selic1 = sgs.get({'SELIC': 432}, start='2015-01-01', end='2019-12-31')
selic2 = sgs.get({'SELIC': 432}, start='2020-01-01', end='2026-09-18')
df = (
    pd.concat([selic0, selic1, selic2])
    .rename(columns={'SELIC': 'Selic'})
)
# %% IPCA 12m
df["IPCA 12m"] =sgs.get({'IPCA 12m': 13522},
                        start='2010-01-01', end='2026-09-18').reindex(
                            df.index, method='ffill'
                        )
# %% IPCA 12m Média dos núcleos
nucleos = [11427, 16121, 27838, 27839, 11426, 4466, 16122, 28751, 28750]
ipca_nucleo = sgs.get(nucleos,
                      start='2010-01-01', end='2026-09-18')
ipca_nucleo["Média"] = ipca_nucleo.mean(axis=1)
ipca_nucleo["Média Núcleos 12m"] = (ipca_nucleo["Média"].rolling(window=12).apply(
    lambda x: np.prod(1 + x / 100) - 1) * 100).astype(float).round(2)
ipca_nucleo = ipca_nucleo[["Média Núcleos 12m"]].reindex(df.index, method='ffill')
df = pd.concat([df, ipca_nucleo], axis=1).dropna()
# %% FRED Rate max range
from config_local import FRED_API_KEY
fred_api = FRED_API_KEY
fred_rate = fred.Fred(api_key=fred_api).get_series('DFEDTARU', observation_start='2010-01-01', observation_end='2026-09-18')
df["FRED Rate"] = fred_rate.reindex(df.index, method='ffill')
# %% CPI 12m
cpi = fred.Fred(api_key=fred_api).get_series('CPIAUCSL', observation_start='2010-01-01', observation_end='2026-09-18')
cpi_12m = (cpi.pct_change(periods=12) * 100).astype(float).round(2)
df["CPI 12m"] = cpi_12m.reindex(df.index, method='ffill')
# %% PCE 12m
pce = fred.Fred(api_key=fred_api).get_series('PCEPILFE', observation_start='2010-01-01', observation_end='2026-09-18')
pce_12m = (pce.pct_change(periods=12) * 100).astype(float).round(2)
df["PCE 12m"] = pce_12m.reindex(df.index, method='ffill')
#%% Juros Eurozone
ecbmrrfr = fred.Fred(api_key=fred_api).get_series(
    'ECBMRRFR', observation_start='2010-01-01',
    observation_end='2026-09-18')
ecbdfr = fred.Fred(api_key=fred_api).get_series(
    'ECBDFR', observation_start='2010-01-01',
    observation_end='2026-09-18')
ecbmlfr = fred.Fred(api_key=fred_api).get_series(
    'ECBMLFR', observation_start='2010-01-01',
    observation_end='2026-09-18')
df["ECB MRRFR"] = ecbmrrfr.reindex(df.index, method='ffill')
df["ECB DFR"] = ecbdfr.reindex(df.index, method='ffill')
df["ECB MLFR"] = ecbmlfr.reindex(df.index, method='ffill')
inflacao_eurozone = fred.Fred(api_key=fred_api).get_series(
    'CP0000EZ19M086NEST', observation_start='2009-01-01',
    observation_end='2026-09-18'
)
nucleo_inflacao_eurozone = fred.Fred(api_key=fred_api).get_series(
    'TOTNRGFOODEA20MI15XM', observation_start='2009-01-01',
    observation_end='2026-09-18'
)
# Calcular antes de alinhar ao calendário diário: 12 períodos = 12 meses.
inflacao_eurozone_12m = (
    inflacao_eurozone.pct_change(periods=12, fill_method=None) * 100
).round(2)
nucleo_inflacao_eurozone_12m = (
    nucleo_inflacao_eurozone.pct_change(periods=12, fill_method=None) * 100
).round(2)
df["Inflacao Eurozone"] = inflacao_eurozone_12m.reindex(df.index, method='ffill')
df["Nucleo Inflacao Eurozone"] = nucleo_inflacao_eurozone_12m.reindex(
    df.index, method='ffill'
)
#%%visualiza df
df.columns = ["Selic",
              "IPCA 12m",
              "Média Núcleos 12m",
              "FRED Rate",
              "CPI 12m",
              "PCE 12m",
              "ECB MRRFR",
              "ECB DFR",
              "ECB MLFR",
              "Inflacao Eurozone",
              "Nucleo Inflacao Eurozone"
              ]
df.tail()
# %% Gráfico — Brasil, Estados Unidos e Zona do Euro

def rotular_ultimos_valores(ax, series):
    """Identifica cada série e seu valor em boxes externos sem sobreposição."""
    from textwrap import fill

    itens = []
    renderer = fig.canvas.get_renderer()
    for coluna, nome, cor in series:
        validos = df[coluna].dropna()
        if validos.empty:
            continue
        valor = validos.iloc[-1]
        y = ax.transData.transform((0, valor))[1]
        texto = fill(nome.replace('\n', ' '), width=38)
        texto += "\n" + f"{valor:.2f}%".replace('.', ',')
        anotacao = ax.annotate(
            texto, xy=(validos.index[-1], valor), xycoords='data',
            xytext=(1.035, 0.5), textcoords='axes fraction',
            ha='left', va='center', fontsize=8, linespacing=1.25,
            color='black' if cor == 'lightblue' else 'white',
            bbox=dict(fc=cor, ec='none', boxstyle='round,pad=0.4', alpha=0.85),
            arrowprops=dict(arrowstyle='-', color=cor, lw=0.9,
                            relpos=(0, 0.5)),
            annotation_clip=False,
        )
        anotacao.update_bbox_position_size(renderer)
        altura = anotacao.get_bbox_patch().get_window_extent(renderer).height
        itens.append((y, altura, anotacao))
    itens.sort(key=lambda item: item[0])
    if not itens:
        return
    # Medir os boxes para considerar nomes com diferentes números de linhas.
    margem = 4 * fig.dpi / 72
    intervalo = 7 * fig.dpi / 72
    inferior, superior = ax.bbox.y0 + margem, ax.bbox.y1 - margem
    posicoes = []
    for i, (y, altura, _) in enumerate(itens):
        minimo = inferior + altura / 2
        if i:
            minimo = max(minimo, posicoes[-1] + itens[i - 1][1] / 2
                         + intervalo + altura / 2)
        posicoes.append(max(y, minimo))
    posicoes[-1] = min(posicoes[-1], superior - itens[-1][1] / 2)
    for i in range(len(itens) - 2, -1, -1):
        posicoes[i] = min(posicoes[i], posicoes[i + 1] - itens[i + 1][1] / 2
                          - intervalo - itens[i][1] / 2)
    for (_, _, anotacao), y in zip(itens, posicoes):
        anotacao.set_position((1.035, (y - ax.bbox.y0) / ax.bbox.height))


paineis = [
    ("Brasil", [
        ("Selic", "Juros", "black"),
        ("IPCA 12m", "Inflação", "darkblue"),
        ("Média Núcleos 12m", "Núcleo (média dos núcleos)", "lightblue"),
    ], [(4.5, "Teto: 4,5%", "gray"), (3, "Meta: 3%", "black"),
        (1.5, "Piso: 1,5%", "gray")]),
    ("EUA", [
        ("FRED Rate", "Juros", "black"),
        ("CPI 12m", "CPI", "darkblue"),
        ("PCE 12m", "PCE", "lightblue"),
    ], [(2, "Meta: 2%", "black")]),
    ("Zona do Euro", [
        ("ECB DFR", "Juros", "black"),
        ("Inflacao Eurozone", "Inflação", "darkblue"),
        ("Nucleo Inflacao Eurozone", "Núcleo (ex-energia, alimentos, álcool e tabaco)", "lightblue"),
    ], [(2, "Meta: 2%", "black")]),
]
fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(17, 12))
# Reservar espaço para cabeçalhos, rótulos externos e rodapé.
fig.subplots_adjust(left=0.06, right=0.73, bottom=0.08, top=0.92, hspace=0.55)
for ax, (titulo, series, metas) in zip(axes, paineis):
    for coluna, legenda, cor in series:
        ax.plot(df[coluna], color=cor, label=legenda, alpha=0.7)
    linhas_metas = [ax.axhline(valor, color=cor, linestyle='--', lw=1.2, zorder=0)
                    for valor, _, cor in metas]
    ax.set_title(titulo, loc='left', fontweight='bold', fontsize=14,
                 color='darkblue', pad=36)
    # As metas permanecem acima; os nomes das séries ficam nos boxes.
    ax.legend(linhas_metas, [texto for _, texto, _ in metas], loc='lower left',
              bbox_to_anchor=(0, 1.01), ncol=3, fontsize=8,
              frameon=False, borderaxespad=0)
    ax.margins(x=0.015, y=0.1)

# O posicionamento depende das dimensões finais dos painéis.
fig.canvas.draw()
for ax, (_, series, _) in zip(axes, paineis):
    rotular_ultimos_valores(ax, series)
fig.text(0.1, 0.02, "Fonte: IBGE / Banco Central do Brasil (BCB) / FED Saint Louis",
         ha='left', va='bottom', fontsize=9)
fig.text(0.75, 0.02, "Elaborado por: Fabricio Orlandin, CFP®",
         ha='right', va='bottom', fontsize=9)
plt.show()

# %%
