#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

import modelagem as mv
from ccdf_modelos import MODELOS, ROTULO, graus_das_replicas

METRICAS = [
    ("d", "$\\langle d \\rangle$", "⟨d⟩"),
    ("diam", "Diâm.", "Diâm."),
    ("cc_med", "CA médio", "CA médio"),
    ("cc_glob", "CA global", "CA global"),
    ("Q", "$Q$", "Q"),
    ("kmax", "$k_{\\max}$", "k_max"),
]
TETO = 1.0


def erros_replica(row, graus, base, graus_base):
    e = {}
    for m, _, _ in METRICAS:
        e[m] = min(abs(row[m] - base[m]) / abs(base[m]), TETO)
    e["ks"] = float(ks_2samp(graus_base, graus).statistic)
    e["media"] = float(np.mean(list(e.values())))
    return e


def vir(x, c=1):
    return f"{x:.{c}f}".replace(".", "{,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dados_de_extracao")
    ap.add_argument("--modelagem", default="resultados_modelagem")
    ap.add_argument("--out", default="resultados_aderencia")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    resumo = json.load(open(os.path.join(a.modelagem, "resumo.json"), encoding="utf-8"))
    G, _ = mv.carregar_baseline(a.data)
    N, L = G.number_of_nodes(), G.number_of_edges()
    graus_base = np.array([d for _, d in G.degree()])
    b = resumo["baseline"]
    base = {
        "d": b["d"],
        "diam": b["diam"],
        "cc_med": b["cc_med"],
        "cc_glob": b["cc_glob"],
        "Q": b["Q_medio"],
        "kmax": b["kmax"],
    }
    print("Baseline:", {k: round(v, 4) for k, v in base.items()})

    linhas, matriz = [], {}
    for nome in MODELOS:
        print(f"Processando {nome} ...", flush=True)
        df = pd.read_csv(os.path.join(a.modelagem, f"replicas_{nome}.csv"))
        graus = graus_das_replicas(nome, G, N, L, resumo, a.reps)
        n = min(len(df), len(graus), a.reps)
        regs = []
        for i in range(n):
            if int(df.loc[i, "kmax"]) != int(graus[i].max()):
                print(f"  AVISO: réplica {i} de {nome} não confere com o CSV (kmax)")
            regs.append(erros_replica(df.loc[i], graus[i], base, graus_base))
        E = pd.DataFrame(regs)
        matriz[nome] = E.mean()
        linhas.append({
            "modelo": nome,
            **{f"erro_{k}": E[k].mean() for k in E.columns},
            "media_desvio": E["media"].std(ddof=1) if n > 1 else 0.0,
            "n_replicas": n,
            "n_metricas_ate_10pct": int(sum(E[m].mean() <= 0.10 for m, _, _ in METRICAS)),
        })

    R = pd.DataFrame(linhas).set_index("modelo")
    R.to_csv(os.path.join(a.out, "aderencia_modelos.csv"), encoding="utf-8-sig")
    rank = R.sort_values("erro_media")
    print("\nRanking por distância média:")
    for nome, r in rank.iterrows():
        print(f"  {nome:<8} {100*r.erro_media:5.1f}% ± {100*r.media_desvio:4.1f}  "
              f"(propriedades com erro <= 10%: {int(r.n_metricas_ate_10pct)}/6)")

    colunas = [m for m, _, _ in METRICAS] + ["ks"]
    minimos = {c: R[f"erro_{c}"].min() for c in colunas}
    corpo = []
    for nome in MODELOS:
        r = R.loc[nome]
        cel = []
        for c in colunas:
            v = 100 * r[f"erro_{c}"]
            t = vir(v, 1)
            cel.append(f"\\textbf{{{t}}}" if np.isclose(r[f"erro_{c}"], minimos[c]) else t)
        m, dp = 100 * r.erro_media, 100 * r.media_desvio
        cel.append(f"${vir(m, 1)} \\pm {vir(dp, 1)}$")
        corpo.append(" & ".join([ROTULO[nome]] + cel) + " \\\\")

    cab = " & ".join(["\\textbf{Modelo}"] + [f"\\textbf{{{lat}}}" for _, lat, _ in METRICAS]
                     + ["\\textbf{$D_{KS}$}", "\\textbf{Média}"])
    tex = (
        "\\begin{table*}[t]\n\\centering\n\\footnotesize\n"
        "\\caption{Aderência quantitativa dos modelos à baseline: erro relativo (\\%) em cada "
        "propriedade, distância KS entre as distribuições de grau ($D_{KS}\\times100$) e média dos "
        "sete componentes ($\\pm$ desvio-padrão entre réplicas).}\n\\label{tab:aderencia}\n"
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n"
        + cab + " \\\\\n\\midrule\n" + "\n".join(corpo) +
        "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table*}\n"
    )
    with open(os.path.join(a.out, "tabela_aderencia.tex"), "w", encoding="utf-8") as f:
        f.write(tex)

    cols = [m for m, _, _ in METRICAS] + ["ks"]
    nomes_col = [pl for _, _, pl in METRICAS] + ["D_KS"]
    M = np.array([[100 * matriz[n][c] for c in cols] for n in MODELOS])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(nomes_col, fontsize=8)
    ax.set_yticks(range(len(MODELOS)))
    ax.set_yticklabels([ROTULO[n].replace("$\\mu$", "μ") for n in MODELOS], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 60 else "black")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Erro (%)", fontsize=8)
    ax.set_title("Distância dos modelos à baseline", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "aderencia_heatmap.png"), dpi=300)
    fig.savefig(os.path.join(a.out, "aderencia_heatmap.pdf"))

    print(f"\nConcluído. Arquivos salvos em: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()