#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.special import erfc, zeta

import modelagem as mv

MODELOS = ["ER", "WS", "BA", "HK", "CM", "LFR_0.1", "LFR_0.3"]
ROTULO = {
    "ER": "ER", "WS": "WS", "BA": "BA", "HK": "HK", "CM": "Configuração",
    "LFR_0.1": "LFR ($\\mu$=0,1)", "LFR_0.3": "LFR ($\\mu$=0,3)",
}
COR = {
    "ER": "#1f77b4", "WS": "#2ca02c", "BA": "#d62728", "HK": "#9467bd",
    "CM": "#8c564b", "LFR_0.1": "#ff7f0e", "LFR_0.3": "#bcbd22",
}
ESTILO = {"ER": "-", "WS": "--", "BA": "-.", "HK": "-", "CM": "--", "LFR_0.1": "-.", "LFR_0.3": ":"}
PAINEL_A = ["ER", "WS", "BA"]
PAINEL_B = ["CM", "HK", "LFR_0.1"]


def ccdf_em_grade(graus, grade):
    graus = np.asarray(graus)
    return np.array([(graus >= k).mean() for k in grade])


def alfa_mle(cauda, kmin):
    n = len(cauda)
    soma_ln = np.log(cauda).sum()
    f = lambda a: a * soma_ln + n * np.log(zeta(a, kmin))
    r = minimize_scalar(f, bounds=(1.01, 8.0), method="bounded")
    return float(r.x)


def ks_pl(cauda, kmin, alfa):
    xs = np.unique(cauda)
    emp = np.array([(cauda >= x).mean() for x in xs])
    mod = zeta(alfa, xs) / zeta(alfa, kmin)
    return float(np.max(np.abs(emp - mod)))


def ajustar_pl(graus, n_min_cauda=50):
    graus = np.asarray(graus, float)
    melhor = None
    for kmin in np.unique(graus):
        if kmin < 1:
            continue
        cauda = graus[graus >= kmin]
        if len(cauda) < n_min_cauda or len(np.unique(cauda)) < 4:
            continue
        a = alfa_mle(cauda, kmin)
        d = ks_pl(cauda, kmin, a)
        if melhor is None or d < melhor[2]:
            melhor = (int(kmin), a, d, int(len(cauda)))
    return melhor


def lr_lognormal(graus, kmin, alfa, xmax=100000):
    x = np.asarray(graus, float)
    x = x[x >= kmin]
    dom = np.arange(kmin, xmax + 1, dtype=float)
    ll_pl = -alfa * np.log(x) - np.log(zeta(alfa, kmin))

    def logpmf_ln(params):
        mu, sg = params
        sg = abs(sg) + 1e-6
        f = np.exp(-(np.log(dom) - mu) ** 2 / (2 * sg ** 2)) / dom
        z = f.sum()
        return -(np.log(x) - mu) ** 2 / (2 * sg ** 2) - np.log(x) - np.log(z)

    nll = lambda p: -logpmf_ln(p).sum()
    ini = [np.log(x).mean(), max(np.log(x).std(), 0.3)]
    r = minimize(nll, ini, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 600})
    ll_ln = logpmf_ln(r.x)
    d = ll_pl - ll_ln
    R = float(d.sum())
    sd = float(d.std(ddof=0) * np.sqrt(len(d)))
    p = float(erfc(abs(R) / (np.sqrt(2) * sd))) if sd > 0 else float("nan")
    return {"R": R, "p": p, "mu_ln": float(r.x[0]), "sigma_ln": float(abs(r.x[1]))}


def graus_das_replicas(nome, G, N, L, resumo, reps):
    sem = resumo[nome]["sementes"][:reps]
    lp = resumo["lfr_parametros"]
    out = []
    for s in sem:
        if nome == "ER":
            g = mv.gerar_er(N, L, s)
        elif nome == "WS":
            g = mv.gerar_ws(N, L, mv.WS_P, s)
        elif nome == "BA":
            g = mv.gerar_hk(N, L, 0.0, s)
        elif nome == "HK":
            g = mv.gerar_hk(N, L, mv.HK_P, s)
        elif nome == "CM":
            g = mv.gerar_config(G, s)
        else:
            mu = float(nome.split("_")[1])
            g = mv.gerar_lfr(N, mu, s, lp["tau1"], lp["tau2"], lp["min_community"], lp["max_community"])
        out.append(np.array([d for _, d in g.degree()]))
    return out


def fmt(x, c):
    return f"{x:.{c}f}".replace(".", "{,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dados_de_extracao")
    ap.add_argument("--resumo", default="resultados_modelagem/resumo.json")
    ap.add_argument("--out", default="resultados_ccdf")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    resumo = json.load(open(a.resumo, encoding="utf-8"))
    G, _ = mv.carregar_baseline(a.data)
    N, L = G.number_of_nodes(), G.number_of_edges()
    g_base = np.array([d for _, d in G.degree()])
    print(f"Baseline: N={N} L={L}, grau máx = {g_base.max()}")

    graus = {}
    for nome in MODELOS:
        print(f"Gerando réplicas de {nome} ...", flush=True)
        graus[nome] = graus_das_replicas(nome, G, N, L, resumo, a.reps)

    kmax = int(max(g_base.max(), max(max(d.max() for d in v) for v in graus.values())))
    grade = np.arange(1, kmax + 1)
    piso = 1.0 / N
    c_base = ccdf_em_grade(g_base, grade)

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8})
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, lista, titulo in ((axs[0], PAINEL_A, "(a) Modelos clássicos"),
                              (axs[1], PAINEL_B, "(b) Configuração, HK e LFR")):
        for nome in lista:
            M = np.array([ccdf_em_grade(d, grade) for d in graus[nome]])
            med = np.median(M, axis=0)
            lo, hi = M.min(axis=0), M.max(axis=0)
            vis = hi > 0
            ax.fill_between(grade[vis], np.maximum(lo[vis], piso), hi[vis],
                            color=COR[nome], alpha=0.25, linewidth=0)
            vm = med > 0
            ax.plot(grade[vm], med[vm], ESTILO[nome], color=COR[nome], lw=1.4, label=ROTULO[nome])
        vb = c_base > 0
        ax.plot(grade[vb], c_base[vb], "o", color="black", ms=2.8, label="Baseline", zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Grau $k$")
        ax.set_title(titulo, fontsize=9)
        ax.set_ylim(piso * 0.7, 1.3)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="lower left", frameon=True)
    axs[0].set_ylabel("$P(K \\geq k)$")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "ccdf_comparativo.png"), dpi=300)
    fig.savefig(os.path.join(a.out, "ccdf_comparativo.pdf"))

    linhas, resumo_ccdf = [], {}
    fit_b = ajustar_pl(g_base)
    if fit_b is None:
        raise SystemExit("Cauda da baseline insuficiente para ajuste.")
    kmin_b, alfa_b, d_b, n_b = fit_b
    lr = lr_lognormal(g_base, kmin_b, alfa_b)
    print(f"Baseline: kmin={kmin_b} alfa={alfa_b:.2f} D={d_b:.3f} n_cauda={n_b} | "
          f"LR (PL x log-normal): R={lr['R']:.2f} p={lr['p']:.3f}")
    resumo_ccdf["Baseline"] = {"kmin": kmin_b, "alfa": alfa_b, "D": d_b, "n_cauda": n_b, "LR_vs_lognormal": lr}
    linhas.append(["Baseline", kmin_b, alfa_b, 0.0, d_b, 0.0, n_b, 1])

    for nome in MODELOS:
        fits = [ajustar_pl(d) for d in graus[nome]]
        ok = [f for f in fits if f is not None]
        if not ok:
            linhas.append([nome, "", "", "", "", "", 0, 0])
            continue
        kmins = np.array([f[0] for f in ok])
        alfas = np.array([f[1] for f in ok])
        ds = np.array([f[2] for f in ok])
        ns = np.array([f[3] for f in ok])
        if alfas.mean() >= 7.5:
            linhas.append([nome, "", "", "", "", "", 0, 0])
            continue
        linhas.append([nome, float(np.median(kmins)), alfas.mean(), alfas.std(ddof=1) if len(ok) > 1 else 0.0,
                       ds.mean(), ds.std(ddof=1) if len(ok) > 1 else 0.0, float(ns.mean()), len(ok)])
        resumo_ccdf[nome] = {
            "kmin_mediana": float(np.median(kmins)),
            "alfa_media": float(alfas.mean()),
            "D_medio": float(ds.mean()),
            "n_ajustes": len(ok),
            "n_replicas": len(fits),
        }
        print(f"{nome:<8} kmin~{np.median(kmins):.0f} alfa={alfas.mean():.2f} D={ds.mean():.3f} "
              f"({len(ok)}/{len(fits)} convergiram)")

    with open(os.path.join(a.out, "ajuste_lei_de_potencia.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rede", "kmin", "alfa_media", "alfa_desvio", "D_medio", "D_desvio", "n_cauda", "n_ajustes"])
        w.writerows(linhas)

    with open(os.path.join(a.out, "resumo_ccdf.json"), "w", encoding="utf-8") as f:
        json.dump(resumo_ccdf, f, indent=1, default=float)

    corpo = []
    for l in linhas:
        nome, kmin, am, ad, dm, dd, nc, nf = l
        rot = "Baseline" if nome == "Baseline" else ROTULO[nome].replace("$\\mu$=", "$\\mu{=}$")
        if nf == 0:
            corpo.append(f"{rot} & --- & --- & --- & --- \\\\")
        elif nome == "Baseline":
            corpo.append(f"{rot} & {int(kmin)} & {fmt(am, 2)} & {fmt(dm, 3)} & {int(nc)} \\\\")
        else:
            corpo.append(f"{rot} & {fmt(kmin, 0)} & ${fmt(am, 2)} \\pm {fmt(ad, 2)}$ & "
                         f"${fmt(dm, 3)} \\pm {fmt(dd, 3)}$ & {int(round(nc))} \\\\")

    tex = (
        "\\begin{table}[htbp]\n\\centering\n\\footnotesize\n"
        "\\caption{Ajuste de lei de potência discreta à cauda da distribuição de grau "
        "(estimador MLE com $k_{\\min}$ otimizado via KS). "
        "$\\alpha$: expoente; $D$: estatística KS; $n$: nós na cauda. Média $\\pm$ desvio-padrão.}\n"
        "\\label{tab:ajuste_pl}\n\\begin{tabular}{lrrrr}\n\\toprule\n"
        "\\textbf{Rede} & $k_{\\min}$ & $\\alpha$ & $D$ & $n$ \\\\\n\\midrule\n"
        + "\n".join(corpo) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    with open(os.path.join(a.out, "tabela_ajuste.tex"), "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"\nConcluído. Arquivos salvos em: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()