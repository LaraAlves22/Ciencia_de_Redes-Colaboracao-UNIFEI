#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from scipy.special import zeta

import modelagem as mv
from ccdf_modelos import MODELOS, ajustar_pl, graus_das_replicas
from aderencia_modelos import METRICAS, erros_replica

ROT = {
    "ER": "ER", "WS": "WS", "BA": "BA", "HK": "HK", "CM": "Config.",
    "LFR_0.1": "LFR\n$\\mu$=0,1", "LFR_0.3": "LFR\n$\\mu$=0,3"
}
COLUNAS_ERRO = [m for m, _, _ in METRICAS] + ["ks"]


def carregar(a):
    resumo = json.load(open(os.path.join(a.modelagem, "resumo.json"), encoding="utf-8"))
    G, _ = mv.carregar_baseline(a.data)
    b = resumo["baseline"]
    base = {
        "d": b["d"],
        "diam": b["diam"],
        "cc_med": b["cc_med"],
        "cc_glob": b["cc_glob"],
        "Q": b["Q_medio"],
        "kmax": b["kmax"],
    }
    graus_base = np.array([d for _, d in G.degree()])
    return resumo, G, base, graus_base


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    m = len(p)
    aj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(o):
        run = max(run, (m - i) * p[idx])
        aj[idx] = min(run, 1.0)
    return aj


def cliff(x, y):
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2 * u / (len(x) * len(y)) - 1


def etapa_boxplots(a, resumo, base, out):
    nomes = [
        ("d", "$\\langle d \\rangle$"),
        ("diam", "Diâmetro"),
        ("cc_med", "CA médio"),
        ("cc_glob", "CA global"),
        ("Q", "Modularidade $Q$"),
        ("kmax", "$k_{\\max}$"),
    ]
    dados = {m: pd.read_csv(os.path.join(a.modelagem, f"replicas_{m}.csv")) for m in MODELOS}
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, (col, tit) in zip(axs.ravel(), nomes):
        ax.boxplot([dados[m][col].values for m in MODELOS],
                   showmeans=True, patch_artist=True,
                   boxprops=dict(facecolor="#dfe9f3"), medianprops=dict(color="k"),
                   meanprops=dict(marker="d", markerfacecolor="w", markeredgecolor="k", markersize=4))
        ax.set_xticks(range(1, len(MODELOS) + 1))
        ax.set_xticklabels([ROT[m] for m in MODELOS])
        ax.axhline(base[col], color="#d62728", ls="--", lw=1.4, label="Baseline")
        ax.set_title(tit, fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=.3)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Variabilidade das réplicas por modelo (linha tracejada = baseline)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "boxplots_modelos.png"), dpi=250)
    fig.savefig(os.path.join(out, "boxplots_modelos.pdf"))
    plt.close(fig)
    print("boxplots ok", flush=True)


def etapa_testes(a, resumo, G, base, graus_base, out):
    N, L = G.number_of_nodes(), G.number_of_edges()
    E = {}
    for nome in MODELOS:
        df = pd.read_csv(os.path.join(a.modelagem, f"replicas_{nome}.csv"))
        graus = graus_das_replicas(nome, G, N, L, resumo, a.reps)
        n = min(len(df), len(graus), a.reps)
        regs = []
        for i in range(n):
            if int(df.loc[i, "kmax"]) != int(graus[i].max()):
                print(f"  AVISO: {nome} réplica {i} não confere com o CSV (kmax)")
            regs.append(erros_replica(df.loc[i], graus[i], base, graus_base))
        E[nome] = pd.DataFrame(regs)
        print(f"  {nome}: {n} réplicas", flush=True)

    pd.concat({k: v for k, v in E.items()}, names=["modelo", "replica"]).to_csv(
        os.path.join(out, "erros_por_replica.csv"), encoding="utf-8-sig")

    resumo_t = {"n_replicas": {k: len(v) for k, v in E.items()}}
    H, p_kw = kruskal(*[E[m]["media"].values for m in MODELOS])
    resumo_t["kruskal_wallis_erro_medio"] = {"H": float(H), "p": float(p_kw)}

    linhas = []
    for m in MODELOS:
        if m == "HK":
            continue
        x, y = E["HK"]["media"].values, E[m]["media"].values
        u = mannwhitneyu(x, y, alternative="two-sided")
        linhas.append({
            "comparacao": f"HK x {m}",
            "mediana_HK": np.median(x),
            "mediana_outro": np.median(y),
            "U": u.statistic,
            "p": u.pvalue,
            "cliff_delta": cliff(x, y),
        })
    T = pd.DataFrame(linhas)
    T["p_holm"] = holm(T["p"].values)
    T.to_csv(os.path.join(out, "mannwhitney_HK_vs_demais.csv"), index=False)

    linhas = []
    for c in COLUNAS_ERRO:
        x, y = E["HK"][c].values, E["LFR_0.1"][c].values
        u = mannwhitneyu(x, y, alternative="two-sided")
        linhas.append({
            "componente": c,
            "media_erro_HK": x.mean(),
            "media_erro_LFR": y.mean(),
            "p": u.pvalue,
            "cliff_delta": cliff(x, y),
            "menor_erro": "HK" if x.mean() < y.mean() else "LFR",
        })
    C = pd.DataFrame(linhas)
    C["p_holm"] = holm(C["p"].values)
    C.to_csv(os.path.join(out, "mannwhitney_HK_vs_LFR_por_componente.csv"), index=False)

    print("\nKruskal-Wallis: H=%.1f p=%.2e" % (H, p_kw))
    print(T.round(4).to_string(index=False))
    print(C.round(4).to_string(index=False))
    json.dump(resumo_t, open(os.path.join(out, "resumo_testes.json"), "w"), indent=1)


def etapa_varredura(a, resumo, G, base, graus_base, out):
    N, L = G.number_of_nodes(), G.number_of_edges()
    lp = resumo["lfr_parametros"]
    linhas = []
    t0 = time.time()

    def registrar(familia, par, gerador, seeds, tol=None):
        ok, rej, fal = 0, 0, 0
        for s in seeds:
            if ok == a.reps:
                break
            try:
                g = gerador(s)
            except Exception:
                fal += 1
                continue
            if tol is not None and abs(g.number_of_edges() - L) / L > tol:
                rej += 1
                continue
            row = mv.metricas(g, seed_louvain=s)
            graus = np.array([d for _, d in g.degree()])
            e = erros_replica(row, graus, base, graus_base)
            linhas.append({
                "familia": familia, "param": par, "seed": s, "L": row["L"], "d": row["d"],
                "diam": row["diam"], "cc_med": row["cc_med"], "cc_glob": row["cc_glob"],
                "Q": row["Q"], "kmax": row["kmax"], **{f"erro_{k}": v for k, v in e.items()}
            })
            ok += 1
        print(f"  {familia} {par}: ok={ok} rejeitadas={rej} falhas={fal} ({time.time()-t0:.0f}s)", flush=True)

    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        registrar("HK", p, lambda s, p=p: mv.gerar_hk(N, L, p, s), range(a.reps))
    for mu in (0.1, 0.2, 0.3, 0.4):
        registrar("LFR", mu, lambda s, mu=mu: mv.gerar_lfr(N, mu, s, lp["tau1"], lp["tau2"],
                                                          lp["min_community"], lp["max_community"]),
                  range(400), tol=mv.LFR_TOL)

    D = pd.DataFrame(linhas)
    D.to_csv(os.path.join(out, "varredura_replicas.csv"), index=False)
    R = D.groupby(["familia", "param"]).agg(
        n=("seed", "count"), L=("L", "mean"), d=("d", "mean"), diam=("diam", "mean"),
        cc_med=("cc_med", "mean"), cc_med_dp=("cc_med", "std"), cc_glob=("cc_glob", "mean"),
        Q=("Q", "mean"), Q_dp=("Q", "std"), kmax=("kmax", "mean"),
        erro_medio=("erro_media", "mean"), erro_medio_dp=("erro_media", "std")).reset_index()
    R.to_csv(os.path.join(out, "varredura_resumo.csv"), index=False)
    print("\n", R.round(3).to_string(index=False))

    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    for lin, (fam, xl) in enumerate([("HK", "p (prob. de fechamento de tríade)"), ("LFR", "$\\mu$ (fração inter-comunidades)")]):
        S = R[R.familia == fam]
        for ax, (col, dp, tit, ref) in zip(axs[lin], [("cc_med", "cc_med_dp", "CA médio", base["cc_med"]),
                                                      ("Q", "Q_dp", "Modularidade $Q$", base["Q"]),
                                                      ("erro_medio", "erro_medio_dp", "Erro médio vs baseline", None)]):
            ax.errorbar(S.param, S[col], yerr=S[dp], marker="o", capsize=3, color="#1f4e79")
            if ref is not None:
                ax.axhline(ref, color="#d62728", ls="--", label="Baseline")
                ax.legend(fontsize=8)
            ax.set_xlabel(xl, fontsize=9)
            ax.set_title(f"{fam}: {tit}", fontsize=10)
            ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "varredura_HK_LFR.png"), dpi=250)
    fig.savefig(os.path.join(out, "varredura_HK_LFR.pdf"))
    plt.close(fig)


def amostra_sintetica(rng, corpo, n_cauda, kmin, alfa, n, kmax=20000):
    dom = np.arange(kmin, kmax + 1, dtype=float)
    pmf = dom ** -alfa
    pmf /= pmf.sum()
    cauda = rng.choice(dom, size=n, p=pmf)
    corpo_s = rng.choice(corpo, size=n) if len(corpo) else cauda
    esc = rng.random(n) < n_cauda / n
    return np.where(esc, cauda, corpo_s)


def etapa_bootstrap(a, G, out):
    graus = np.array([d for _, d in G.degree()], float)
    kmin, alfa, d_obs, n_cauda = ajustar_pl(graus)
    n = len(graus)
    corpo = graus[graus < kmin]
    rng = np.random.default_rng(12345)
    t0 = time.time()
    ds, ks, als, falhas = [], [], [], 0
    for b in range(a.boot):
        amostra = amostra_sintetica(rng, corpo, n_cauda, kmin, alfa, n)
        fit = ajustar_pl(amostra)
        if fit is None:
            falhas += 1
            continue
        ks.append(fit[0])
        als.append(fit[1])
        ds.append(fit[2])
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b+1}/{a.boot} ({time.time()-t0:.0f}s)", flush=True)

    ds = np.array(ds)
    p = float((ds >= d_obs).mean())
    res = {
        "kmin": kmin, "alfa": alfa, "D_obs": d_obs, "n_cauda": n_cauda, "n_nos": n,
        "n_boot": len(ds), "falhas": falhas, "p_valor": p,
        "D_sint_mediana": float(np.median(ds)), "D_sint_p95": float(np.percentile(ds, 95)),
        "alfa_sint_media": float(np.mean(als)), "kmin_sint_mediano": float(np.median(ks)),
        "interpretacao": ("p >= 0,1: a lei de potência NÃO é rejeitada" if p >= 0.1 else "p < 0,1: a lei de potência é rejeitada")
    }
    json.dump(res, open(os.path.join(out, "bootstrap_lei_de_potencia.json"), "w"), indent=1, ensure_ascii=False)
    pd.DataFrame({"D_sintetico": ds}).to_csv(os.path.join(out, "bootstrap_D_sinteticos.csv"), index=False)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.hist(ds, bins=40, color="#9ab7d3", edgecolor="w")
    ax.axvline(d_obs, color="#d62728", lw=2, label=f"$D_{{obs}}$={d_obs:.3f}  (p={p:.3f})")
    ax.set_xlabel("Distância KS das amostras sintéticas")
    ax.set_ylabel("Frequência")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bootstrap_lei_de_potencia.png"), dpi=250)
    plt.close(fig)
    print(json.dumps(res, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dados_de_extracao")
    ap.add_argument("--modelagem", default="resultados_modelagem")
    ap.add_argument("--out", default="resultados_at3_complementos")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--etapas", default="boxplots,testes,varredura,bootstrap")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    etapas = [e.strip() for e in a.etapas.split(",")]
    resumo, G, base, graus_base = carregar(a)
    print("Baseline:", {k: round(v, 4) for k, v in base.items()}, flush=True)

    if "boxplots" in etapas:
        etapa_boxplots(a, resumo, base, a.out)
    if "testes" in etapas:
        etapa_testes(a, resumo, G, base, graus_base, a.out)
    if "varredura" in etapas:
        etapa_varredura(a, resumo, G, base, graus_base, a.out)
    if "bootstrap" in etapas:
        etapa_bootstrap(a, G, a.out)

    print(f"\nConcluído. Arquivos salvos em: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()