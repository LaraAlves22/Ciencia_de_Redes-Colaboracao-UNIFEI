#!/usr/bin/env python3
import argparse
import json
import os
import platform
import random
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd
import scipy
from networkx.algorithms.community import louvain_communities, modularity
from networkx.generators.community import LFR_benchmark_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

WS_P = 0.1
HK_P = 1.0
M0 = 3
LFR_MIN_DEG = 2
LFR_MAX_DEG = 114
LFR_MUS = (0.1, 0.3)
LFR_TOL = 0.03
N_SWAPS_FACTOR = 10
LOUVAIN_SEEDS_BASELINE = 20


def carregar_baseline(data_dir):
    nodes = pd.read_csv(os.path.join(data_dir, "nodes.csv"), encoding="utf-8-sig")
    edges = pd.read_csv(os.path.join(data_dir, "edges.csv"), encoding="utf-8-sig")
    G = nx.Graph()
    G.add_nodes_from(nodes["id"])
    G.add_edges_from(zip(edges["source"], edges["target"]))
    gc = max(nx.connected_components(G), key=len)
    H = nx.convert_node_labels_to_integers(G.subgraph(gc).copy(), label_attribute="oid")
    inst = dict(zip(nodes["id"], nodes["instituto"]))
    institutos = {n: inst.get(d["oid"], "Não identificado") for n, d in H.nodes(data=True)}
    return H, institutos


def q_louvain(g, seed):
    com = louvain_communities(g, seed=seed)
    return modularity(g, com), len(com)


def metricas(g, seed_louvain=0):
    n, l = g.number_of_nodes(), g.number_of_edges()
    A = nx.to_scipy_sparse_array(g, format="csr")
    n_comp, lab = connected_components(A, directed=False)
    big = np.argmax(np.bincount(lab))
    idx = np.where(lab == big)[0]
    sub = A[idx][:, idx]
    D = shortest_path(csr_matrix(sub), unweighted=True, directed=False)
    m = len(idx)
    q, ncom = q_louvain(g, seed_louvain)
    return {
        "N": n, "L": l, "k": 2 * l / n, "comp": int(n_comp),
        "d": float(D.sum() / (m * (m - 1))), "diam": int(D.max()),
        "cc_med": nx.average_clustering(g), "cc_glob": nx.transitivity(g),
        "kmax": max(dict(g.degree()).values()),
        "Q": q, "n_com": ncom,
    }


def nmi(a, b):
    ct = pd.crosstab(pd.Series(a, name="a"), pd.Series(b, name="b")).values.astype(float)
    n = ct.sum()
    pij = ct / n
    pi = pij.sum(1, keepdims=True)
    pj = pij.sum(0, keepdims=True)
    nz = pij > 0
    mi = (pij[nz] * np.log(pij[nz] / (pi @ pj)[nz])).sum()
    hi = -(pi[pi > 0] * np.log(pi[pi > 0])).sum()
    hj = -(pj[pj > 0] * np.log(pj[pj > 0])).sum()
    return float(mi / ((hi + hj) / 2)) if (hi + hj) > 0 else 0.0


def analisar_comunidades(G, institutos, out_dir):
    qs, melhor, melhor_q = [], None, -1
    for s in range(LOUVAIN_SEEDS_BASELINE):
        com = louvain_communities(G, seed=s)
        q = modularity(G, com)
        qs.append(q)
        if q > melhor_q:
            melhor, melhor_q = com, q
    print(f"Baseline Louvain ({LOUVAIN_SEEDS_BASELINE} sementes): "
          f"Q = {np.mean(qs):.4f} +- {np.std(qs, ddof=1):.4f}; "
          f"melhor Q = {melhor_q:.4f}; nº comunidades = {len(melhor)}")

    rows, rot_com, rot_inst = [], [], []
    for cid, c in enumerate(sorted(melhor, key=len, reverse=True)):
        ids = [institutos[n].split(";")[0].strip() for n in c]
        ident = [i for i in ids if i != "Não identificado"]
        vc = pd.Series(ident).value_counts() if ident else pd.Series(dtype=int)
        rows.append({
            "comunidade": cid, "tamanho": len(c), "identificados": len(ident),
            "instituto_dominante": vc.index[0] if len(vc) else "",
            "participacao_dominante": round(vc.iloc[0] / len(ident), 3) if len(ident) else np.nan,
            "segundo_instituto": vc.index[1] if len(vc) > 1 else "",
        })
        for n in c:
            if institutos[n].split(";")[0].strip() != "Não identificado":
                rot_com.append(cid)
                rot_inst.append(institutos[n].split(";")[0].strip())
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "comunidades_baseline.csv"), index=False, encoding="utf-8-sig")
    nmi_val = nmi(rot_com, rot_inst)
    print(f"NMI Louvain vs instituto: {nmi_val:.3f}")
    sizes = np.array(sorted(len(c) for c in melhor))
    return {
        "Q_medio": float(np.mean(qs)), "Q_desvio": float(np.std(qs, ddof=1)),
        "Q_max": float(melhor_q), "n_comunidades": len(melhor), "NMI_instituto": nmi_val,
        "tamanhos": sizes.tolist()
    }


def estimar_tau2(sizes, smin, smax):
    x = np.array([s for s in sizes if smin <= s <= smax], float)
    dom = np.arange(smin, smax + 1, dtype=float)
    melhor, ll_max = None, -np.inf
    for tau in np.arange(1.05, 4.0, 0.01):
        ll = -tau * np.log(x).sum() - len(x) * np.log((dom ** -tau).sum())
        if ll > ll_max:
            melhor, ll_max = tau, ll
    return float(melhor)


def gerar_er(N, L, seed):
    return nx.gnm_random_graph(N, L, seed=seed)


def gerar_ws(N, L, p, seed):
    rng = random.Random(seed)
    kb = int(2 * L / N) // 2 * 2
    half = kb // 2
    E = set()
    for i in range(N):
        for j in range(1, half + 1):
            E.add(tuple(sorted((i, (i + j) % N))))
    extra = L - len(E)
    ids = list(range(N))
    rng.shuffle(ids)
    for i in ids[:extra]:
        E.add(tuple(sorted((i, (i + half + 1) % N))))
    assert len(E) == L, (len(E), L)
    E = list(E)
    S = set(E)
    for k, (u, v) in enumerate(E):
        if rng.random() < p:
            for _ in range(50):
                w = rng.randrange(N)
                e = tuple(sorted((u, w)))
                if w != u and e not in S:
                    S.remove((u, v))
                    S.add(e)
                    E[k] = e
                    break
    g = nx.Graph()
    g.add_nodes_from(range(N))
    g.add_edges_from(S)
    return g


def gerar_hk(N, L, p, seed):
    rng = random.Random(seed)
    n_new = N - (M0 + 1)
    base = (M0 + 1) * M0 // 2
    resto = L - base - n_new * M0
    assert 0 <= resto <= n_new, "L incompatível com M0"
    ms = [M0 + 1] * resto + [M0] * (n_new - resto)
    rng.shuffle(ms)
    g = nx.complete_graph(M0 + 1)
    rep = []
    for u, v in g.edges():
        rep += [u, v]
    for k, node in enumerate(range(M0 + 1, N)):
        m = ms[k]
        alvos, last = set(), None
        while len(alvos) < m:
            if last is not None and rng.random() < p:
                viz = [x for x in g[last] if x != node and x not in alvos]
                if viz:
                    t = rng.choice(viz)
                    alvos.add(t)
                    last = t
                    continue
            t = rng.choice(rep)
            if t in alvos:
                continue
            alvos.add(t)
            last = t
        g.add_node(node)
        for t in alvos:
            g.add_edge(node, t)
            rep += [node, t]
    assert g.number_of_edges() == L, (g.number_of_edges(), L)
    return g


def gerar_config(G, seed):
    g = G.copy()
    L = g.number_of_edges()
    nx.double_edge_swap(g, nswap=N_SWAPS_FACTOR * L, max_tries=100 * L, seed=seed)
    return g


def gerar_lfr(N, mu, seed, tau1, tau2, cmin, cmax):
    g = LFR_benchmark_graph(
        n=N, tau1=tau1, tau2=tau2, mu=mu,
        min_degree=LFR_MIN_DEG, max_degree=LFR_MAX_DEG,
        min_community=cmin, max_community=cmax,
        seed=seed, max_iters=2000,
    )
    h = nx.Graph(g)
    h.remove_edges_from(nx.selfloop_edges(h))
    return h


def calibrar_tau1(N, L, tau2, cmin, cmax):
    melhor, dif = None, np.inf
    for tau1 in np.arange(2.10, 2.65, 0.05):
        Ls = []
        for s in range(3):
            try:
                Ls.append(gerar_lfr(N, 0.1, s, tau1, tau2, cmin, cmax).number_of_edges())
            except Exception:
                pass
        if len(Ls) >= 2 and abs(np.mean(Ls) - L) < dif:
            melhor, dif = float(tau1), abs(np.mean(Ls) - L)
    return melhor


def rodar(nome, gerador, reps, graus_out, seeds, L_alvo=None, tol=None):
    t0 = time.time()
    rows, usadas, falhas, rejeitadas = [], [], 0, 0
    for s in seeds:
        if len(rows) == reps:
            break
        try:
            g = gerador(s)
        except Exception:
            falhas += 1
            continue
        if tol is not None and abs(g.number_of_edges() - L_alvo) / L_alvo > tol:
            rejeitadas += 1
            continue
        rows.append(metricas(g, seed_louvain=s))
        usadas.append(s)
        if nome not in graus_out:
            graus_out[nome] = sorted(d for _, d in g.degree())
    df = pd.DataFrame(rows)
    print(f"{nome:<8} ok={len(rows):>2} falhas={falhas:>3} rejeitadas={rejeitadas:>3} "
          f"L={df.L.mean():.0f} d={df.d.mean():.2f} diam={df.diam.mean():.1f} "
          f"cc={df.cc_med.mean():.3f} Q={df.Q.mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
    return df, usadas


def mil(x):
    return f"{x:,.0f}".replace(",", ".")


def vir(x, c):
    return f"{x:.{c}f}".replace(".", "{,}")


def tabela_latex(base, res):
    def cel(df, col, casas):
        m, s = df[col].mean(), df[col].std()
        return f"${vir(m, casas)} \\pm {vir(s, casas)}$"

    linhas = [
        f"Baseline & {mil(base['L'])} & 1 & {vir(base['d'], 2)} & {int(base['diam'])} & "
        f"{vir(base['cc_med'], 3)} & {vir(base['cc_glob'], 3)} & {vir(base['Q_medio'], 3)} & "
        f"{int(base['kmax'])} \\\\",
        "\\midrule",
    ]
    rotulos = {
        "ER": "ER", "WS": "WS ($p{=}0{,}1$)", "BA": "BA", "HK": "HK ($p{=}1{,}0$)",
        "CM": "Configuração", "LFR_0.1": "LFR ($\\mu{=}0{,}1$)", "LFR_0.3": "LFR ($\\mu{=}0{,}3$)"
    }
    for k, rot in rotulos.items():
        if k not in res:
            continue
        df = res[k]
        Lcel = mil(df.L.mean()) if df.L.std() == 0 else f"${mil(df.L.mean())} \\pm {mil(df.L.std())}$"
        comp = "1" if (df.comp.std() == 0 and df.comp.mean() == 1) else cel(df, "comp", 1)
        linhas.append(" & ".join([rot, Lcel, comp, cel(df, "d", 2), cel(df, "diam", 1),
                                  cel(df, "cc_med", 3), cel(df, "cc_glob", 3), cel(df, "Q", 3),
                                  cel(df, "kmax", 0)]) + " \\\\")
    corpo = "\n".join(linhas)
    return (
        "\\begin{table*}[t]\n\\centering\n\\footnotesize\n"
        "\\caption{Comparação entre a baseline e os modelos de referência. "
        "Média $\\pm$ desvio-padrão de réplicas; $Q$: modularidade.}\n"
        "\\label{tab:modelos}\n\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n"
        "\\textbf{Rede} & \\textbf{$L$} & \\textbf{Compon.} & \\textbf{$\\langle d \\rangle$} & "
        "\\textbf{Diâm.} & \\textbf{CA médio} & \\textbf{CA global} & \\textbf{$Q$} & "
        "\\textbf{$k_{\\max}$} \\\\\n\\midrule\n" + corpo +
        "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table*}\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dados_de_extracao")
    ap.add_argument("--out", default="resultados_modelagem")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    G, institutos = carregar_baseline(a.data)
    N, L = G.number_of_nodes(), G.number_of_edges()
    print(f"Baseline: N={N} L={L}")
    base = metricas(G, seed_louvain=0)
    com = analisar_comunidades(G, institutos, a.out)
    base["Q_medio"] = com["Q_medio"]
    print("Baseline:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in base.items()})

    sizes = com["tamanhos"]
    cmin = max(int(np.percentile(sizes, 10)), LFR_MIN_DEG + 1)
    cmax = int(max(sizes))
    tau2 = estimar_tau2(sizes, cmin, cmax)
    tau1 = calibrar_tau1(N, L, tau2, cmin, cmax)
    if tau1 is None:
        raise SystemExit("Não foi possível calibrar tau1 do LFR.")
    if tau2 <= 1.06:
        print("AVISO: tau2 no limite inferior da grade.")
    print(f"LFR: tau1={tau1:.2f} tau2={tau2:.2f} comunidades [{cmin}, {cmax}] grau [{LFR_MIN_DEG}, {LFR_MAX_DEG}]")

    graus = {"Baseline": sorted(d for _, d in G.degree())}
    res, seeds = {}, {}
    plano = [
        ("ER", lambda s: gerar_er(N, L, s), range(a.reps), None),
        ("WS", lambda s: gerar_ws(N, L, WS_P, s), range(a.reps), None),
        ("BA", lambda s: gerar_hk(N, L, 0.0, s), range(a.reps), None),
        ("HK", lambda s: gerar_hk(N, L, HK_P, s), range(a.reps), None),
        ("CM", lambda s: gerar_config(G, s), range(a.reps), None),
    ]
    for mu in LFR_MUS:
        plano.append((f"LFR_{mu}",
                      (lambda m: (lambda s: gerar_lfr(N, m, s, tau1, tau2, cmin, cmax)))(mu),
                      range(300), LFR_TOL))

    for nome, fn, sem, tol in plano:
        df, usadas = rodar(nome, fn, a.reps, graus, sem, L_alvo=L, tol=tol)
        res[nome] = df
        seeds[nome] = usadas
        df.to_csv(os.path.join(a.out, f"replicas_{nome}.csv"), index=False)

    resumo = {
        "ambiente": {
            "python": sys.version.split()[0], "networkx": nx.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "so": platform.platform()
        },
        "lfr_parametros": {
            "tau1": tau1, "tau2": tau2, "min_community": cmin, "max_community": cmax,
            "min_degree": LFR_MIN_DEG, "max_degree": LFR_MAX_DEG, "tolerancia_L": LFR_TOL
        },
        "baseline": base,
        "comunidades_baseline": {k: v for k, v in com.items() if k != "tamanhos"},
        "tamanhos_comunidades_baseline": com["tamanhos"],
    }
    for nome, df in res.items():
        resumo[nome] = {
            "mean": df.mean().to_dict(),
            "std": df.std().to_dict(),
            "n_replicas": len(df),
            "sementes": seeds[nome]
        }

    with open(os.path.join(a.out, "resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=1, default=float)

    np.savez(os.path.join(a.out, "graus_primeira_replica.npz"),
             **{k: np.array(v) for k, v in graus.items()})

    with open(os.path.join(a.out, "tabela_modelos.tex"), "w", encoding="utf-8") as f:
        f.write(tabela_latex(base, res))

    print(f"\nConcluído. Arquivos salvos em: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()