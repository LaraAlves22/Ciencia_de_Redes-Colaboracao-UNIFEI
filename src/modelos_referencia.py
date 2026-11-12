#!/usr/bin/env python3
"""AT3 - modelos de referência (ER, WS, BA, HK, configuração, LFR) para a rede baseline do IMC.
Baseline = maior componente da rede completa (docentes + externos), sem pesos. Réplicas: sementes 0..29.
Os parâmetros livres (p do WS e do HK, mu do LFR) são escolhidos em réplicas separadas (sementes 100..109)."""
import json, os, random, sys, time
import numpy as np, pandas as pd, networkx as nx, powerlaw
from networkx.algorithms.community import louvain_communities, modularity
from networkx.generators.community import LFR_benchmark_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy import stats

PASTA = "dados/rede"; SAIDA = "resultados/at3"
R = 30; R_SEL = 10; SEM_SEL = 100
PROPS = ["d", "diam", "cc_med", "cc_glob", "Q", "kmax", "r", "S"]

def carregar():
    n = pd.read_csv(f"{PASTA}/nos.csv", encoding="utf-8-sig"); e = pd.read_csv(f"{PASTA}/arestas.csv", encoding="utf-8-sig")
    G = nx.Graph(); G.add_nodes_from(n.id); G.add_edges_from(zip(e.source, e.target))
    return nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)).copy())

def metricas(g, N0, seed=0):
    A = nx.to_scipy_sparse_array(g, format="csr")
    nc, lab = connected_components(A, directed=False)
    big = np.argmax(np.bincount(lab)); idx = np.where(lab == big)[0]
    D = shortest_path(csr_matrix(A[idx][:, idx]), unweighted=True, directed=False)
    m = len(idx)
    com = louvain_communities(g, seed=seed)
    return dict(N=g.number_of_nodes(), L=g.number_of_edges(), d=float(D.sum() / (m * (m - 1))), diam=int(D.max()),
                cc_med=nx.average_clustering(g), cc_glob=nx.transitivity(g), Q=modularity(g, com),
                kmax=max(d for _, d in g.degree()), r=nx.degree_assortativity_coefficient(g), S=m / g.number_of_nodes())

def gerar_er(N, L, s): return nx.gnm_random_graph(N, L, seed=s)

def gerar_ws(N, L, p, s):
    rng = random.Random(s); half = (int(2 * L / N) // 2 * 2) // 2
    E = {tuple(sorted((i, (i + j) % N))) for i in range(N) for j in range(1, half + 1)}
    ids = list(range(N)); rng.shuffle(ids)
    for i in ids[:L - len(E)]: E.add(tuple(sorted((i, (i + half + 1) % N))))
    assert len(E) == L
    E = list(E); S = set(E)
    for k, (u, v) in enumerate(E):
        if rng.random() < p:
            for _ in range(50):
                w = rng.randrange(N); e = tuple(sorted((u, w)))
                if w != u and e not in S: S.remove((u, v)); S.add(e); E[k] = e; break
    g = nx.Graph(); g.add_nodes_from(range(N)); g.add_edges_from(S); return g

def gerar_hk(N, L, p, s):
    """p=0: BA; p>0: Holme-Kim. Cada novo nó traz 1 ou 2 arestas (misturadas para somar exatamente L)."""
    rng = random.Random(s); g = nx.complete_graph(3); rep = [u for e in g.edges() for u in e]
    n_new = N - 3; n2 = L - 3 - n_new; assert 0 <= n2 <= n_new
    ms = [2] * n2 + [1] * (n_new - n2); rng.shuffle(ms)
    for k, node in enumerate(range(3, N)):
        alvos, last = set(), None
        while len(alvos) < ms[k]:
            if last is not None and rng.random() < p:
                viz = [x for x in g[last] if x != node and x not in alvos]
                if viz: t = rng.choice(viz); alvos.add(t); last = t; continue
            t = rng.choice(rep)
            if t not in alvos: alvos.add(t); last = t
        g.add_node(node)
        for t in alvos: g.add_edge(node, t); rep += [node, t]
    assert g.number_of_edges() == L; return g

def gerar_cm(G, s):
    g = G.copy(); L = g.number_of_edges()
    nx.double_edge_swap(g, nswap=10 * L, max_tries=100 * L, seed=s); return g

def gerar_lfr(N, mu, s, par):
    g = nx.Graph(LFR_benchmark_graph(n=N, tau1=par["tau1"], tau2=par["tau2"], mu=mu, min_degree=1, max_degree=par["kmax"],
                                     min_community=par["cmin"], max_community=par["cmax"], seed=s, max_iters=2000))
    g.remove_edges_from(nx.selfloop_edges(g)); return g

def erro(m, base):
    e = []
    for k in PROPS:
        den = abs(base[k]) if base[k] != 0 else 1
        e.append(min(abs(m[k] - base[k]) / den, 1.0))
    return np.array(e)

def ks_grau(g, G):
    a = np.array([d for _, d in g.degree()]); b = np.array([d for _, d in G.degree()])
    return float(stats.ks_2samp(a, b).statistic)

def rodar(nome, fn, sementes, base, G, alvoL=None, tol=None, quer=None):
    rows = []
    for s in sementes:
        if quer and len(rows) == quer: break
        try: g = fn(s)
        except Exception: continue
        if tol and abs(g.number_of_edges() - alvoL) / alvoL > tol: continue
        m = metricas(g, G.number_of_nodes(), s); m["DKS"] = ks_grau(g, G); m["seed"] = s
        m["erro"] = float(erro(m, base).mean()); rows.append(m)
    df = pd.DataFrame(rows); df.insert(0, "modelo", nome); return df

def main():
    t0 = time.time(); os.makedirs(SAIDA, exist_ok=True)
    G = carregar(); N, L = G.number_of_nodes(), G.number_of_edges()
    base = metricas(G, N, 0)
    qs = [modularity(G, louvain_communities(G, seed=s)) for s in range(30)]; base["Q"] = float(np.mean(qs)); base["Q_sd"] = float(np.std(qs, ddof=1))
    com = louvain_communities(G, seed=0); sz = sorted(len(c) for c in com)
    inter = sum(1 for u, v in G.edges() if not any(u in c and v in c for c in com))
    base["mu_hat"] = inter / L; base["n_com"] = len(com)
    print("baseline", base, flush=True)
    # LFR: tau2 por MV nos tamanhos das comunidades da baseline, tau1 calibrado para L
    cmin, cmax = int(np.percentile(sz, 10)), int(max(sz))
    x = np.array([s for s in sz if cmin <= s <= cmax], float); dom = np.arange(cmin, cmax + 1, dtype=float)
    tau2 = max(np.arange(1.05, 4, .01), key=lambda t: -t * np.log(x).sum() - len(x) * np.log((dom ** -t).sum()))
    par = dict(tau2=float(tau2), cmin=cmin, cmax=cmax, kmax=int(base["kmax"]))
    melhor, dif = None, 1e9
    for t1 in np.arange(2.2, 3.0, 0.05):
        par["tau1"] = float(t1); Ls = []
        for s in range(3):
            try: Ls.append(gerar_lfr(N, base["mu_hat"], s, par).number_of_edges())
            except Exception: pass
        if len(Ls) >= 2 and abs(np.mean(Ls) - L) < dif: melhor, dif = float(t1), abs(np.mean(Ls) - L)
    par["tau1"] = melhor; print("LFR", par, flush=True)
    # varredura de parâmetros (sementes 100..109)
    sel = []
    for p in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
        sel.append(rodar(f"WS_p={p}", lambda s, p=p: gerar_ws(N, L, p, s), range(SEM_SEL, SEM_SEL + R_SEL), base, G))
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        sel.append(rodar(f"HK_p={p}", lambda s, p=p: gerar_hk(N, L, p, s), range(SEM_SEL, SEM_SEL + R_SEL), base, G))
    for mu in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
        sel.append(rodar(f"LFR_mu={mu}", lambda s, mu=mu: gerar_lfr(N, mu, s, par), range(SEM_SEL, SEM_SEL + 200), base, G, L, 0.03, R_SEL))
    sel = pd.concat(sel); sel.to_csv(f"{SAIDA}/varredura_parametros.csv", index=False)
    resumo_sel = sel.groupby("modelo")[PROPS + ["DKS", "erro"]].mean(); resumo_sel.to_csv(f"{SAIDA}/varredura_resumo.csv")
    print(resumo_sel.round(3), flush=True)
    best = lambda pref: resumo_sel[resumo_sel.index.str.startswith(pref)]["erro"].idxmin()
    pws = float(best("WS").split("=")[1]); phk = float(best("HK").split("=")[1]); mu = float(best("LFR").split("=")[1])
    print("escolhidos", pws, phk, mu, flush=True)
    # BA com m fixo (L varia) - só para a discussão do parâmetro m
    mrows = []
    for m_ in (1, 2, 3):
        for s in range(10):
            g = nx.barabasi_albert_graph(N, m_, seed=s); mm = metricas(g, N, s); mm["m"] = m_; mrows.append(mm)
    pd.DataFrame(mrows).groupby("m")[["L", "d", "diam", "cc_med", "kmax"]].mean().to_csv(f"{SAIDA}/ba_variando_m.csv")
    # execução final (sementes 0..29)
    fin = [rodar("ER", lambda s: gerar_er(N, L, s), range(R), base, G),
           rodar("WS", lambda s: gerar_ws(N, L, pws, s), range(R), base, G),
           rodar("BA", lambda s: gerar_hk(N, L, 0.0, s), range(R), base, G),
           rodar("HK", lambda s: gerar_hk(N, L, phk, s), range(R), base, G),
           rodar("Configuração", lambda s: gerar_cm(G, s), range(R), base, G),
           rodar("LFR", lambda s: gerar_lfr(N, mu, s, par), range(2000), base, G, L, 0.03, R)]
    fin = pd.concat(fin); fin.to_csv(f"{SAIDA}/replicas.csv", index=False)
    # teste de aderência por propriedade: erro relativo <= 10 %
    tab = fin.groupby("modelo")[PROPS + ["L", "DKS", "erro"]].agg(["mean", "std"])
    tab.to_csv(f"{SAIDA}/tabela_modelos.csv")
    ader = {}
    for mname, df in fin.groupby("modelo"):
        ader[mname] = {k: float(np.mean([abs(v - base[k]) / abs(base[k]) <= 0.10 for v in df[k]])) for k in PROPS}
    # sensibilidade ao peso: ordem dos modelos sob 10.000 vetores de pesos Dirichlet
    E = {m: np.array([np.minimum(np.abs(df[k] - base[k]) / abs(base[k]), 1).mean() for k in PROPS]) for m, df in fin.groupby("modelo")}
    rng = np.random.default_rng(0); W = rng.dirichlet(np.ones(len(PROPS)), 10000); nomes = list(E)
    sc = np.array([W @ E[m] for m in nomes]).T
    primeiro = {m: float(np.mean(sc.argmin(1) == i)) for i, m in enumerate(nomes)}
    # MWU entre o melhor e os demais (erro por réplica), Holm
    ordem = fin.groupby("modelo")["erro"].mean().sort_values(); mel = ordem.index[0]; pv = []; res_mw = []
    for m in ordem.index[1:]:
        a, b = fin[fin.modelo == mel]["erro"], fin[fin.modelo == m]["erro"]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided"); delta = (2 * u / (len(a) * len(b))) - 1
        res_mw.append(dict(contra=m, U=float(u), p=float(p), delta_cliff=float(delta)))
    ps = np.array([r["p"] for r in res_mw]); o = np.argsort(ps); holm = np.empty(len(ps)); run = 0
    for rank, i in enumerate(o): run = max(run, min(1, (len(ps) - rank) * ps[i])); holm[i] = run
    for r, h in zip(res_mw, holm): r["p_holm"] = float(h)
    # lei de potência (Clauset et al. 2009) para o grau
    k = np.array([d for _, d in G.degree()]); fit = powerlaw.Fit(k, discrete=True, verbose=False)
    pl = dict(kmin=int(fit.xmin), alpha=float(fit.alpha), D=float(fit.D), n_cauda=int((k >= fit.xmin).sum()))
    Rv, pv_ = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True); pl["vuong_R"] = float(Rv); pl["vuong_p"] = float(pv_)
    nb = 200; rngb = np.random.default_rng(0); Ds = []; kb = k[k < fit.xmin]; nt = (k >= fit.xmin).sum()
    for b in range(nb):
        cauda = powerlaw.Power_Law(xmin=fit.xmin, parameters=[fit.alpha], discrete=True).generate_random(int(nt))
        sint = np.concatenate([rngb.choice(kb, len(k) - int(nt)), np.round(cauda).astype(int)])
        Ds.append(powerlaw.Fit(sint, discrete=True, verbose=False).D)
    pl["bootstrap_n"] = nb; pl["bootstrap_p"] = float((1 + np.sum(np.array(Ds) >= fit.D)) / (nb + 1)); pl["D_mediana_sint"] = float(np.median(Ds))
    for nm in ("BA", "HK", "LFR"):
        kk = np.array([d for _, d in {"BA": gerar_hk(N, L, 0.0, 0), "HK": gerar_hk(N, L, phk, 0), "LFR": gerar_lfr(N, mu, int(fin[fin.modelo == "LFR"].seed.iloc[0]), par)}[nm].degree()])
        f2 = powerlaw.Fit(kk, discrete=True, verbose=False); pl[f"alpha_{nm}"] = float(f2.alpha); pl[f"kmin_{nm}"] = int(f2.xmin)
    json.dump(dict(baseline=base, lfr=par, escolhidos=dict(p_ws=pws, p_hk=phk, mu_lfr=mu), aderencia_10pct=ader, primeiro_lugar_dirichlet=primeiro,
                   mannwhitney=res_mw, lei_potencia=pl, tamanhos_comunidades=sz, tempo_s=time.time() - t0),
              open(f"{SAIDA}/resumo.json", "w"), indent=1, default=float)
    np.savez(f"{SAIDA}/graus.npz", baseline=np.array(sorted(d for _, d in G.degree())),
             **{nm: np.array(sorted(d for _, d in g.degree())) for nm, g in
                {"ER": gerar_er(N, L, 0), "WS": gerar_ws(N, L, pws, 0), "BA": gerar_hk(N, L, 0.0, 0), "HK": gerar_hk(N, L, phk, 0),
                 "CM": gerar_cm(G, 0), "LFR": gerar_lfr(N, mu, int(fin[fin.modelo == "LFR"].seed.iloc[0]), par)}.items()})
    print("fim", round(time.time() - t0), "s", flush=True)

if __name__ == "__main__": main()
