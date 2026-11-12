#!/usr/bin/env python3
"""AT4 - percolação e robustez da rede do IMC ao afastamento de docentes.
Unidade de perturbação: docente (48). Funções preservadas: (F1) integração interna = maior componente de docentes;
(F2) alcance externo = fração dos 2.065 externos que mantêm ao menos um docente como colaborador."""
import json, os, sys, time
import numpy as np, pandas as pd, networkx as nx
from scipy.sparse.csgraph import connected_components, shortest_path
sys.path.insert(0, os.path.dirname(__file__))
import modelos_referencia as mr

PASTA, SAIDA = "dados/rede", "resultados/at4"; NR = 1000

def carregar(modelo="M2"):
    n = pd.read_csv(f"{PASTA}/nos.csv", encoding="utf-8-sig"); e = pd.read_csv(f"{PASTA}/arestas.csv", encoding="utf-8-sig")
    G = nx.Graph()
    for r in n.itertuples(): G.add_node(r.id, tipo=r.tipo_no, camada=r.camada, rotulo=r.rotulo)
    for r in e.itertuples(): G.add_edge(r.source, r.target, w=getattr(r, modelo))
    return G

class Rede:
    def __init__(s, G):
        s.G = G; s.D = sorted([v for v, d in G.nodes(data=True) if d["tipo"] == "D"], key=lambda v: G.nodes[v]["rotulo"])
        s.X = [v for v in G if v not in set(s.D)]; s.nD = len(s.D); iD = {v: i for i, v in enumerate(s.D)}; iX = {v: i for i, v in enumerate(s.X)}
        s.A = np.zeros((s.nD, s.nD), bool); s.B = np.zeros((s.nD, len(s.X)), bool)
        s.ed = []  # (u, v, w) com índices: docente-docente (i,j) ou docente-externo (i, -1-k)
        for u, v, d in G.edges(data=True):
            if u in iD and v in iD: s.A[iD[u], iD[v]] = s.A[iD[v], iD[u]] = True; s.ed.append((iD[u], iD[v], d["w"]))
            else:
                a, b = (u, v) if u in iD else (v, u)
                if a in iD and b in iX: s.B[iD[a], iX[b]] = True; s.ed.append((iD[a], -1 - iX[b], d["w"]))
        s.W0 = sum(w for _, _, w in s.ed); s.camada = np.array([G.nodes[v]["camada"] for v in s.D])
        s.EI = np.array([(i, j) for i in range(s.nD) for j in range(i + 1, s.nD) if s.A[i, j] and s.camada[i] != s.camada[j]])
        s.ed_i = np.array([a for a, b, w in s.ed]); s.ed_j = np.array([b for a, b, w in s.ed]); s.ed_w = np.array([w for a, b, w in s.ed])
    def estado(s, vivo):
        idx = np.where(vivo)[0]; sub = s.A[np.ix_(idx, idx)]
        if len(idx) == 0: return dict(SD=0, ED=0, cob=0, W=0, inter=0)
        nc, lab = connected_components(sub, directed=False); SD = np.bincount(lab).max() / s.nD
        Dm = shortest_path(sub.astype(float), unweighted=True, directed=False); np.fill_diagonal(Dm, np.inf)
        ED = (1 / Dm).sum() / (s.nD * (s.nD - 1))
        cob = (s.B[vivo].sum(0) > 0).mean()
        ok = vivo[s.ed_i] & np.where(s.ed_j >= 0, vivo[np.maximum(s.ed_j, 0)], True)
        inter = int(sum(vivo[i] and vivo[j] for i, j in s.EI))
        return dict(SD=SD, ED=ED, cob=cob, W=s.ed_w[ok].sum() / s.W0, inter=inter)
    def curva(s, ordem):
        vivo = np.ones(s.nD, bool); out = [s.estado(vivo)]
        for v in ordem: vivo[v] = False; out.append(s.estado(vivo))
        return pd.DataFrame(out)
    def adaptativa(s, criterio):
        vivo = np.ones(s.nD, bool); ordem = []
        compartilhados = [k for k in range(len(s.X)) if s.B[:, k].sum() >= 2]
        for _ in range(s.nD):
            idx = np.where(vivo)[0]
            if criterio == "grau_DD": sc = s.A[np.ix_(idx, idx)].sum(1).astype(float)
            elif criterio == "exclusivos":
                cnt = s.B[vivo].sum(0); sc = np.array([((s.B[i] & (cnt == 1)).sum()) for i in idx], float)
            elif criterio == "intermediacao":
                H = nx.Graph(); H.add_nodes_from(int(i) for i in idx)
                for k in compartilhados:
                    ds = [int(i) for i in np.where(s.B[:, k] & vivo)[0]]
                    if len(ds) >= 2:
                        for i in ds: H.add_edge(i, ("x", k))
                for a in idx:
                    for b in idx:
                        if a < b and s.A[a, b]: H.add_edge(int(a), int(b))
                bc = nx.betweenness_centrality(H); sc = np.array([bc[int(i)] for i in idx])
            # desempate determinístico: maior grau total, depois menor índice
            tot = s.A[np.ix_(idx, idx)].sum(1) + s.B[idx].sum(1)
            k = max(range(len(idx)), key=lambda t: (sc[t], tot[t], -idx[t])); ordem.append(int(idx[k])); vivo[idx[k]] = False
        return ordem

def f_falha(serie, lim):
    """menor k com série < lim (None se nunca)"""
    k = np.where(np.asarray(serie) < lim)[0]; return int(k[0]) if len(k) else None

def Rindex(serie): return float(np.mean(serie[1:]))

def percolacao(g):
    k = np.array([d for _, d in g.degree()]); kap = (k ** 2).mean() / k.mean()
    return dict(k_med=float(k.mean()), k2_med=float((k ** 2).mean()), kappa=float(kap), fc_aleatorio=float(1 - 1 / (kap - 1)) if kap > 2 else 0.0,
                r_grau=float(nx.degree_assortativity_coefficient(g)))

def ataque_geral(g, N0, fs, ordem=None, rng=None):
    """remoção de fração f dos nós; devolve S(f)=LCC/N0"""
    nodes = list(g.nodes()); A = nx.to_scipy_sparse_array(g, nodelist=nodes, format="csr"); out = []
    pos = {v: i for i, v in enumerate(nodes)}
    ordem = ordem if ordem is not None else list(rng.permutation(len(nodes)))
    for f in fs:
        m = np.ones(len(nodes), bool); m[ordem[:int(round(f * len(nodes)))]] = False; idx = np.where(m)[0]
        nc, lab = connected_components(A[idx][:, idx], directed=False); out.append(np.bincount(lab).max() / N0)
    return np.array(out)

def main():
    t0 = time.time(); os.makedirs(SAIDA, exist_ok=True)
    G = carregar(); R = Rede(G); nD = R.nD; res = {}
    base = R.estado(np.ones(nD, bool)); res["inicial"] = {k: float(v) for k, v in base.items()}
    res["inicial"]["n_docentes"] = nD; res["inicial"]["n_externos"] = len(R.X)
    res["externos_grau1"] = float((R.B.sum(0) == 1).mean())
    # percolação teórica
    DD = G.subgraph(R.D).copy(); res["percolacao_DD"] = percolacao(DD); res["percolacao_rede_completa"] = percolacao(G)
    curvas = {}
    rng = np.random.default_rng(0); aleat = []
    for s in range(NR): aleat.append(R.curva(list(rng.permutation(nD))))
    for k in ("SD", "ED", "cob", "W", "inter"): curvas[f"aleatoria_{k}"] = np.mean([a[k].values for a in aleat], axis=0)
    arr = {k: np.array([a[k].values for a in aleat]) for k in ("SD", "ED", "cob", "W", "inter")}
    for k in arr: pd.DataFrame(arr[k]).describe(percentiles=[.025, .25, .5, .75, .975]).T.to_csv(f"{SAIDA}/aleatoria_{k}.csv")
    alvos = {}
    for crit in ("grau_DD", "intermediacao", "exclusivos"):
        ordem = R.adaptativa(crit); alvos[crit] = (ordem, R.curva(ordem))
        pd.DataFrame({"passo": range(1, nD + 1), "docente": [G.nodes[R.D[i]]["rotulo"] for i in ordem]}).to_csv(f"{SAIDA}/ordem_{crit}.csv", index=False)
    # camada: remoção aleatória só de uma área
    cam = {}
    for nome in ("Matemática", "Computação"):
        idx = [i for i in range(nD) if R.camada[i] == nome]; cs = []
        for s in range(NR):
            o = list(rng.permutation(idx)) + [i for i in range(nD) if i not in idx]; cs.append(R.curva(o[:len(idx)])[["SD", "cob", "inter"]].values)
        cam[nome] = np.mean(cs, axis=0)
    # indicadores por cenário
    lim_SD, lim_cob = 0.5 * base["SD"], 0.5
    tab = []
    def linha(nome, SD, cob, W, inter, ED):
        tab.append(dict(cenario=nome, R_SD=Rindex(SD / base["SD"]), k_falha_SD=f_falha(SD, lim_SD), R_cob=Rindex(cob), k_falha_cob=f_falha(cob, lim_cob),
                        R_W=Rindex(W), SD_25pct=float(np.interp(0.25 * nD, range(nD + 1), SD)), cob_25pct=float(np.interp(0.25 * nD, range(nD + 1), cob)),
                        inter_zero_k=f_falha(inter, 0.5)))
    # aleatória: indicadores por sequência -> mediana e IQR
    kSD = [f_falha(a["SD"].values, lim_SD) for a in aleat]; kcob = [f_falha(a["cob"].values, lim_cob) for a in aleat]
    kint = [f_falha(a["inter"].values, 0.5) for a in aleat]; rD = [Rindex(a["SD"].values / base["SD"]) for a in aleat]; rC = [Rindex(a["cob"].values) for a in aleat]
    rW = [Rindex(a["W"].values) for a in aleat]
    q = lambda x: [float(np.nanpercentile([v if v is not None else np.nan for v in x], p)) for p in (2.5, 25, 50, 75, 97.5)]
    res["aleatoria"] = dict(R_SD=q(rD), R_cob=q(rC), R_W=q(rW), k_falha_SD=q(kSD), k_falha_cob=q(kcob), k_falha_inter=q(kint),
                            frac_sem_falha_SD=float(np.mean([v is None for v in kSD])), frac_sem_falha_cob=float(np.mean([v is None for v in kcob])))
    for crit, (ordem, cv) in alvos.items():
        linha(crit, cv.SD.values, cv.cob.values, cv.W.values, cv.inter.values, cv.ED.values)
    pd.DataFrame(tab).to_csv(f"{SAIDA}/cenarios_dirigidos.csv", index=False)
    res["dirigidos"] = tab
    res["ordem_top5"] = {c: [G.nodes[R.D[i]]["rotulo"] for i in o[:5]] for c, (o, _) in alvos.items()}
    # remoção de 1 docente (efeito individual) já está em afastamento_individual.csv (src/analises.py)
    # camadas
    res["por_camada"] = {n: dict(SD_final=float(c[-1, 0]), cob_final=float(c[-1, 1]), inter_final=float(c[-1, 2])) for n, c in cam.items()}
    # ---- Relação com a AT3: remoção de qualquer nó na baseline e nos modelos
    Gb = mr.carregar(); N0 = Gb.number_of_nodes(); fs = np.round(np.arange(0, 0.2001, 0.005), 3)
    resm = json.load(open("resultados/at3/resumo.json")); esc = resm["escolhidos"]; par = resm["lfr"]
    gens = {"Baseline": None, "ER": lambda s: mr.gerar_er(N0, Gb.number_of_edges(), s), "BA": lambda s: mr.gerar_hk(N0, Gb.number_of_edges(), 0.0, s),
            "HK": lambda s: mr.gerar_hk(N0, Gb.number_of_edges(), esc["p_hk"], s), "Configuração": lambda s: mr.gerar_cm(Gb, s),
            "WS": lambda s: mr.gerar_ws(N0, Gb.number_of_edges(), esc["p_ws"], s)}
    geral = []
    for nm, fn in gens.items():
        for s in range(30):
            g = Gb if fn is None else fn(s)
            if fn is None and s > 0: pass
            rs = ataque_geral(g, N0, fs, rng=np.random.default_rng(s))
            ordem = [n for n, _ in sorted(g.degree(), key=lambda t: -t[1])]; pos = {v: i for i, v in enumerate(g.nodes())}
            at = ataque_geral(g, N0, fs, ordem=[pos[v] for v in ordem])
            geral.append(dict(modelo=nm, seed=s, **{f"rand_{f}": v for f, v in zip(fs, rs)}, **{f"atk_{f}": v for f, v in zip(fs, at)}))
    gdf = pd.DataFrame(geral); gdf.to_csv(f"{SAIDA}/ataque_geral_modelos.csv", index=False)
    json.dump(res, open(f"{SAIDA}/resumo.json", "w"), indent=1, default=float)
    np.savez(f"{SAIDA}/curvas.npz", fs=fs, **{k: v for k, v in curvas.items()},
             **{f"dir_{c}_{k}": cv[k].values for c, (_, cv) in alvos.items() for k in ("SD", "ED", "cob", "W", "inter")},
             **{f"cam_{n}": c for n, c in cam.items()}, **{f"arr_{k}": v for k, v in arr.items()})
    print("fim", round(time.time() - t0), "s")

if __name__ == "__main__": main()
