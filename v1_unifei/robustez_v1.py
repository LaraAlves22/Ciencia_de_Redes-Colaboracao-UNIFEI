#!/usr/bin/env python3
import argparse
import heapq
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

import modelagem as mv

LIMIAR_FALHA = 0.05


def curva_S(adj, ordem, n_total):
    n = len(adj)
    pai = list(range(n))
    tam = [1] * n
    presente = [False] * n

    def achar(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    maior = 0

    def inserir(v):
        nonlocal maior
        presente[v] = True
        maior = max(maior, 1)
        for u in adj[v]:
            if presente[u]:
                a, b = achar(u), achar(v)
                if a != b:
                    if tam[a] < tam[b]:
                        a, b = b, a
                    pai[b] = a
                    tam[a] += tam[b]
                    maior = max(maior, tam[a])

    removidos = set(ordem)
    for v in range(n):
        if v not in removidos:
            inserir(v)
    k = len(ordem)
    S = np.zeros(k + 1)
    S[k] = maior
    for i in range(k - 1, -1, -1):
        inserir(ordem[i])
        S[i] = maior
    return S / n_total


def adj_lista(G, peso=False):
    n = G.number_of_nodes()
    adj = [[] for _ in range(n)]
    for u, v, d in G.edges(data=True):
        adj[u].append(v)
        adj[v].append(u)
    return adj


def ordem_aleatoria(n, seed):
    o = list(range(n))
    random.Random(seed).shuffle(o)
    return o


def ordem_adaptativa(adj, seed, pesos=None):
    n = len(adj)
    rng = random.Random(seed)
    if pesos is None:
        val = [len(a) for a in adj]
    else:
        val = [sum(pesos[(u, v) if u < v else (v, u)] for v in adj[u]) for u in range(n)]
    desempate = [rng.random() for _ in range(n)]
    heap = [(-val[u], desempate[u], u) for u in range(n)]
    heapq.heapify(heap)
    vivo = [True] * n
    ordem = []
    while heap:
        v, _, u = heapq.heappop(heap)
        if not vivo[u] or -v != val[u]:
            continue
        vivo[u] = False
        ordem.append(u)
        for w in adj[u]:
            if vivo[w]:
                val[w] -= 1 if pesos is None else pesos[(u, w) if u < w else (w, u)]
                heapq.heappush(heap, (-val[w], desempate[w], w))
    return ordem


def ordem_por_score(score, seed):
    rng = random.Random(seed)
    n = len(score)
    return sorted(range(n), key=lambda u: (-score[u], rng.random()))


def f_estrela(S, n, limiar=LIMIAR_FALHA):
    idx = np.where(S < limiar)[0]
    return (idx[0] / n) if len(idx) else np.nan


def carregar_completa(data_dir):
    nodes = pd.read_csv(os.path.join(data_dir, "nodes.csv"), encoding="utf-8-sig")
    edges = pd.read_csv(os.path.join(data_dir, "edges.csv"), encoding="utf-8-sig")
    G = nx.Graph()
    G.add_nodes_from(nodes["id"])
    for s, t, w in zip(edges["source"], edges["target"], edges["weight"]):
        G.add_edge(s, t, weight=int(w))
    G = nx.convert_node_labels_to_integers(G, label_attribute="oid")
    nome = dict(zip(nodes["id"], nodes["label"]))
    inst = dict(zip(nodes["id"], nodes["instituto"]))
    for n, d in G.nodes(data=True):
        d["label"] = nome[d["oid"]]
        d["inst"] = [i.strip() for i in str(inst[d["oid"]]).split(";") if i.strip() != "Não identificado"]
    return G


def candidatos_ponte(G, bet):
    n = G.number_of_nodes()
    nviz = np.zeros(n)
    for u in G:
        s = set()
        for v in G[u]:
            s.update(G.nodes[v]["inst"])
        nviz[u] = len(s)
    top_b = set(np.argsort(-bet)[:25])
    top_v = set(np.argsort(-nviz, kind="stable")[:25])
    return sorted(top_b | top_v, key=lambda u: -bet[u]), nviz


def media_aleatoria(adj, reps, n_total, seed0=0):
    n = len(adj)
    M = np.array([curva_S(adj, ordem_aleatoria(n, seed0 + r), n_total) for r in range(reps)])
    return M.mean(0), M.std(0, ddof=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dados_de_extracao")
    ap.add_argument("--modelagem", default="resultados_modelagem")
    ap.add_argument("--out", default="resultados_robustez")
    ap.add_argument("--reps-aleat", type=int, default=30)
    ap.add_argument("--reps-modelos", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    G = carregar_completa(a.data)
    N = G.number_of_nodes()
    adj = adj_lista(G)
    pesos = {(min(u, v), max(u, v)): d["weight"] for u, v, d in G.edges(data=True)}
    print(f"Baseline completa: N={N} L={G.number_of_edges()}", flush=True)

    print("Calculando intermediação...", flush=True)
    bet = nx.betweenness_centrality(G)
    bet = np.array([bet[u] for u in range(N)])
    pa = set(nx.articulation_points(G))
    cand, nviz = candidatos_ponte(G, bet)

    curvas = {}
    m, s = media_aleatoria(adj, a.reps_aleat, N)
    curvas["aleatoria"], curvas["aleatoria_dp"] = m, s
    curvas["grau"] = curva_S(adj, ordem_adaptativa(adj, 0), N)
    curvas["intermediacao"] = curva_S(adj, ordem_por_score(bet, 0), N)
    curvas["forca"] = curva_S(adj, ordem_adaptativa(adj, 0, pesos), N)
    ordem_pa = sorted(pa, key=lambda u: -bet[u]) + [u for u in ordem_por_score(bet, 0) if u not in pa]
    curvas["articulacao"] = curva_S(adj, ordem_pa, N)
    print(f"  baseline concluída ({time.time()-t0:.0f}s)", flush=True)

    k_c = len(cand)
    S_cand = curva_S(adj, cand, N)[k_c]
    S_rand_k = curvas["aleatoria"][k_c]
    S_grau_k = curvas["grau"][k_c]
    S0 = curvas["grau"][0]

    grade = [0, .01, .02, .03, .04, .05, .06, .07, .08, .09, .10, .20, .30, .40, .50, .70, .90]
    linhas = []
    for f in grade:
        k = int(round(f * N))
        linhas.append({
            "f": f, "k": k,
            **{c: curvas[c][k] for c in ["aleatoria", "aleatoria_dp", "grau", "intermediacao", "forca", "articulacao"]}
        })
    T = pd.DataFrame(linhas)
    T.to_csv(os.path.join(a.out, "tabela_robustez_baseline_completa.csv"), index=False)

    fe = {c: f_estrela(curvas[c], N) for c in ["aleatoria", "grau", "intermediacao", "forca", "articulacao"]}
    print("f*:", {k: (round(v, 4) if v == v else None) for k, v in fe.items()})

    ordem_g = ordem_adaptativa(adj, 0)
    crit = []
    for i, u in enumerate(ordem_g[:15]):
        crit.append({
            "ordem": i + 1, "pesquisador": G.nodes[u]["label"], "inst": ";".join(G.nodes[u]["inst"]) or "NI",
            "grau_inicial": len(adj[u]), "interm": bet[u], "ponto_articulacao": u in pa,
            "S_apos_remocao": curvas["grau"][i + 1]
        })
    pd.DataFrame(crit).to_csv(os.path.join(a.out, "nos_criticos_ataque_grau.csv"), index=False, encoding="utf-8-sig")

    pd.DataFrame([{
        "pesquisador": G.nodes[u]["label"], "inst": ";".join(G.nodes[u]["inst"]) or "NI",
        "grau": len(adj[u]), "interm": bet[u], "inst_vizinhos": int(nviz[u]),
        "ponto_articulacao": u in pa
    } for u in cand]).to_csv(os.path.join(a.out, "candidatos_ponte.csv"), index=False, encoding="utf-8-sig")

    resumo = json.load(open(os.path.join(a.modelagem, "resumo.json"), encoding="utf-8"))
    GB, _ = mv.carregar_baseline(a.data)
    n_gc, L_gc = GB.number_of_nodes(), GB.number_of_edges()
    lp = resumo["lfr_parametros"]
    adjB = adj_lista(GB)
    res = {}
    mB, sB = media_aleatoria(adjB, a.reps_aleat, n_gc)
    res["Baseline (GC)"] = {"aleat": (mB, sB), "grau": (curva_S(adjB, ordem_adaptativa(adjB, 0), n_gc), None)}

    def gerar(nome, s):
        if nome == "ER": return mv.gerar_er(n_gc, L_gc, s)
        if nome == "BA": return mv.gerar_hk(n_gc, L_gc, 0.0, s)
        if nome == "HK": return mv.gerar_hk(n_gc, L_gc, mv.HK_P, s)
        if nome == "CM": return mv.gerar_config(GB, s)
        if nome == "WS": return mv.gerar_ws(n_gc, L_gc, mv.WS_P, s)
        mu = float(nome.split("_")[1])
        return mv.gerar_lfr(n_gc, mu, s, lp["tau1"], lp["tau2"], lp["min_community"], lp["max_community"])

    for nome in ["ER", "WS", "BA", "HK", "CM", "LFR_0.1"]:
        sem = resumo[nome]["sementes"][: a.reps_modelos]
        ca, cg = [], []
        for s in sem:
            g = nx.convert_node_labels_to_integers(gerar(nome, s))
            ad = adj_lista(g)
            n = len(ad)
            ca.append(curva_S(ad, ordem_aleatoria(n, s), n))
            cg.append(curva_S(ad, ordem_adaptativa(ad, s), n))
        ca, cg = np.array(ca), np.array(cg)
        res[nome] = {"aleat": (ca.mean(0), ca.std(0, ddof=1)), "grau": (cg.mean(0), cg.std(0, ddof=1))}
        print(f"  {nome} ok ({len(sem)} réplicas, {time.time()-t0:.0f}s)", flush=True)

    for nome, d in res.items():
        s0 = d["aleat"][0][0]
        for c in ("aleat", "grau"):
            m, s = d[c]
            d[c] = (m / s0, None if s is None else s / s0)

    linhas = []
    for nome, d in res.items():
        n = n_gc
        for c, rot in (("aleat", "aleatoria"), ("grau", "grau")):
            S = d[c][0]
            idx = np.where(S < LIMIAR_FALHA)[0]
            linhas.append({
                "rede": nome, "cenario": rot,
                "S_f=0.05": S[int(.05 * n)], "S_f=0.10": S[int(.10 * n)], "S_f=0.20": S[int(.20 * n)],
                "S_f=0.50": S[int(.50 * n)], "f_estrela(S/S0<0.05)": idx[0] / n if len(idx) else np.nan
            })
    M = pd.DataFrame(linhas)
    M.to_csv(os.path.join(a.out, "robustez_modelos_vs_baseline.csv"), index=False)
    print("\n", M.round(4).to_string(index=False))

    x = np.arange(N + 1) / N
    xg = np.arange(n_gc + 1) / n_gc
    fc_mr = 0.9433
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cores = {
        "aleatoria": "#1f77b4", "grau": "#d62728", "intermediacao": "#2ca02c",
        "forca": "#ff7f0e", "articulacao": "#9467bd"
    }
    for c, lab in [("aleatoria", "Aleatória (30 exec.)"), ("grau", "Grau (adaptativo)"),
                   ("intermediacao", "Intermediação (estática)"), ("forca", "Força (adaptativa)"),
                   ("articulacao", "Pontos de articulação")]:
        for k in (0, 1):
            ax[k].plot(x, curvas[c], color=cores[c], label=lab if k == 0 else None, lw=1.4)
    ax[0].fill_between(x, curvas["aleatoria"] - curvas["aleatoria_dp"], curvas["aleatoria"] + curvas["aleatoria_dp"],
                       color=cores["aleatoria"], alpha=.25)
    for k in (0, 1):
        ax[k].axhline(LIMIAR_FALHA, color="k", ls=":", lw=.8)
        ax[k].set_xlabel("Fração de nós removidos (f)")
        ax[k].set_ylabel("S(f) (rede completa, N=3.938)")
    ax[0].axvline(fc_mr, color="gray", ls="--", lw=.8)
    ax[0].text(fc_mr - .02, .55, "$f_c$ (Molloy-Reed)", rotation=90, fontsize=7, ha="right", color="gray")
    ax[0].set_title("(a) Baseline completa, estratégias de remoção")
    ax[0].legend(fontsize=7)
    ax[1].set_xlim(0, .10)
    ax[1].set_title("(b) Zoom em 0 a 10%")
    cm = {
        "Baseline (GC)": "k", "ER": "#1f77b4", "WS": "#2ca02c", "BA": "#d62728", "HK": "#9467bd",
        "CM": "#8c564b", "LFR_0.1": "#ff7f0e"
    }
    for nome, d in res.items():
        ax[2].plot(xg, d["grau"][0], color=cm[nome], lw=2 if "Base" in nome else 1.2, label=nome)
    ax[2].set_xlim(0, .5)
    ax[2].axhline(LIMIAR_FALHA, color="k", ls=":", lw=.8)
    ax[2].set_xlabel("Fração de nós removidos (f)")
    ax[2].set_ylabel("S/S(0)")
    ax[2].set_title("(c) Ataque por grau: baseline vs modelos")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_robustez.png"), dpi=250)
    fig.savefig(os.path.join(a.out, "fig_robustez.pdf"))

    fig, ax = plt.subplots(figsize=(5.2, 4))
    for nome, d in res.items():
        ax.plot(xg, d["aleat"][0], color=cm[nome], lw=2 if "Base" in nome else 1.2, label=nome)
    ax.set_xlabel("Fração de nós removidos (f)")
    ax.set_ylabel("S/S(0)")
    ax.set_title("Remoção aleatória: baseline vs modelos")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_robustez_aleatoria_modelos.png"), dpi=250)

    resumo_out = {
        "N": N, "S0_completa": float(S0), "f_estrela": fe,
        "candidatos_ponte": {
            "n": k_c, "S_apos_remover": float(S_cand),
            "S_apos_remover_k_aleatorios": float(S_rand_k),
            "S_apos_remover_k_maiores_graus": float(S_grau_k),
            "n_pontos_articulacao_entre_candidatos": int(sum(u in pa for u in cand))
        },
        "n_pontos_articulacao_rede_completa": len(pa)
    }
    json.dump(resumo_out, open(os.path.join(a.out, "resumo_robustez.json"), "w"), indent=1)
    print(json.dumps(resumo_out, indent=1))
    print(f"\nConcluído em {time.time()-t0:.0f}s -> {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()