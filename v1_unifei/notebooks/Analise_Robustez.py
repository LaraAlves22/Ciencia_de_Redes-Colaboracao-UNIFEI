import heapq
import random
import time
import os

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

nodes = pd.read_csv("nodes.csv")
edges = pd.read_csv("edges.csv")

INSTITUTOS = ["IEM", "IEPG", "IESTI", "IFQ", "IMC", "IRN", "ISEE", "IEI"]

CORES = {
    "IEM":  "#e6194B",
    "IEPG": "#3cb44b",
    "IESTI":"#4363d8",
    "IFQ":  "#f58231",
    "IMC":  "#911eb4",
    "IRN":  "#42d4f4",
    "ISEE": "#f032e6",
    "IEI":  "#9acd32",
    "Não identificado": "#c9c9c9",
}

def instituto_principal(valor):
    if pd.isna(valor):
        return "Não identificado"
    return valor.split(";")[0].strip()

nodes["instituto_cor"] = nodes["instituto"].apply(instituto_principal)

os.makedirs("/content/figuras", exist_ok=True)

print("Nós:", len(nodes), "| Arestas:", len(edges))

G = nx.Graph()
for _, row in nodes.iterrows():
    G.add_node(row["id"], label=row["label"], instituto=row["instituto_cor"])
for _, row in edges.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["weight"])

def curva_percolacao(G, ordem_remocao):
    """
    Dado um grafo G e uma ordem COMPLETA de remocao (lista com todos os N
    nos, na ordem em que seriam removidos), retorna f, S(f) e numero de
    componentes para TODOS os valores de m = 0, 1, ..., N nos removidos,
    em tempo quase linear.

    Ideia: simular a remocao "de tras para frente" e um Problema de UNIAO
    (adicionar nos), nao de separacao -- e exatamente isso que Union-Find
    resolve com eficiencia.
    """
    N = G.number_of_nodes()
    ordem_adicao = list(reversed(ordem_remocao))

    parent = {}
    size = {}

    def find(x):
        raiz = x
        while parent[raiz] != raiz:
            raiz = parent[raiz]
        while parent[x] != raiz:
            parent[x], x = raiz, parent[x]
        return raiz

    ativos = set()
    max_componente = 0
    n_componentes = 0

    tamanho_max_por_k = np.zeros(N + 1, dtype=np.int64)
    n_comp_por_k = np.zeros(N + 1, dtype=np.int64)

    adj = G.adj

    for k, u in enumerate(ordem_adicao, start=1):
        parent[u] = u
        size[u] = 1
        ativos.add(u)
        n_componentes += 1
        raiz_u = u

        for v in adj[u]:
            if v in ativos:
                raiz_v = find(v)
                raiz_u = find(raiz_u)
                if raiz_u != raiz_v:
                    if size[raiz_u] < size[raiz_v]:
                        raiz_u, raiz_v = raiz_v, raiz_u
                    parent[raiz_v] = raiz_u
                    size[raiz_u] += size[raiz_v]
                    n_componentes -= 1

        max_componente = max(max_componente, size[find(u)])
        tamanho_max_por_k[k] = max_componente
        n_comp_por_k[k] = n_componentes

    # Reindexar por m = numero de nos REMOVIDOS (m = N - k)
    k_idx = N - np.arange(N + 1)
    S_por_m = tamanho_max_por_k[k_idx] / N
    ncomp_por_m = n_comp_por_k[k_idx]
    ncomp_por_m[-1] = 0  # quando m = N (tudo removido), 0 componentes

    f = np.arange(N + 1) / N
    return f, S_por_m, ncomp_por_m

def ordem_remocao_aleatoria(G, seed):
    nos = list(G.nodes())
    rng = random.Random(seed)
    rng.shuffle(nos)
    return nos


def ordem_remocao_grau_adaptativo(G):
    """
    Calcula a ordem de remocao por ataque a hubs, RECALCULANDO o grau a
    cada remocao (ataque adaptativo), usando um heap com "lazy deletion"
    em vez de re-ordenar a lista inteira a cada passo. O(E log N).
    """
    grau_atual = dict(G.degree())
    heap = [(-d, n) for n, d in grau_atual.items()]
    heapq.heapify(heap)
    removido = set()
    ordem = []
    adj = G.adj

    while heap:
        neg_d, n = heapq.heappop(heap)
        if n in removido or -neg_d != grau_atual[n]:
            continue  # entrada obsoleta (grau ja mudou) -> descarta
        removido.add(n)
        ordem.append(n)
        for viz in adj[n]:
            if viz not in removido:
                grau_atual[viz] -= 1
                heapq.heappush(heap, (-grau_atual[viz], viz))
    return ordem


def ordem_remocao_grau_fixo(G):
    """Ranking de grau calculado uma unica vez sobre a rede original."""
    return [n for n, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)]

def distancia_media_amostrada(G, n_amostras=200, seed=0):
    """
    Estima a distancia media do maior componente conexo de G usando BFS
    a partir de uma amostra de nos-fonte, em vez de todos os pares.
    """
    componentes = list(nx.connected_components(G))
    if not componentes:
        return np.nan
    maior = max(componentes, key=len)
    if len(maior) <= 1:
        return np.nan

    sub = G.subgraph(maior)
    rng = random.Random(seed)
    fontes = rng.sample(list(sub.nodes()), min(n_amostras, len(sub)))

    somas = []
    for fonte in fontes:
        distancias = nx.single_source_shortest_path_length(sub, fonte)
        somas.extend(d for alvo, d in distancias.items() if alvo != fonte)

    return float(np.mean(somas)) if somas else np.nan


def distancias_em_checkpoints(G, ordem_remocao, fracoes_checkpoint, n_amostras=200):
    """
    Calcula a distancia media amostrada apenas nas fracoes f indicadas
    (ex.: [0.0, 0.05, 0.10, ..., 0.95]), removendo nos incrementalmente.
    """
    N = G.number_of_nodes()
    H = G.copy()
    resultados = {}
    checkpoints_m = sorted(int(round(fc * N)) for fc in fracoes_checkpoint)

    idx_atual = 0
    for m_alvo in checkpoints_m:
        while idx_atual < m_alvo:
            H.remove_node(ordem_remocao[idx_atual])
            idx_atual += 1
        f_atual = m_alvo / N
        resultados[f_atual] = distancia_media_amostrada(H, n_amostras=n_amostras)

    return resultados

t0 = time.time()
N_EXECUCOES = 30

S_todas = []
ncomp_todas = []
f_ref = None

for execucao in range(N_EXECUCOES):
    ordem = ordem_remocao_aleatoria(G, seed=execucao)
    f, S, ncomp = curva_percolacao(G, ordem)
    if f_ref is None:
        f_ref = f
    S_todas.append(S)
    ncomp_todas.append(ncomp)

S_todas = np.array(S_todas)
ncomp_todas = np.array(ncomp_todas)

df_aleatorio = pd.DataFrame({
    "f": f_ref,
    "S_media": S_todas.mean(axis=0),
    "S_desvio": S_todas.std(axis=0),
    "n_componentes_media": ncomp_todas.mean(axis=0),
})

checkpoints = np.linspace(0, 0.95, 20)
ordem_ref = ordem_remocao_aleatoria(G, seed=0)
dist_aleatorio = distancias_em_checkpoints(G, ordem_ref, checkpoints, n_amostras=200)

print(f"Cenario aleatorio (30 execucoes) concluido em {time.time() - t0:.2f}s")

t0 = time.time()
ordem_grau = ordem_remocao_grau_adaptativo(G)
f_grau, S_grau, ncomp_grau = curva_percolacao(G, ordem_grau)

df_grau = pd.DataFrame({
    "f": f_grau,
    "S": S_grau,
    "n_componentes": ncomp_grau,
})

dist_grau = distancias_em_checkpoints(G, ordem_grau, checkpoints, n_amostras=200)

print(f"Cenario por grau (adaptativo) concluido em {time.time() - t0:.2f}s")


df_aleatorio.to_csv("resultado_remocao_aleatoria.csv", index=False)
df_grau.to_csv("resultado_remocao_grau.csv", index=False)

pd.DataFrame(list(dist_aleatorio.items()), columns=["f", "dist_media"]) \
    .to_csv("distancia_media_aleatoria.csv", index=False)
pd.DataFrame(list(dist_grau.items()), columns=["f", "dist_media"]) \
    .to_csv("distancia_media_grau.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(df_aleatorio["f"], df_aleatorio["S_media"],
        label="Remoção aleatória", color="#1f77b4")
ax.fill_between(
    df_aleatorio["f"],
    df_aleatorio["S_media"] - df_aleatorio["S_desvio"],
    df_aleatorio["S_media"] + df_aleatorio["S_desvio"],
    color="#1f77b4", alpha=0.2,
)

ax.plot(df_grau["f"], df_grau["S"],
        label="Remoção direcionada por grau", color="#d62728")

ax.set_xlabel("Fração de nós removidos (f)")
ax.set_ylabel("Tamanho relativo do componente gigante S(f)")
ax.set_title("Robustez da rede de colaboração científica — UNIFEI")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("figura_robustez_S_f.png", dpi=300)
print("Figura salva em figura_robustez_S_f.png")

def limiar_percolacao_molloy_reed(G):
    graus = np.array([d for _, d in G.degree()])
    k1 = graus.mean()
    k2 = (graus ** 2).mean()
    kappa = k2 / k1
    f_c = 1 - 1 / (kappa - 1)
    return f_c, k1, k2


f_c, k1, k2 = limiar_percolacao_molloy_reed(G)
print(f"\nCriterio de Molloy-Reed: <k>={k1:.3f}, <k^2>={k2:.3f}, f_c (teorico) = {f_c:.4f}")