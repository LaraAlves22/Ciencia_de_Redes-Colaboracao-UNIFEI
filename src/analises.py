# ============================================================================
# Analises da rede de cooperacao do IMC: metricas (micro/meso/macro), testes das
# hipoteses (QP principal, QP1-QP3), afastamento de docentes, sensibilidade e figuras.
# ============================================================================
import json
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score

ALFA = 0.05
CATEGORIAS = ["docente_outro_instituto", "outro_vinculo_unifei", "nacional", "estrangeira", "nao_identificado"]
TIPO_GRUPO = {"artigo_periodico": "P", "trabalho_completo_evento": "P", "livro_capitulo": "P",
              "orientacao": "F", "coorientacao": "F", "banca": "F",
              "projeto_pesquisa": "J", "projeto_extensao": "J"}
CORES = {"Matemática": "#2b6cb0", "Computação": "#dd6b20"}
CORES_CAT = {"docente_outro_instituto": "#38a169", "outro_vinculo_unifei": "#9ae6b4", "nacional": "#a0aec0",
             "estrangeira": "#805ad5", "nao_identificado": "#e2e8f0"}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})


# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------
def carregar_rede(pasta, modelo="M2", variante=""):
    nos = pd.read_csv(os.path.join(pasta, f"nos{variante}.csv"), dtype=str, keep_default_na=False)
    arestas = pd.read_csv(os.path.join(pasta, f"arestas{variante}.csv"))
    return montar_grafo(nos, arestas, modelo), nos, arestas


def montar_grafo(nos, arestas, modelo):
    G = nx.Graph()
    for r in nos.itertuples():
        G.add_node(r.id, tipo_no=r.tipo_no, camada=r.camada, categoria=r.categoria, instituto=r.instituto,
                   rotulo=r.rotulo)
    for r in arestas.itertuples():
        w = getattr(r, modelo)
        if w > 0:
            G.add_edge(r.source, r.target, weight=float(w), n_eventos=int(r.n_eventos), M1=float(r.M1))
    G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0 and G.nodes[n]["tipo_no"] == "X"])
    return G


def docentes_de(G):
    return [n for n, d in G.nodes(data=True) if d["tipo_no"] == "D"]


def subrede_docentes(G):
    return G.subgraph(docentes_de(G)).copy()


TAMANHO_GRUPO = Counter(TIPO_GRUPO.values())


def recalcular_m2(arestas, pesos):
    """Recalcula M2 a partir das contagens por tipo (para a varredura de pesos), com cada
    grupo entrando pela media de seus tipos, como no notebook 03."""
    tipos = [c[2:] for c in arestas.columns if c.startswith("n_") and c != "n_eventos"]
    tot = defaultdict(lambda: defaultdict(float))
    for r in arestas.itertuples():
        for t in tipos:
            v = getattr(r, "n_" + t)
            tot[r.source][t] += v
            tot[r.target][t] += v
    m2 = []
    for r in arestas.itertuples():
        s = 0.0
        for t in tipos:
            v = getattr(r, "n_" + t)
            if v > 0:
                g = TIPO_GRUPO[t]
                s += pesos[g] / TAMANHO_GRUPO[g] * 2 * v / (tot[r.source][t] + tot[r.target][t])
        m2.append(s)
    out = arestas.copy()
    out["M2"] = m2
    return out


# ----------------------------------------------------------------------------
# Comunidades (rede de docentes)
# ----------------------------------------------------------------------------
def melhor_louvain(H, n_exec=20, peso="weight"):
    if H.number_of_edges() == 0:
        return [set(H)], 0.0, 1.0
    parts = [louvain_communities(H, weight=peso, seed=s) for s in range(n_exec)]
    qs = [modularity(H, p, weight=peso) for p in parts]
    melhor = parts[int(np.argmax(qs))]
    rot = [rotulos_particao(p, list(H)) for p in parts]
    nmis = [normalized_mutual_info_score(rot[i], rot[j]) for i in range(n_exec) for j in range(i + 1, n_exec)]
    return melhor, float(max(qs)), float(np.mean(nmis)) if nmis else 1.0


def rotulos_particao(part, ordem):
    m = {n: i for i, c in enumerate(sorted(part, key=lambda c: (-len(c), sorted(c)[0]))) for n in c}
    return [m.get(n, -1) for n in ordem]


def rewire_grau(H, seed, n_trocas=None):
    """Modelo nulo de configuracao: troca dupla de arestas preservando o grau; pesos
    redistribuidos aleatoriamente entre as arestas."""
    R = nx.Graph()
    R.add_nodes_from(H.nodes(data=True))
    R.add_edges_from(H.edges())
    m = R.number_of_edges()
    if m >= 2:
        try:
            nx.double_edge_swap(R, nswap=n_trocas or 10 * m, max_tries=100 * m, seed=seed)
        except nx.NetworkXAlgorithmError:
            pass
    rng = np.random.default_rng(seed)
    pesos = [d["weight"] for _, _, d in H.edges(data=True)]
    evs = [d["n_eventos"] for _, _, d in H.edges(data=True)]
    idx = rng.permutation(len(pesos))
    for k, (u, v) in enumerate(R.edges()):
        R[u][v]["weight"] = pesos[idx[k]]
        R[u][v]["n_eventos"] = evs[idx[k]]
    return R


def rewire_fronteira(G, seed):
    """Nulo para a rede completa respeitando a fronteira de um passo: D-D por troca dupla
    dentro dos docentes; D-X trocando os externos entre arestas D-X (graus preservados,
    nenhuma aresta X-X criada)."""
    rng = np.random.default_rng(seed)
    D = set(docentes_de(G))
    GD = G.subgraph(D).copy()
    R = rewire_grau(GD, seed)
    dx = [(u, v) if u in D else (v, u) for u, v in G.edges() if (u in D) != (v in D)]
    dx = [list(e) for e in dx]
    existentes = {(d, x) for d, x in dx}
    m = len(dx)
    for _ in range(10 * m):
        a, b = rng.integers(0, m, 2)
        (d1, x1), (d2, x2) = dx[a], dx[b]
        if d1 == d2 or x1 == x2 or (d1, x2) in existentes or (d2, x1) in existentes:
            continue
        existentes -= {(d1, x1), (d2, x2)}
        existentes |= {(d1, x2), (d2, x1)}
        dx[a][1], dx[b][1] = x2, x1
    R.add_nodes_from(G.nodes(data=True))
    R.add_edges_from((d, x, {"weight": 1.0, "n_eventos": 1}) for d, x in dx)
    return R


# ----------------------------------------------------------------------------
# Metricas por docente (nivel micro)
# ----------------------------------------------------------------------------
def anos_exposicao(docentes_csv, inicio, fim):
    exp = {}
    for r in docentes_csv.itertuples():
        try:
            ing = int(r.ano_ingresso_unifei)
        except (TypeError, ValueError):
            ing = inicio
        exp[r.id_lattes] = max(0, fim - max(inicio, ing) + 1)
    return exp


def metricas_docentes(G, docentes_csv, inicio, fim, comunidades=None):
    D = docentes_de(G)
    GD = G.subgraph(D)
    bet = nx.betweenness_centrality(G, normalized=True)
    bet_d = nx.betweenness_centrality(GD, normalized=True) if GD.number_of_edges() else {d: 0 for d in D}
    exp = anos_exposicao(docentes_csv, inicio, fim)
    prod = {r.id_lattes: int(r.n_publicacoes_periodo or 0) for r in docentes_csv.itertuples()}
    com = {}
    if comunidades:
        for i, c in enumerate(sorted(comunidades, key=len, reverse=True)):
            for n in c:
                com[n] = i
    linhas = []
    for d in D:
        nd = G.nodes[d]
        viz = list(G.neighbors(d))
        vd = [v for v in viz if G.nodes[v]["tipo_no"] == "D"]
        vx = [v for v in viz if G.nodes[v]["tipo_no"] == "X"]
        cats = Counter(G.nodes[v]["categoria"] for v in vx)
        insts = Counter(G.nodes[v]["instituto"] for v in vx if G.nodes[v]["categoria"] == "docente_outro_instituto")
        grupos = Counter(G.nodes[v]["camada"] for v in vd)
        grupos.update(G.nodes[v]["categoria"] for v in vx)
        k = sum(grupos.values())
        diversidade = 1 - sum((c / k) ** 2 for c in grupos.values()) if k else 0.0
        linha = {
            "id": d, "docente": nd["rotulo"], "camada": nd["camada"],
            "grau_total": len(viz), "grau_docentes_imc": len(vd),
            "grau_mesma_area": sum(1 for v in vd if G.nodes[v]["camada"] == nd["camada"]),
            "grau_outra_area": sum(1 for v in vd if G.nodes[v]["camada"] != nd["camada"]),
            "grau_externo": len(vx),
            "forca": sum(G[d][v]["weight"] for v in viz),
            "forca_docentes_imc": sum(G[d][v]["weight"] for v in vd),
            "intermediacao_rede": bet[d], "intermediacao_docentes": bet_d.get(d, 0.0),
            "diversidade_vinculos": diversidade, "comunidade": com.get(d, -1),
            "institutos_unifei": ";".join(f"{i}:{c}" for i, c in insts.most_common()),
            "n_institutos_unifei": len(insts),
            "publicacoes_periodo": prod.get(d, np.nan), "anos_exposicao": exp.get(d, np.nan),
        }
        for c in CATEGORIAS:
            linha[f"ext_{c}"] = cats.get(c, 0)
        e = linha["anos_exposicao"]
        linha["publicacoes_por_ano"] = linha["publicacoes_periodo"] / e if e and e > 0 else np.nan
        linhas.append(linha)
    return pd.DataFrame(linhas).sort_values("docente").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Testes das hipoteses
# ----------------------------------------------------------------------------
def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def teste_qp1(G, n_perm=10000, seed=0):
    """H1: coautorias mais frequentes dentro de cada area do que entre areas.
    Nulo: permutacao dos rotulos de camada entre docentes (preserva grafo, graus e tamanho das camadas)."""
    GD = subrede_docentes(G)
    nos = list(GD)
    cam = np.array([GD.nodes[n]["camada"] for n in nos])
    idx = {n: i for i, n in enumerate(nos)}
    E = np.array([(idx[u], idx[v]) for u, v in GD.edges()])
    W = np.array([GD[u][v]["weight"] for u, v in GD.edges()])
    if len(E) == 0:
        return {"erro": "rede de docentes sem arestas"}
    inter = lambda c: (c[E[:, 0]] != c[E[:, 1]])
    obs = inter(cam).mean()
    obs_w = (W * inter(cam)).sum() / W.sum()
    rng = np.random.default_rng(seed)
    nulo = np.empty(n_perm)
    nulo_w = np.empty(n_perm)
    for k in range(n_perm):
        c = rng.permutation(cam)
        m = inter(c)
        nulo[k], nulo_w[k] = m.mean(), (W * m).sum() / W.sum()
    p = (1 + (nulo <= obs).sum()) / (n_perm + 1)
    p_w = (1 + (nulo_w <= obs_w).sum()) / (n_perm + 1)
    pares_inter = [(GD.nodes[u]["rotulo"], GD.nodes[v]["rotulo"]) for u, v in GD.edges()
                   if GD.nodes[u]["camada"] != GD.nodes[v]["camada"]]
    return {"fracao_inter_obs": float(obs), "fracao_inter_nulo_media": float(nulo.mean()),
            "fracao_inter_nulo_ic95": [float(np.quantile(nulo, .025)), float(np.quantile(nulo, .975))],
            "p_valor": float(p), "fracao_inter_ponderada_obs": float(obs_w),
            "fracao_inter_ponderada_nulo_media": float(nulo_w.mean()), "p_valor_ponderado": float(p_w),
            "assortatividade_camada": float(nx.attribute_assortativity_coefficient(GD, "camada")),
            "n_arestas_DD": int(len(E)), "n_arestas_inter": len(pares_inter), "pares_inter": pares_inter,
            "decisao": "rejeita H0" if p < ALFA and len(pares_inter) > 0 else "não rejeita H0",
            "_nulo": nulo}


def benjamini_hochberg(p):
    """Valores-p ajustados de Benjamini-Hochberg (controle da taxa de falsas descobertas)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    ordem = np.argsort(p)
    aj = p[ordem] * n / np.arange(1, n + 1)
    aj = np.minimum.accumulate(aj[::-1])[::-1]
    out = np.empty(n)
    out[ordem] = np.minimum(aj, 1.0)
    return out


def teste_qp2(G, met, docentes_csv, inicio, fim, n_rep=1000, seed=0):
    """H1: vinculos externos e posicoes de ligacao nao se distribuem uniformemente.
    (a) concentracao (Gini) dos vinculos externos vs alocacao proporcional aos anos de exposicao;
    (b) docentes de ligacao: intermediacao acima do esperado sob nulo que preserva graus e fronteira,
        com valor-p empirico por docente corrigido por Benjamini-Hochberg (48 testes).
    Rejeita H0 se p_Gini < ALFA e ha ao menos um docente de ligacao."""
    rng = np.random.default_rng(seed)
    exp = anos_exposicao(docentes_csv, inicio, fim)
    ids = list(met["id"])
    ext = met.set_index("id")[["ext_docente_outro_instituto", "ext_outro_vinculo_unifei", "ext_nacional",
                               "ext_estrangeira"]].sum(axis=1)
    pesos = np.array([max(exp.get(i, 1), 0) for i in ids], dtype=float)
    pesos = pesos / pesos.sum() if pesos.sum() else np.ones(len(ids)) / len(ids)
    total = int(ext.sum())
    g_obs = gini(ext.loc[ids].values)
    g_nulo = np.array([gini(rng.multinomial(total, pesos)) for _ in range(5000)])
    p_gini = (1 + (g_nulo >= g_obs).sum()) / (len(g_nulo) + 1)
    # intermediacao na rede sem colaboradores exclusivos (externos de grau 1, que nao ligam
    # ninguem), comparada a um nulo que preserva graus e a fronteira de um passo
    D = docentes_de(G)
    Gp = G.copy()
    Gp.remove_nodes_from([x for x in list(Gp) if Gp.nodes[x]["tipo_no"] == "X" and Gp.degree(x) <= 1])
    obs_b = nx.betweenness_centrality(Gp, normalized=True)
    amostras = defaultdict(list)
    for r in range(n_rep):
        R = rewire_fronteira(Gp, seed + r)
        b = nx.betweenness_centrality(R, normalized=True)
        for d in D:
            amostras[d].append(b.get(d, 0.0))
    z, p_emp = {}, {}
    for d in D:
        a_ = np.array(amostras[d])
        z[d] = (obs_b[d] - a_.mean()) / a_.std() if a_.std() > 0 else 0.0
        p_emp[d] = (1 + (a_ >= obs_b[d]).sum()) / (len(a_) + 1)
    p_aj = dict(zip(D, benjamini_hochberg([p_emp[d] for d in D])))
    met = met.copy()
    met["intermediacao_ligacao"] = met["id"].map(obs_b)
    met["intermediacao_z"] = met["id"].map(z)
    met["intermediacao_p"] = met["id"].map(p_emp)
    met["intermediacao_p_bh"] = met["id"].map(p_aj)
    met["docente_ligacao"] = (met["intermediacao_p_bh"] < ALFA) & (met["intermediacao_ligacao"] > 0)
    # criterio anterior (z >= 2, sem correcao), mantido so para comparacao
    met["docente_ligacao_z2"] = (met["intermediacao_z"] >= 2) & (met["intermediacao_ligacao"] > 0)
    n_lig = int(met["docente_ligacao"].sum())
    decisao = "rejeita H0" if (p_gini < ALFA and n_lig > 0) else "não rejeita H0"
    return met, {"externos_compartilhados": int(sum(1 for x in Gp if Gp.nodes[x]["tipo_no"] == "X")),
                 "gini_vinculos_externos_obs": g_obs, "gini_nulo_media": float(g_nulo.mean()),
                 "p_valor_gini": float(p_gini), "n_docentes_ligacao": n_lig,
                 "docentes_ligacao": list(met.loc[met["docente_ligacao"], "docente"]),
                 "n_docentes_ligacao_criterio_z2_sem_correcao": int(met["docente_ligacao_z2"].sum()),
                 "replicas_nulo_intermediacao": n_rep, "correcao": "Benjamini-Hochberg", "decisao": decisao}


def teste_qp3(G, n_rep=200, n_exec=20, seed=0):
    """H1: ha subconjuntos de docentes com coesao e recorrencia acima do esperado.
    Coesao: modularidade (Louvain) vs modelo de configuracao. Recorrencia: n medio de eventos
    nas arestas intracomunidade vs permutacao dos valores entre as arestas."""
    GD = subrede_docentes(G)
    GD.remove_nodes_from([n for n in list(GD) if GD.degree(n) == 0])
    part, q_obs, estab = melhor_louvain(GD, n_exec)
    q_nulo = []
    for r in range(n_rep):
        R = rewire_grau(GD, seed + r)
        q_nulo.append(melhor_louvain(R, n_exec=3)[1])
    q_nulo = np.array(q_nulo)
    p_q = (1 + (q_nulo >= q_obs).sum()) / (n_rep + 1)
    com = {n: i for i, c in enumerate(part) for n in c}
    intra = np.array([com[u] == com[v] for u, v in GD.edges()])
    ev = np.array([GD[u][v]["n_eventos"] for u, v in GD.edges()], dtype=float)
    rec_obs = ev[intra].mean() if intra.any() else 0.0
    rng = np.random.default_rng(seed)
    rec_nulo = np.array([rng.permutation(ev)[intra].mean() for _ in range(10000)]) if intra.any() else np.array([0])
    p_rec = (1 + (rec_nulo >= rec_obs).sum()) / (len(rec_nulo) + 1)
    cam = [GD.nodes[n]["camada"] for n in GD]
    nmi_cam = normalized_mutual_info_score(cam, rotulos_particao(part, list(GD)))
    grupos = []
    for i, c in enumerate(sorted(part, key=len, reverse=True)):
        S = GD.subgraph(c)
        grupos.append({"comunidade": i, "n_docentes": len(c),
                       "camadas": dict(Counter(GD.nodes[n]["camada"] for n in c)),
                       "densidade_interna": nx.density(S) if len(c) > 1 else 0.0,
                       "eventos_medios_por_par": float(np.mean([S[u][v]["n_eventos"] for u, v in S.edges()])) if S.number_of_edges() else 0.0,
                       "docentes": sorted(GD.nodes[n]["rotulo"] for n in c)})
    decisao = "rejeita H0" if p_q < ALFA and p_rec < ALFA else (
        "rejeita H0 (só coesão)" if p_q < ALFA else "não rejeita H0")
    return part, {"modularidade_obs": q_obs, "modularidade_nulo_media": float(q_nulo.mean()),
                  "p_valor_modularidade": float(p_q), "estabilidade_nmi_louvain": estab,
                  "eventos_medios_intra_obs": float(rec_obs), "eventos_medios_intra_nulo": float(rec_nulo.mean()),
                  "p_valor_recorrencia": float(p_rec), "nmi_comunidade_camada": float(nmi_cam),
                  "n_comunidades": len(part), "grupos": grupos, "decisao": decisao, "_q_nulo": q_nulo}


def teste_principal(met, n_boot=10000, seed=0):
    """H1: a estrutura revela padroes nao descritos pela contagem de publicacoes.
    Regra declarada: rejeita H0 se o limite superior do IC 95% (bootstrap) de
    rho(producao, intermediacao de ligacao) for < 0,7 -- inicio da faixa de correlacao forte
    (Schober et al., 2018) -- e houver docente de ligacao com producao <= mediana."""
    m = met.dropna(subset=["publicacoes_periodo"])
    corr = {}
    for c in ["grau_total", "grau_docentes_imc", "grau_externo", "intermediacao_ligacao", "diversidade_vinculos",
              "n_institutos_unifei"]:
        rho, p = spearmanr(m["publicacoes_periodo"], m[c])
        corr[c] = {"rho": float(rho), "p": float(p)}
    x = m["publicacoes_periodo"].to_numpy(dtype=float)
    y = m["intermediacao_ligacao"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        r_ = spearmanr(x[idx], y[idx])[0]
        if not np.isnan(r_):
            boot.append(r_)
    ic = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
    med = m["publicacoes_periodo"].median()
    lig = m.get("docente_ligacao", pd.Series(False, index=m.index)).astype(bool)
    abaixo = m[(m["publicacoes_periodo"] <= med) & lig]
    # papel relacional mais amplo (ligacao, vinculo entre areas ou outro instituto), so descritivo
    amplo = m[(m["publicacoes_periodo"] <= med) & (lig | (m["grau_outra_area"] > 0) | (m["n_institutos_unifei"] > 0))]
    return {"correlacoes_spearman_com_producao": corr, "rho_intermediacao_ic95_bootstrap": ic,
            "replicas_bootstrap": len(boot), "mediana_publicacoes": float(med),
            "docentes_ligacao_abaixo_mediana": list(abaixo["docente"]),
            "docentes_abaixo_mediana_com_papel_relacional_amplo": list(amplo["docente"]),
            "decisao": "rejeita H0" if ic[1] < 0.7 and len(abaixo) > 0 else "não rejeita H0"}


# ----------------------------------------------------------------------------
# Afastamento de docentes (aposentadoria, licenca, redistribuicao, obito)
# ----------------------------------------------------------------------------
def afastamento_individual(G):
    D = docentes_de(G)
    GD = G.subgraph(D).copy()
    comp_principal = max(nx.connected_components(GD), key=len) if GD.number_of_nodes() else set()
    base_gc = len(comp_principal)
    base_ef = nx.global_efficiency(GD)
    base_inter = sum(1 for u, v in GD.edges() if GD.nodes[u]["camada"] != GD.nodes[v]["camada"])
    linhas = []
    for d in D:
        H = GD.copy()
        H.remove_node(d)
        gc = max((len(c) for c in nx.connected_components(H)), default=0)
        inter = sum(1 for u, v in H.edges() if H.nodes[u]["camada"] != H.nodes[v]["camada"])
        orfaos = [x for x in G.neighbors(d) if G.nodes[x]["tipo_no"] == "X" and
                  all(G.nodes[y]["tipo_no"] == "X" or y == d for y in G.neighbors(x))]
        cats = Counter(G.nodes[x]["categoria"] for x in orfaos)
        insts = Counter(G.nodes[x]["instituto"] for x in orfaos if G.nodes[x]["categoria"] == "docente_outro_instituto")
        linhas.append({"id": d, "docente": G.nodes[d]["rotulo"], "camada": G.nodes[d]["camada"],
                       "docentes_desconectados": base_gc - gc - (1 if d in comp_principal else 0),
                       "perda_eficiencia": (base_ef - nx.global_efficiency(H)) / base_ef if base_ef else 0.0,
                       "vinculos_interarea_perdidos": base_inter - inter,
                       "externos_que_perdem_vinculo_com_imc": len(orfaos),
                       "docentes_outros_institutos_perdidos": cats.get("docente_outro_instituto", 0),
                       "institutos_unifei_perdidos": ";".join(f"{i}:{c}" for i, c in insts.most_common())})
    return pd.DataFrame(linhas).sort_values(["docentes_desconectados", "perda_eficiencia",
                                            "vinculos_interarea_perdidos"], ascending=False)


def afastamento_sequencial(G, n_rand=100, seed=0):
    GD = subrede_docentes(G)
    n = GD.number_of_nodes()

    def curva(ordem_fn):
        H = GD.copy()
        out = [max(len(c) for c in nx.connected_components(H)) / n]
        for _ in range(n - 1):
            v = ordem_fn(H)
            H.remove_node(v)
            out.append(max((len(c) for c in nx.connected_components(H)), default=0) / n)
        return np.array(out)

    grau = curva(lambda H: max(H.degree(), key=lambda kv: (kv[1], kv[0]))[0])
    rng = np.random.default_rng(seed)
    aleat = np.array([curva(lambda H: list(H)[rng.integers(H.number_of_nodes())]) for _ in range(n_rand)])
    return grau, aleat


# ----------------------------------------------------------------------------
# Sensibilidade
# ----------------------------------------------------------------------------
def sensibilidade(pasta, base_modelo="M2", n_exec=10):
    resultados = []
    variantes = {}
    for modelo in ("M1", "M2", "M3"):
        variantes[f"{modelo}"] = carregar_rede(pasta, modelo)[0]
    if os.path.exists(os.path.join(pasta, "arestas_estrito.csv")):
        variantes[f"{base_modelo}_estrito"] = carregar_rede(pasta, base_modelo, "_estrito")[0]
    for v in ("_trig07", "_trig09"):  # limiar de trigramas de 0,7 e 0,9 (notebooks 01 e 03)
        if os.path.exists(os.path.join(pasta, f"arestas{v}.csv")):
            variantes[f"{base_modelo}{v}"] = carregar_rede(pasta, base_modelo, v)[0]
    nos = pd.read_csv(os.path.join(pasta, "nos.csv"), dtype=str, keep_default_na=False)
    if os.path.exists(os.path.join(pasta, "arestas_sem_massivos.csv")):
        variantes[f"{base_modelo}_sem_massivos"] = montar_grafo(
            nos, pd.read_csv(os.path.join(pasta, "arestas_sem_massivos.csv")), base_modelo)
    part, forca = {}, {}
    for nome, G in variantes.items():
        GD = subrede_docentes(G)
        GD.remove_nodes_from([v for v in list(GD) if GD.degree(v) == 0])
        part[nome] = (melhor_louvain(GD, n_exec)[0], GD)
        forca[nome] = {d: sum(dd["weight"] for _, _, dd in G.edges(d, data=True)) for d in docentes_de(G)}
    nomes = list(variantes)
    nmi = pd.DataFrame(index=nomes, columns=nomes, dtype=float)
    rho = pd.DataFrame(index=nomes, columns=nomes, dtype=float)
    for a in nomes:
        for b in nomes:
            comuns = sorted(set(part[a][1]) & set(part[b][1]))
            nmi.loc[a, b] = normalized_mutual_info_score(rotulos_particao(part[a][0], comuns),
                                                         rotulos_particao(part[b][0], comuns))
            ds = sorted(set(forca[a]) & set(forca[b]))
            rho.loc[a, b] = spearmanr([forca[a][d] for d in ds], [forca[b][d] for d in ds])[0]
    # varredura dos pesos do M2 (simplex, passo 0,1)
    arestas = pd.read_csv(os.path.join(pasta, "arestas.csv"))
    G0 = montar_grafo(nos, arestas, "M2")
    GD0 = subrede_docentes(G0)
    GD0.remove_nodes_from([v for v in list(GD0) if GD0.degree(v) == 0])
    p0 = melhor_louvain(GD0, n_exec)[0]
    for wp in np.round(np.arange(0, 1.01, 0.1), 1):
        for wf in np.round(np.arange(0, 1.01 - wp, 0.1), 1):
            wj = round(1 - wp - wf, 1)
            a2 = recalcular_m2(arestas, {"P": wp, "F": wf, "J": wj})
            G2 = montar_grafo(nos, a2, "M2")
            GD2 = subrede_docentes(G2)
            GD2.remove_nodes_from([v for v in list(GD2) if GD2.degree(v) == 0])
            if GD2.number_of_edges() == 0:
                continue
            p2, q2, _ = melhor_louvain(GD2, n_exec=3)
            comuns = sorted(set(GD0) & set(GD2))
            resultados.append({"peso_P": wp, "peso_F": wf, "peso_J": wj, "modularidade": q2,
                               "n_comunidades": len(p2),
                               "nmi_vs_pesos_base": normalized_mutual_info_score(
                                   rotulos_particao(p0, comuns), rotulos_particao(p2, comuns))})
    return nmi, rho, pd.DataFrame(resultados)


# ----------------------------------------------------------------------------
# Figuras
# ----------------------------------------------------------------------------
def fig_rede(G, comunidades, caminho):
    """Docentes em arco interno (Matematica a esquerda, Computacao a direita, ordenados por
    comunidade); externos em aneis externos por categoria, no angulo medio dos seus docentes."""
    D = docentes_de(G)
    com = {n: i for i, c in enumerate(sorted(comunidades, key=len, reverse=True)) for n in c}
    pos, ang = {}, {}
    for camada, (a0, a1) in {"Matemática": (100, 260), "Computação": (-80, 80)}.items():
        ds = sorted([d for d in D if G.nodes[d]["camada"] == camada], key=lambda d: (com.get(d, 99), G.nodes[d]["rotulo"]))
        for k, d in enumerate(ds):
            a = np.deg2rad(a0 + (a1 - a0) * (k + 0.5) / max(len(ds), 1))
            ang[d] = a
            pos[d] = (np.cos(a), np.sin(a))
    raio = {"docente_outro_instituto": 1.55, "outro_vinculo_unifei": 1.8, "nacional": 2.05, "estrangeira": 2.25,
            "nao_identificado": 2.45}
    rng = np.random.default_rng(0)
    for x in G:
        if G.nodes[x]["tipo_no"] != "X":
            continue
        vs = [ang[d] for d in G.neighbors(x) if d in ang]
        if not vs:
            continue
        a = np.arctan2(np.mean(np.sin(vs)), np.mean(np.cos(vs))) + rng.normal(0, 0.03)
        r = raio.get(G.nodes[x]["categoria"], 2.45) + rng.normal(0, 0.04)
        pos[x] = (r * np.cos(a), r * np.sin(a))
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    for u, v in G.edges():
        if u not in pos or v not in pos:
            continue
        du, dv = G.nodes[u]["tipo_no"] == "D", G.nodes[v]["tipo_no"] == "D"
        if du and dv:
            cor = "#c53030" if G.nodes[u]["camada"] != G.nodes[v]["camada"] else "#4a5568"
            ax.plot(*zip(pos[u], pos[v]), color=cor, lw=0.4 + 1.5 * min(G[u][v]["weight"], 1), alpha=0.8, zorder=2)
        else:
            ax.plot(*zip(pos[u], pos[v]), color="#cbd5e0", lw=0.25, alpha=0.35, zorder=1)
    for cat, cor in CORES_CAT.items():
        xs = [pos[x] for x in G if x in pos and G.nodes[x]["tipo_no"] == "X" and G.nodes[x]["categoria"] == cat]
        if xs:
            ax.scatter(*zip(*xs), s=4, color=cor, edgecolors="none", zorder=3, label=cat.replace("_", " "))
    for camada, cor in CORES.items():
        ds = [d for d in D if G.nodes[d]["camada"] == camada]
        ax.scatter([pos[d][0] for d in ds], [pos[d][1] for d in ds], s=[20 + 3 * G.degree(d) ** 0.8 for d in ds],
                   color=cor, edgecolors="white", linewidths=0.6, zorder=4, label=f"Docente – {camada}")
    from matplotlib.lines import Line2D
    rotulos_cat = {"docente_outro_instituto": "docente de outro instituto UNIFEI",
                   "outro_vinculo_unifei": "outro vínculo UNIFEI", "nacional": "outra instituição nacional",
                   "estrangeira": "instituição estrangeira", "nao_identificado": "afiliação não identificada"}
    alcas = [Line2D([], [], marker="o", ls="", color=c, markersize=8, label=f"Docente – {k}") for k, c in CORES.items()]
    alcas += [Line2D([], [], marker="o", ls="", color=CORES_CAT[c], markersize=4, label=rotulos_cat[c])
              for c in CORES_CAT if any(G.nodes[x]["categoria"] == c for x in G if G.nodes[x]["tipo_no"] == "X")]
    alcas += [Line2D([], [], color="#4a5568", label="docente–docente, mesma área"),
              Line2D([], [], color="#c53030", label="docente–docente, entre áreas")]
    ax.legend(handles=alcas, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_matriz(G, caminho):
    D = docentes_de(G)
    colunas = ["Matemática", "Computação", "IEM", "IEPG", "IESTI", "IFQ", "IRN", "ISEE", "IEI",
               "outro vínculo UNIFEI", "nacional", "estrangeira", "não identificado"]
    nomes_cat = {"outro_vinculo_unifei": "outro vínculo UNIFEI", "nacional": "nacional", "estrangeira": "estrangeira",
                 "nao_identificado": "não identificado"}
    M = pd.DataFrame(0.0, index=["Matemática", "Computação"], columns=colunas)
    for d in D:
        lin = G.nodes[d]["camada"]
        for v in G.neighbors(d):
            nv = G.nodes[v]
            if nv["tipo_no"] == "D":
                M.loc[lin, nv["camada"]] += 0.5 if nv["camada"] == lin else 1
            elif nv["categoria"] == "docente_outro_instituto" and nv["instituto"] in M.columns:
                M.loc[lin, nv["instituto"]] += 1
            else:
                M.loc[lin, nomes_cat.get(nv["categoria"], "não identificado")] += 1
    M = M.round().astype(int)
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.imshow(np.log1p(M.values), cmap="Blues", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, M.iat[i, j], ha="center", va="center", fontsize=7,
                    color="white" if np.log1p(M.iat[i, j]) > np.log1p(M.values.max()) * 0.6 else "black")
    ax.set_xticks(range(len(colunas)), colunas, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(2), M.index)
    ax.set_title("Vínculos (pares distintos) dos docentes por área de origem e tipo de colaborador", fontsize=8)
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return M


def fig_nulos(res1, res3, caminho):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4))
    axs[0].hist(res1["_nulo"], bins=30, color="#cbd5e0")
    axs[0].axvline(res1["fracao_inter_obs"], color="#c53030", lw=1.5, label="observado")
    axs[0].set_xlabel("Fração de arestas D–D entre áreas")
    axs[0].set_ylabel("Frequência")
    axs[0].set_title(f"QP1: permutação de camadas (p={res1['p_valor']:.4f})", fontsize=8)
    axs[0].legend(frameon=False, fontsize=7)
    axs[1].hist(res3["_q_nulo"], bins=30, color="#cbd5e0")
    axs[1].axvline(res3["modularidade_obs"], color="#c53030", lw=1.5)
    axs[1].set_xlabel("Modularidade (Louvain)")
    axs[1].set_title(f"QP3: modelo de configuração (p={res3['p_valor_modularidade']:.4f})", fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_comunidades(G, comunidades, met, caminho):
    GD = subrede_docentes(G)
    conectados = [n for n in GD if GD.degree(n) > 0]
    pos = {}
    comps = sorted(nx.connected_components(GD.subgraph(conectados)), key=len, reverse=True) if conectados else []
    for k, c in enumerate(comps):
        S = GD.subgraph(c)
        p = nx.spring_layout(S, weight=None, k=1.8 / np.sqrt(len(S)), iterations=2000, seed=3) if len(S) > 1 \
            else {next(iter(c)): np.zeros(2)}
        xy = np.array(list(p.values()), dtype=float)
        esc = np.abs(xy - xy.mean(0)).max() or 1.0
        xy = (xy - xy.mean(0)) / esc
        if k == 0:
            pos.update(zip(p, xy))
        else:
            pos.update(zip(p, xy * 0.12 + np.array([-1.35, 0.9 - 0.4 * (k - 1)])))
    isolados = [n for n in GD if GD.degree(n) == 0]
    for k, n in enumerate(sorted(isolados)):
        pos[n] = np.array([1.25, 1.0 - 0.25 * k])
    com = {n: i for i, c in enumerate(sorted(comunidades, key=len, reverse=True)) for n in c}
    cmap = plt.get_cmap("tab10")
    lig = set(met.loc[met.get("docente_ligacao", False) == True, "id"]) if "docente_ligacao" in met else set()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for u, v in GD.edges():
        inter = GD.nodes[u]["camada"] != GD.nodes[v]["camada"]
        ax.plot(*zip(pos[u], pos[v]), color="#c53030" if inter else "#a0aec0",
                lw=1.2 if inter else 0.3 + 2 * min(GD[u][v]["weight"], 1), alpha=0.9 if inter else 0.6,
                zorder=1.5 if inter else 1)
    for camada, marc in {"Matemática": "o", "Computação": "s"}.items():
        ds = [d for d in GD if GD.nodes[d]["camada"] == camada]
        ax.scatter([pos[d][0] for d in ds], [pos[d][1] for d in ds], marker=marc, s=70,
                   c=[cmap(com.get(d, 9) % 10) if com.get(d, -1) >= 0 else "#e2e8f0" for d in ds],
                   edgecolors=["black" if d in lig else "white" for d in ds],
                   linewidths=[1.6 if d in lig else 0.6 for d in ds], zorder=2)
    for d in GD:
        ax.text(pos[d][0], pos[d][1] + 0.04, nome_curto(GD.nodes[d]["rotulo"]), fontsize=5.5, ha="center", zorder=3)
    if isolados:
        ax.text(1.25, 1.12, "sem cooperação\ncom docentes", fontsize=6, ha="center", color="#718096")
    ax.scatter([], [], marker="o", color="#718096", label="Matemática")
    ax.scatter([], [], marker="s", color="#718096", label="Computação")
    ax.scatter([], [], marker="o", facecolors="none", edgecolors="black", linewidths=1.6, label="docente de ligação")
    ax.plot([], [], color="#c53030", label="cooperação entre áreas")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4)
    ax.axis("off")
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_afastamento(grau, aleat, indiv, caminho):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    f = np.arange(len(grau)) / len(grau)
    axs[0].plot(f, grau, color="#c53030", label="maior grau primeiro")
    axs[0].plot(f, aleat.mean(0), color="#2b6cb0", label="aleatória (média)")
    axs[0].fill_between(f, np.quantile(aleat, .05, 0), np.quantile(aleat, .95, 0), color="#2b6cb0", alpha=0.2)
    axs[0].set_xlabel("Fração de docentes afastados")
    axs[0].set_ylabel("Fração no componente principal")
    axs[0].legend(frameon=False, fontsize=7)
    top = indiv.head(10).iloc[::-1]
    nomes = [nome_curto(n) for n in top["docente"]]
    vals = 100 * top["perda_eficiencia"]
    axs[1].barh(nomes, vals, color=["#c53030" if i > 0 else "#a0aec0" for i in top["vinculos_interarea_perdidos"]])
    for k, (v, o, dc) in enumerate(zip(vals, top["externos_que_perdem_vinculo_com_imc"], top["docentes_desconectados"])):
        axs[1].text(v + vals.max() * 0.02, k, f"{int(o)} ext." + (f", {int(dc)} desc." if dc else ""),
                    va="center", fontsize=6)
    axs[1].set_xlim(0, vals.max() * 1.35)
    axs[1].set_xlabel("Perda de eficiência da rede de docentes (%)")
    axs[1].set_title("Afastamento individual (vermelho: elimina vínculo entre áreas)", fontsize=7.5)
    axs[1].tick_params(axis="y", labelsize=6.5)
    fig.tight_layout()
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_sensibilidade(nmi, rho, caminho):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, M, tit in ((axs[0], nmi, "NMI das comunidades"), (axs[1], rho, "Spearman da força dos docentes")):
        im = ax.imshow(M.values.astype(float), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(M)), M.columns, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(M)), M.index, fontsize=6.5)
        for i in range(len(M)):
            for j in range(len(M)):
                ax.text(j, i, f"{float(M.iat[i, j]):.2f}", ha="center", va="center", fontsize=6,
                        color="white" if float(M.iat[i, j]) < 0.6 else "black")
        ax.set_title(tit, fontsize=8)
    fig.tight_layout()
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def nome_curto(n):
    t = n.split()
    sobren = [x for x in t[1:] if x.upper().strip(".") not in ("JUNIOR", "JÚNIOR", "FILHO", "NETO")]
    return f"{t[0]} {sobren[-1]}" if sobren else t[0]


def _limpo(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def executar_analises(pasta_redes, docentes_csv, pasta_saida, modelo="M2", inicio=2020, fim=2025,
                      n_perm=10000, n_rep_intermediacao=1000, n_rep_modularidade=200, rodar_sensibilidade=True):
    os.makedirs(pasta_saida, exist_ok=True)
    doc = pd.read_csv(docentes_csv, dtype=str, keep_default_na=False)
    G, nos, arestas = carregar_rede(pasta_redes, modelo)
    GD = subrede_docentes(G)
    macro = {"modelo": modelo, "docentes": GD.number_of_nodes(), "externos": G.number_of_nodes() - GD.number_of_nodes(),
             "arestas_total": G.number_of_edges(), "arestas_DD": GD.number_of_edges(),
             "densidade_DD": nx.density(GD), "componentes_DD": nx.number_connected_components(GD),
             "docentes_isolados_DD": [GD.nodes[n]["rotulo"] for n in GD if GD.degree(n) == 0],
             "maior_componente_DD": max(len(c) for c in nx.connected_components(GD)),
             "agrupamento_medio_DD": nx.average_clustering(GD),
             "externos_por_categoria": dict(Counter(nos[nos.tipo_no == "X"].categoria))}
    # fracao das arestas docente-externo que saem da UNIFEI, com limites para os nao identificados
    cat_ext = [G.nodes[v]["categoria"] if G.nodes[v]["tipo_no"] == "X" else G.nodes[u]["categoria"]
               for u, v in G.edges() if (G.nodes[u]["tipo_no"] == "X") != (G.nodes[v]["tipo_no"] == "X")]
    ce = Counter(cat_ext)
    tot = sum(ce.values()) or 1
    fora = ce.get("nacional", 0) + ce.get("estrangeira", 0)
    macro["fracao_vinculos_externos_fora_unifei"] = {
        "limite_inferior": fora / tot, "limite_superior": (fora + ce.get("nao_identificado", 0)) / tot,
        "arestas_docente_externo": tot, "por_categoria": dict(ce)}
    part, r3 = teste_qp3(G, n_rep=n_rep_modularidade)
    met = metricas_docentes(G, doc, inicio, fim, part)
    met, r2 = teste_qp2(G, met, doc, inicio, fim, n_rep=n_rep_intermediacao)
    r1 = teste_qp1(G, n_perm=n_perm)
    r0 = teste_principal(met)
    indiv = afastamento_individual(G)
    grau, aleat = afastamento_sequencial(G)
    met.to_csv(os.path.join(pasta_saida, "metricas_docentes.csv"), index=False, encoding="utf-8-sig")
    indiv.to_csv(os.path.join(pasta_saida, "afastamento_individual.csv"), index=False, encoding="utf-8-sig")
    fig_rede(G, part, os.path.join(pasta_saida, "fig_rede_imc.png"))
    M = fig_matriz(G, os.path.join(pasta_saida, "fig_matriz_cooperacao.png"))
    M.to_csv(os.path.join(pasta_saida, "matriz_cooperacao.csv"), encoding="utf-8-sig")
    fig_nulos(r1, r3, os.path.join(pasta_saida, "fig_modelos_nulos.png"))
    fig_comunidades(G, part, met, os.path.join(pasta_saida, "fig_comunidades_docentes.png"))
    fig_afastamento(grau, aleat, indiv, os.path.join(pasta_saida, "fig_afastamento.png"))
    sens = {}
    if rodar_sensibilidade:
        nmi, rho, varr = sensibilidade(pasta_redes, modelo)
        nmi.to_csv(os.path.join(pasta_saida, "sensibilidade_nmi.csv"), encoding="utf-8-sig")
        rho.to_csv(os.path.join(pasta_saida, "sensibilidade_spearman.csv"), encoding="utf-8-sig")
        varr.to_csv(os.path.join(pasta_saida, "sensibilidade_pesos_m2.csv"), index=False, encoding="utf-8-sig")
        fig_sensibilidade(nmi, rho, os.path.join(pasta_saida, "fig_sensibilidade.png"))
        sens = {"nmi_min_entre_variantes": float(nmi.values.astype(float).min()),
                "spearman_min_entre_variantes": float(rho.values.astype(float).min()),
                "nmi_pesos_m2_mediana": float(varr["nmi_vs_pesos_base"].median()) if len(varr) else None}
    testes = pd.DataFrame([
        {"questao": "QP principal", "hipotese": "estrutura acrescenta informação à contagem de publicações",
         "estatistica": "ρ Spearman (publicações × intermediação de ligação); IC 95% bootstrap",
         "observado": r0["correlacoes_spearman_com_producao"]["intermediacao_ligacao"]["rho"],
         "nulo": "limite superior do IC < 0,7: " + "; ".join(f"{v:.3f}" for v in r0["rho_intermediacao_ic95_bootstrap"]),
         "p_valor": r0["correlacoes_spearman_com_producao"]["intermediacao_ligacao"]["p"], "decisao": r0["decisao"]},
        {"questao": "QP1", "hipotese": "coautorias mais frequentes dentro de cada área",
         "estatistica": "fração de arestas D–D entre áreas", "observado": r1["fracao_inter_obs"],
         "nulo": r1["fracao_inter_nulo_media"], "p_valor": r1["p_valor"], "decisao": r1["decisao"]},
        {"questao": "QP2", "hipotese": "vínculos externos concentrados em poucos docentes",
         "estatistica": f"Gini dos vínculos externos identificados; docentes de ligação (BH): {r2['n_docentes_ligacao']}",
         "observado": r2["gini_vinculos_externos_obs"],
         "nulo": r2["gini_nulo_media"], "p_valor": r2["p_valor_gini"], "decisao": r2["decisao"]},
        {"questao": "QP3", "hipotese": "grupos com coesão acima do esperado",
         "estatistica": "modularidade Louvain", "observado": r3["modularidade_obs"],
         "nulo": r3["modularidade_nulo_media"], "p_valor": r3["p_valor_modularidade"], "decisao": r3["decisao"]},
        {"questao": "QP3", "hipotese": "grupos com recorrência acima do esperado",
         "estatistica": "eventos médios por par intracomunidade", "observado": r3["eventos_medios_intra_obs"],
         "nulo": r3["eventos_medios_intra_nulo"], "p_valor": r3["p_valor_recorrencia"], "decisao": r3["decisao"]},
    ])
    testes.to_csv(os.path.join(pasta_saida, "testes_hipoteses.csv"), index=False, encoding="utf-8-sig")
    import platform, sys, scipy, sklearn, matplotlib as mpl
    ambiente = {"python": sys.version.split()[0], "sistema": platform.platform(), "processador": platform.processor(),
                "cpus": os.cpu_count(), "networkx": nx.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "matplotlib": mpl.__version__}
    resultados = {"ambiente": ambiente, "macro": macro, "qp_principal": r0, "qp1": _limpo(r1), "qp2": r2, "qp3": _limpo(r3),
                  "afastamento_top5": indiv.head(5).to_dict("records"), "sensibilidade": sens}
    with open(os.path.join(pasta_saida, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return resultados, met, testes
