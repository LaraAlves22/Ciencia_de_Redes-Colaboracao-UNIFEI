# ============================================================================
# Integracao Lattes + OpenAlex e construcao das redes de cooperacao (M1, M2, M3)
# Fronteira de um passo: nos = docentes do IMC (D) + colaboradores diretos (X);
# arestas apenas D-D e D-X.
# ============================================================================
import html
import os
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

UNIFEI = "I8165072"
# tipos do OpenAlex aceitos como cooperacao de publicacao (preprints, dissertacoes etc. ficam fora)
TIPOS_OPENALEX = {"article": "artigo_periodico", "review": "artigo_periodico",
                  "conference-paper": "trabalho_completo_evento",
                  "book": "livro_capitulo", "book-chapter": "livro_capitulo"}
PARTICULAS = {"DE", "DA", "DO", "DOS", "DAS", "E", "DI", "DEL", "VAN", "VON", "Y"}
SUFIXOS = {"JUNIOR", "JR", "FILHO", "NETO", "SOBRINHO", "SEGUNDO"}

# Tipos de cooperacao e grupos do modelo multirrelacional (M2)
TIPO_GRUPO = {
    "artigo_periodico": "P", "trabalho_completo_evento": "P", "livro_capitulo": "P",
    "orientacao": "F", "coorientacao": "F", "banca": "F",
    "projeto_pesquisa": "J", "projeto_extensao": "J",
}
TIPOS = list(TIPO_GRUPO)
# numero de tipos em cada grupo (|g|): no M2 cada grupo entra pela media de seus tipos,
# de modo que w_ij fica em (0, 1]
TAMANHO_GRUPO = Counter(TIPO_GRUPO.values())
PESOS_M2 = {"P": 0.5, "F": 0.3, "J": 0.2}
LIMIAR_TRIGRAMA = 0.8  # similaridade minima para casar nomes completos (sensibilidade: 0,7 e 0,9)
LIMITE_AUTORES_MASSIVO = 10
SUBTIPOS_ORIENTACAO = {"orientacao_ic", "orientacao_tcc", "orientacao_mestrado", "orientacao_doutorado",
                       "orientacao_pos_doutorado"}
SUBTIPOS_BANCA = {"banca_tcc", "banca_mestrado", "banca_doutorado", "banca_qualificacao_mestrado",
                  "banca_qualificacao_doutorado"}


# ----------------------------------------------------------------------------
# Normalizacao
# ----------------------------------------------------------------------------
def normalizar(s):
    s = html.unescape("" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s))
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).upper()
    s = re.sub(r"[^A-Z, ]", " ", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    return re.sub(r"\s+", " ", s).strip(" ,")


def norm_titulo(t):
    t = normalizar(t).replace(",", " ")
    return re.sub(r"\s+", " ", t).strip()[:120]


def _partes(nome):
    n = normalizar(nome)
    if "," in n:
        sob, pre = n.split(",", 1)
        return [t for t in sob.split() if t not in PARTICULAS], [t for t in pre.split() if t not in PARTICULAS]
    toks = [t for t in n.split() if t not in PARTICULAS]
    k = 2 if len(toks) >= 2 and toks[-1] in SUFIXOS else 1
    return toks[-k:], toks[:-k]


def chave_canonica(nome):
    sob_t, pre_t = _partes(nome)
    if not sob_t:
        return ""
    sobrenome = " ".join(sob_t[-2:]) if sob_t[-1] in SUFIXOS and len(sob_t) >= 2 else sob_t[-1]
    return f"{sobrenome} {''.join(t[0] for t in pre_t if t)}".strip()


def compativel(a, b):
    if chave_canonica(a) != chave_canonica(b):
        return False
    pa, pb = _partes(a)[1], _partes(b)[1]
    if len(pa) > len(pb):
        pa, pb = pb, pa
    return all(x[0] == y[0] and (len(x) == 1 or len(y) == 1 or x == y) for x, y in zip(pa, pb))


def trigramas(s):
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def similaridade(a, b):
    ta, tb = trigramas(a), trigramas(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def nome_completo(nome):
    n = normalizar(nome)
    return "," not in n and len(n.split()) >= 3


# ----------------------------------------------------------------------------
# Leitura
# ----------------------------------------------------------------------------
def carregar(pasta_lattes, lista_unifei, pasta_openalex=None):
    ler = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    dados = {
        "docentes": ler(os.path.join(pasta_lattes, "docentes.csv")),
        "coop": ler(os.path.join(pasta_lattes, "cooperacoes.csv")),
        "part": ler(os.path.join(pasta_lattes, "cooperacoes_participantes.csv")),
        "unifei": ler(lista_unifei),
    }
    dados["oa_obras"] = dados["oa_aut"] = dados["oa_mapa"] = None
    if pasta_openalex and os.path.exists(os.path.join(pasta_openalex, "obras_openalex.csv")):
        if os.path.getsize(os.path.join(pasta_openalex, "obras_openalex.csv")) > 0:
            dados["oa_obras"] = ler(os.path.join(pasta_openalex, "obras_openalex.csv"))
            dados["oa_aut"] = ler(os.path.join(pasta_openalex, "autorias_openalex.csv"))
            dados["oa_mapa"] = ler(os.path.join(pasta_openalex, "autores_docentes_openalex.csv"))
    return dados


# ----------------------------------------------------------------------------
# Eventos de cooperacao (deduplicados)
# ----------------------------------------------------------------------------
def tipo_de(row):
    st, tp = row["subtipo"], row["tipo"]
    if tp == "publicacao":
        return "livro_capitulo" if st in ("livro", "capitulo_livro") else st
    if st in SUBTIPOS_ORIENTACAO:
        return "coorientacao" if row["papel"] == "coorientador" else "orientacao"
    if st in SUBTIPOS_BANCA:
        return "banca"
    if st in ("projeto_pesquisa", "projeto_extensao"):
        return st
    return None


def montar_eventos(dados, modo_estrito=False):
    """Agrupa os registros dos curriculos em eventos unicos. modo_estrito=True descarta
    os casamentos de participantes feitos apenas por chave canonica."""
    coop = dados["coop"]
    coop = coop[coop["no_periodo"] == "True"].copy()
    coop["tipo_coop"] = coop.apply(tipo_de, axis=1)
    coop = coop[coop["tipo_coop"].notna()]
    part = dados["part"]
    part = part[part["id_registro"].isin(coop["id_registro"])].copy()
    if modo_estrito:
        mask = part["metodo_casamento"].str.startswith("chave") | part["metodo_unifei"].str.startswith("chave")
        part.loc[mask, ["docente_imc_id", "metodo_casamento", "unifei_nome", "unifei_instituto",
                        "unifei_campus", "metodo_unifei"]] = ""
    part_por_reg = {k: g for k, g in part.groupby("id_registro")}
    if modo_estrito:
        docs_reg = {k: set(g.loc[g["docente_imc_id"] != "", "docente_imc_id"]) for k, g in part_por_reg.items()}
        coop["docentes_imc_no_registro"] = [
            ";".join(sorted(docs_reg.get(r, set()) | {d})) for r, d in zip(coop["id_registro"], coop["docente_id_lattes"])]

    def chave_evento(r):
        tc = r["tipo_coop"]
        if tc in ("artigo_periodico", "trabalho_completo_evento", "livro_capitulo"):
            if r["doi"]:
                return "DOI:" + r["doi"].lower().strip()
            return f"T:{norm_titulo(r['titulo'])}|{r['ano_inicio']}"
        if tc in ("orientacao", "coorientacao"):
            return f"O:{normalizar(r['discente'])}|{r['ano_inicio']}|{r['subtipo']}"
        if tc == "banca":
            return f"B:{normalizar(r['discente'])}|{r['ano_inicio']}|{r['subtipo']}"
        return f"J:{norm_titulo(r['titulo'])}|{r['ano_inicio']}"

    coop["chave_evento"] = coop.apply(chave_evento, axis=1)
    # fusao difusa de publicacoes sem DOI: mesmo docente envolvido, ano +-1, titulo >= 0.9
    sem_doi = coop[coop["chave_evento"].str.startswith("T:")]
    alias = {}
    blocos = defaultdict(list)
    for _, r in coop[coop["tipo_coop"].isin(["artigo_periodico", "trabalho_completo_evento", "livro_capitulo"])].iterrows():
        for d in r["docentes_imc_no_registro"].split(";"):
            blocos[d].append((r["chave_evento"], norm_titulo(r["titulo"]), int(r["ano_inicio"] or 0)))
    for d, lst in blocos.items():
        for i in range(len(lst)):
            ki, ti, ai = lst[i]
            if not ki.startswith("T:"):
                continue
            for j in range(len(lst)):
                kj, tj, aj = lst[j]
                if i == j or ki == kj or abs(ai - aj) > 1 or len(ti) < 15:
                    continue
                if similaridade(ti, tj) >= 0.9:
                    alvo = kj if kj.startswith("DOI:") or kj < ki else ki
                    origem = ki if alvo == kj else kj
                    alias[origem] = alias.get(alvo, alvo)
    coop["chave_evento"] = coop["chave_evento"].map(lambda k: alias.get(k, k))

    eventos = []
    for chave, g in coop.groupby("chave_evento"):
        tipos = Counter(g["tipo_coop"])
        tipo = tipos.most_common(1)[0][0]
        if tipo == "coorientacao" and "orientacao" in tipos:
            tipo = "orientacao"
        # registro representativo: maior lista de participantes
        g = g.assign(_n=g["n_participantes"].astype(int)).sort_values("_n", ascending=False)
        rep = g.iloc[0]
        docentes = set()
        papeis = {}
        for _, r in g.iterrows():
            docentes.add(r["docente_id_lattes"])
            papeis[r["docente_id_lattes"]] = r["papel"]
            for d in r["docentes_imc_no_registro"].split(";"):
                if d:
                    docentes.add(d)
        eventos.append({
            "chave_evento": chave, "tipo": tipo, "ano": rep["ano_inicio"], "titulo": rep["titulo"],
            "doi": rep["doi"], "discente": rep.get("discente", ""), "registro_rep": rep["id_registro"],
            "registros": ";".join(g["id_registro"]), "dono_rep": rep["docente_id_lattes"],
            "docentes": docentes, "papeis": papeis, "n_autores": int(rep["n_participantes"] or 0),
            "fonte": "lattes", "texto": rep["texto"],
        })
    return eventos, part_por_reg


# ----------------------------------------------------------------------------
# Entidades e participantes
# ----------------------------------------------------------------------------
def indice_unifei(unifei, docentes=None):
    """Indice da lista do SIGAA. Docentes permanentes do IMC recebem 'id_docente' (id Lattes
    ou SEM_LATTES_*), para que nunca sejam tratados como colaboradores externos."""
    pessoas = unifei.to_dict("records")
    nomes_doc = {} if docentes is None else {normalizar(r["nome"]): r["id_lattes"] for r in docentes.to_dict("records")}
    for p in pessoas:
        p["nome_norm"] = normalizar(p["nome"])
        p["id_docente"] = ""
        if p["instituto"] == "IMC" and p["vinculo"] == "Permanente":
            melhor = max(nomes_doc, key=lambda n: similaridade(n, p["nome_norm"]), default=None)
            if melhor and similaridade(melhor, p["nome_norm"]) >= LIMIAR_TRIGRAMA:
                p["id_docente"] = nomes_doc[melhor]
            else:
                p["id_docente"] = "SEM_LATTES_" + p["nome_norm"].replace(" ", "_")
    return {p["nome_norm"]: p for p in pessoas}


def entidade_unifei(p, nome=None, metodo=""):
    if p.get("id_docente"):
        return {"id": p["id_docente"], "tipo_no": "D"}
    cat = "outro_vinculo_unifei" if p["instituto"] == "IMC" else "docente_outro_instituto"
    return {"id": "U:" + p["nome_norm"], "tipo_no": "X", "nome": nome or p["nome"], "categoria": cat,
            "instituto": p["instituto"], "campus": p["campus"], "metodo": metodo}


def resolver_participantes(eventos, part_por_reg, dados):
    """Atribui a cada participante de cada evento um identificador de entidade.
    Docentes: id Lattes. Externos (criterio conservador):
      OA:<id>    autor OpenAlex (quando o evento casou com uma obra do OpenAlex)
      U:<nome>   docente da UNIFEI identificado na lista do SIGAA
      N:<nome>   nome completo (sem virgula, 3+ palavras)
      K:<chave>|<docente>  nome em formato de citacao: so se funde dentro do mesmo docente
    """
    unifei = indice_unifei(dados["unifei"], dados["docentes"])
    oa = _indice_openalex(dados)
    entidades = {}
    for ev in eventos:
        ev["participantes"] = []
        ev["openalex_work"] = ""
        regs = part_por_reg.get(ev["registro_rep"])
        oa_aut = None
        if oa and ev["tipo"] in ("artigo_periodico", "trabalho_completo_evento", "livro_capitulo"):
            wid = oa["por_doi"].get((ev["doi"] or "").lower()) or oa["por_titulo"].get(
                (norm_titulo(ev["titulo"]), str(ev["ano"])))
            if wid:
                ev["openalex_work"] = wid
                oa_aut = oa["autorias"].get(wid, [])
                oa["usadas"].add(wid)
        vistos = set()
        if regs is not None:
            for _, p in regs.iterrows():
                ent = _entidade_participante(p, ev, unifei, oa_aut, oa)
                if ent is None or ent["id"] in vistos:
                    continue
                vistos.add(ent["id"])
                ev["participantes"].append(ent["id"])
                _registrar(entidades, ent)
        for d in ev["docentes"]:
            if d not in vistos:
                vistos.add(d)
                ev["participantes"].append(d)
                _registrar(entidades, {"id": d, "tipo_no": "D"})
        if ev["tipo"] in ("orientacao", "coorientacao") and ev["discente"]:
            ent = _entidade_discente(ev, unifei)
            if ent["id"] not in vistos:
                ev["participantes"].append(ent["id"])
                _registrar(entidades, ent)
    # obras do OpenAlex que nao estao em nenhum Lattes
    if oa:
        for wid, auts in oa["autorias"].items():
            if wid in oa["usadas"]:
                continue
            ob = oa["obras"][wid]
            docs = {oa["autor_doc"][a["author_id"]] for a in auts if a["author_id"] in oa["autor_doc"]}
            if not docs or ob["tipo"] not in TIPOS_OPENALEX:
                continue
            tipo = TIPOS_OPENALEX[ob["tipo"]]
            ev = {"chave_evento": "OA:" + wid, "tipo": tipo, "ano": ob["ano"], "titulo": ob["titulo"],
                  "doi": ob["doi"], "discente": "", "registro_rep": "", "registros": "", "dono_rep": "",
                  "docentes": docs, "papeis": {}, "n_autores": int(ob["n_autores"] or len(auts)),
                  "fonte": "openalex", "texto": "", "openalex_work": wid, "participantes": []}
            for a in auts:
                if a["author_id"] in oa["autor_doc"]:
                    eid = oa["autor_doc"][a["author_id"]]
                    _registrar(entidades, {"id": eid, "tipo_no": "D"})
                else:
                    ent = _entidade_openalex(a, unifei)
                    eid = ent["id"]
                    _registrar(entidades, ent)
                if eid not in ev["participantes"]:
                    ev["participantes"].append(eid)
            eventos.append(ev)
    return eventos, entidades


def _registrar(entidades, ent):
    e = entidades.setdefault(ent["id"], {"id": ent["id"], "tipo_no": ent["tipo_no"], "nomes": Counter(),
                                         "categoria": "", "instituto": "", "campus": "", "paises": Counter(),
                                         "metodos": Counter()})
    if ent.get("nome"):
        e["nomes"][ent["nome"]] += 1
    for k in ("categoria", "instituto", "campus"):
        if ent.get(k) and not e[k]:
            e[k] = ent[k]
    if ent.get("pais"):
        e["paises"][ent["pais"]] += 1
    if ent.get("metodo"):
        e["metodos"][ent["metodo"]] += 1


def _entidade_participante(p, ev, unifei, oa_aut, oa):
    if p["docente_imc_id"]:
        return {"id": p["docente_imc_id"], "tipo_no": "D"}
    nome = p["nome_raw"]
    if ev["tipo"] in ("orientacao", "coorientacao") and normalizar(nome) == normalizar(ev["discente"]):
        return _entidade_discente(ev, unifei)
    # alinhamento com a autoria do OpenAlex do mesmo trabalho
    if oa_aut:
        for a in oa_aut:
            if a["author_id"] in oa["autor_doc"]:
                continue
            if compativel(nome, a["author_nome"]) or compativel(nome, a["raw_author_name"]):
                ent = _entidade_openalex(a, unifei)
                if ent["tipo_no"] == "D":
                    return ent
                pu = unifei.get(normalizar(p["unifei_nome"])) if p["unifei_instituto"] else None
                if pu and ent["categoria"] != "docente_outro_instituto":
                    e2 = entidade_unifei(pu)
                    if e2["tipo_no"] == "D":
                        return e2
                    ent.update(categoria=e2["categoria"], instituto=e2["instituto"], campus=e2["campus"])
                ent["nome"] = nome
                return ent
    if p["unifei_instituto"]:
        pu = unifei.get(normalizar(p["unifei_nome"]))
        if pu:
            return entidade_unifei(pu, nome, p["metodo_unifei"])
    if nome_completo(nome):
        return {"id": "N:" + normalizar(nome), "tipo_no": "X", "nome": nome, "categoria": "",
                "metodo": "nome_completo"}
    k = chave_canonica(nome)
    if not k:
        return None
    return {"id": f"K:{k}|{ev['dono_rep']}", "tipo_no": "X", "nome": nome, "categoria": "",
            "metodo": "chave_por_docente"}


ESTRANGEIRA_RE = re.compile(r"\b(UNIVERSITY|UNIVERSIDAD|UNIVERSITE|UNIVERSITAT|UNIVERSITA|INSTITUTE OF|COLLEGE)\b")


def instituicao_orientacao(texto):
    """Instituicao do curso do orientando, lida do registro de orientacao do Lattes:
    '... (Graduacao em X) - Universidade Federal de Minas Gerais, ...'."""
    m = re.search(r"\)\s*-\s*([^.]+?)(?:,\s*[A-Z][^,]*?)?(?:\.\s*(?:Orientador|Coorientador)|\.\s*$|$)", texto or "")
    return normalizar(m.group(1)).strip(" ,") if m else ""


def _entidade_discente(ev, unifei):
    """Orientando: docente/servidor da lista do SIGAA; senao, a instituicao do curso informada no
    proprio registro de orientacao (UNIFEI -> outro vinculo; outra instituicao -> nacional ou estrangeira)."""
    nome = ev["discente"]
    n = normalizar(nome)
    if n in unifei:
        return entidade_unifei(unifei[n], nome, "discente_lista_sigaa")
    texto = normalizar(ev.get("texto", ""))
    if "UNIVERSIDADE FEDERAL DE ITAJUBA" in texto or "UNIFEI" in texto:
        cat, metodo = "outro_vinculo_unifei", "discente"
    else:
        inst = instituicao_orientacao(ev.get("texto", ""))
        if inst:
            cat = "estrangeira" if ESTRANGEIRA_RE.search(inst) else "nacional"
            metodo = "discente_instituicao_registro"
        else:
            cat, metodo = "", "discente"
    return {"id": "N:" + n, "tipo_no": "X", "nome": nome, "categoria": cat, "metodo": metodo}


def _entidade_openalex(a, unifei):
    nome = a["author_nome"] or a["raw_author_name"]
    n = normalizar(nome)
    ent = {"id": "OA:" + a["author_id"], "tipo_no": "X", "nome": nome, "metodo": "openalex"}
    if n in unifei:
        e2 = entidade_unifei(unifei[n], nome, "openalex+lista_sigaa")
        if e2["tipo_no"] == "D":
            return e2
        ent.update(categoria=e2["categoria"], instituto=e2["instituto"], campus=e2["campus"])
    elif a["unifei"] == "True":
        ent["categoria"] = "outro_vinculo_unifei"
    elif a["paises"]:
        ent["categoria"] = "nacional" if "BR" in a["paises"].split(";") else "estrangeira"
        ent["pais"] = a["paises"].split(";")[0]
    return ent


def _indice_openalex(dados):
    if dados["oa_obras"] is None:
        return None
    obras = {r["work_id"]: r for r in dados["oa_obras"].to_dict("records")}
    autorias = defaultdict(list)
    for r in dados["oa_aut"].to_dict("records"):
        autorias[r["work_id"]].append(r)
    autor_doc = {r["openalex_author_id"]: r["docente_id_lattes"] for r in dados["oa_mapa"].to_dict("records")}
    por_doi = {r["doi"]: w for w, r in obras.items() if r["doi"]}
    por_titulo = {(norm_titulo(r["titulo"]), str(r["ano"])): w for w, r in obras.items()}
    return {"obras": obras, "autorias": autorias, "autor_doc": autor_doc, "por_doi": por_doi,
            "por_titulo": por_titulo, "usadas": set()}


# ----------------------------------------------------------------------------
# Contagens por par e tipo
# ----------------------------------------------------------------------------
def contar_pares(eventos, entidades):
    """n_t(i,j): fronteira de um passo -> so pares com ao menos um docente.
    Coautoria: contagem fracionaria 1/(a-1)."""
    eh_d = lambda e: entidades[e]["tipo_no"] == "D"
    n = defaultdict(lambda: defaultdict(float))
    n_massivo = defaultdict(lambda: defaultdict(float))
    n_ev = defaultdict(int)
    for ev in eventos:
        t = ev["tipo"]
        ps = ev["participantes"]
        if len(ps) < 2:
            continue
        if t in ("artigo_periodico", "trabalho_completo_evento", "livro_capitulo"):
            a = max(ev["n_autores"], len(ps))
            peso = 1.0 / (a - 1)
            massivo = a > LIMITE_AUTORES_MASSIVO
        else:
            peso, massivo = 1.0, False
        ev["massivo"] = massivo
        pares = set()
        if t in ("orientacao", "coorientacao"):
            docs = [p for p in ps if eh_d(p)]
            outros = [p for p in ps if not eh_d(p)]
            for d in docs:
                for x in outros:
                    tt = "coorientacao" if ev["papeis"].get(d) == "coorientador" else "orientacao"
                    pares.add((d, x, tt))
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    pares.add((docs[i], docs[j], "coorientacao"))
        else:
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    if eh_d(ps[i]) or eh_d(ps[j]):
                        pares.add((ps[i], ps[j], t))
        for k in {tuple(sorted((i, j))) for i, j, _ in pares}:
            n_ev[k] += 1
        for i, j, tt in pares:
            k = tuple(sorted((i, j)))
            n[k][tt] += peso
            if massivo:
                n_massivo[k][tt] += peso
    return n, n_massivo, n_ev


# ----------------------------------------------------------------------------
# Modelos de intensidade
# ----------------------------------------------------------------------------
def modelos(n, pesos=None, q=5, n_ev=None):
    pesos = pesos or PESOS_M2
    tot = defaultdict(lambda: defaultdict(float))
    for (i, j), d in n.items():
        for t, v in d.items():
            tot[i][t] += v
            tot[j][t] += v
    # M3: classes por quantis entre os pares com n_t > 0
    cortes = {}
    for t in TIPOS:
        vals = np.array([d[t] for d in n.values() if d.get(t, 0) > 0])
        cortes[t] = np.quantile(vals, np.linspace(0, 1, q + 1)[1:-1]) if len(vals) else np.array([])
    linhas = []
    for (i, j), d in n.items():
        m1 = sum(d.values())
        m2 = 0.0
        for t, v in d.items():
            # cada razao 2 n_t(i,j) / (n_t(i) + n_t(j)) esta em [0, 1]; o grupo entra pela media de seus tipos
            g = TIPO_GRUPO[t]
            m2 += pesos[g] / TAMANHO_GRUPO[g] * (2 * v / (tot[i][t] + tot[j][t]))
        m3 = 0.0
        for t, v in d.items():
            classe = 1 + int(np.searchsorted(cortes[t], v, side="right")) if len(cortes[t]) else q
            m3 += min(classe, q) / q
        m3 /= len(TIPOS)
        linha = {"source": i, "target": j, "M1": m1, "M2": m2, "M3": m3,
                 "n_eventos": (n_ev or {}).get((i, j), 0)}
        for t in TIPOS:
            linha[f"n_{t}"] = d.get(t, 0.0)
        linhas.append(linha)
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------
# Tabela de nos
# ----------------------------------------------------------------------------
def tabela_nos(entidades, dados, camadas_extra=None):
    doc = dados["docentes"].set_index("id_lattes")
    linhas, mapa_privado = [], []
    seq = 0
    for eid, e in sorted(entidades.items(), key=lambda kv: (kv[1]["tipo_no"], kv[0])):
        if e["tipo_no"] == "D":
            if eid in doc.index:
                nome, camada = doc.loc[eid, "nome"], doc.loc[eid, "camada_sugerida"]
            else:
                nome = eid.replace("SEM_LATTES_", "").replace("_", " ").title()
                camada = (camadas_extra or {}).get(eid, "Matemática")
            linhas.append({"id": eid, "rotulo": nome, "tipo_no": "D", "camada": camada,
                           "categoria": "docente_imc", "instituto": "IMC", "campus": "Itajubá",
                           "metodo_identificacao": "lattes" if eid in doc.index else "sem_lattes"})
        else:
            seq += 1
            pseudo = f"X{seq:05d}"
            cat = e["categoria"] or "nao_identificado"
            linhas.append({"id": eid, "rotulo": pseudo, "tipo_no": "X", "camada": "Externo", "categoria": cat,
                           "instituto": e["instituto"], "campus": e["campus"],
                           "metodo_identificacao": ";".join(e["metodos"]) if e["metodos"] else ""})
            mapa_privado.append({"pseudonimo": pseudo, "id": eid,
                                 "nome_mais_frequente": e["nomes"].most_common(1)[0][0] if e["nomes"] else ""})
    return pd.DataFrame(linhas), pd.DataFrame(mapa_privado)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def executar_integracao(pasta_lattes, lista_unifei, pasta_saida, pasta_openalex=None, pesos=None):
    os.makedirs(pasta_saida, exist_ok=True)
    dados = carregar(pasta_lattes, lista_unifei, pasta_openalex)
    resumo = {}
    for estrito in (False, True):
        sufixo = "_estrito" if estrito else ""
        eventos, part_por_reg = montar_eventos(dados, modo_estrito=estrito)
        eventos, entidades = resolver_participantes(eventos, part_por_reg, dados)
        n, n_massivo, n_ev = contar_pares(eventos, entidades)
        arestas = modelos(n, pesos, n_ev=n_ev)
        nos, mapa = tabela_nos(entidades, dados)
        presentes = set(arestas["source"]) | set(arestas["target"])
        nos["isolado"] = ~nos["id"].isin(presentes)
        nos.to_csv(os.path.join(pasta_saida, f"nos{sufixo}.csv"), index=False, encoding="utf-8-sig")
        arestas.to_csv(os.path.join(pasta_saida, f"arestas{sufixo}.csv"), index=False, encoding="utf-8-sig")
        if not estrito:
            mapa.to_csv(os.path.join(pasta_saida, "mapa_externos_PRIVADO.csv"), index=False, encoding="utf-8-sig")
            # sem trabalhos de autoria massiva (verificacao)
            n_sem = {k: {t: v - n_massivo.get(k, {}).get(t, 0) for t, v in d.items()} for k, d in n.items()}
            n_sem = {k: {t: v for t, v in d.items() if v > 1e-12} for k, d in n_sem.items()}
            n_sem = {k: d for k, d in n_sem.items() if d}
            modelos(n_sem, pesos, n_ev=n_ev).to_csv(os.path.join(pasta_saida, "arestas_sem_massivos.csv"),
                                         index=False, encoding="utf-8-sig")
            ev_df = pd.DataFrame([{k: (";".join(sorted(v)) if isinstance(v, set) else v)
                                   for k, v in ev.items() if k not in ("papeis", "texto")} | {
                "participantes": ";".join(ev["participantes"])} for ev in eventos])
            ev_df.to_csv(os.path.join(pasta_saida, "eventos.csv"), index=False, encoding="utf-8-sig")
        resumo[sufixo or "_padrao"] = {
            "eventos": len(eventos), "docentes": int((nos.tipo_no == "D").sum()),
            "externos": int((nos.tipo_no == "X").sum()), "arestas": len(arestas),
            "arestas_DD": int(((arestas.source.isin(nos.id[nos.tipo_no == "D"])) &
                               (arestas.target.isin(nos.id[nos.tipo_no == "D"]))).sum()),
            "externos_por_categoria": nos[nos.tipo_no == "X"].categoria.value_counts().to_dict(),
            "eventos_por_tipo": Counter(ev["tipo"] for ev in eventos),
            "eventos_massivos": sum(1 for ev in eventos if ev.get("massivo")),
            "eventos_com_openalex": sum(1 for ev in eventos if ev.get("openalex_work")),
        }
        if dados["oa_obras"] is not None:
            # conferencia das obras do OpenAlex: total baixado, fora do escopo, casadas com eventos
            # do Lattes (obras distintas e eventos) e acrescentadas por nao constarem em nenhum Lattes
            obras = dados["oa_obras"]
            casadas = {ev["openalex_work"] for ev in eventos if ev.get("openalex_work") and ev["fonte"] == "lattes"}
            resumo[sufixo or "_padrao"]["conferencia_openalex"] = {
                "obras_baixadas": int(len(obras)),
                "obras_fora_do_escopo": int((~obras["tipo"].isin(list(TIPOS_OPENALEX))).sum()),
                "eventos_lattes_casados": sum(1 for ev in eventos if ev.get("openalex_work") and ev["fonte"] == "lattes"),
                "obras_casadas_distintas": len(casadas),
                "obras_acrescentadas": sum(1 for ev in eventos if ev["fonte"] == "openalex"),
            }
    return resumo
