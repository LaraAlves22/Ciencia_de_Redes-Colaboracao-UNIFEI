# ============================================================================
# Coleta OpenAlex - docentes do IMC/UNIFEI
# Etapa A: identificar os IDs de autor de cada docente (ORCID ou nome + UNIFEI)
# Etapa B: baixar as publicacoes do periodo com autorias e afiliacoes
# ============================================================================
import csv
import html
import json
import os
import re
import time
import unicodedata

import requests

API = "https://api.openalex.org"
UNIFEI = "I8165072"          # identificador institucional da UNIFEI no OpenAlex
PARTICULAS = {"DE", "DA", "DO", "DOS", "DAS", "E", "DI", "DEL", "VAN", "VON", "Y"}
SUFIXOS = {"JUNIOR", "JR", "FILHO", "NETO", "SOBRINHO", "SEGUNDO"}


def normalizar(s):
    s = html.unescape(s or "")
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).upper()
    s = re.sub(r"[^A-Z, ]", " ", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    return re.sub(r"\s+", " ", s).strip(" ,")


def _partes(nome):
    n = normalizar(nome)
    if "," in n:
        sob, pre = n.split(",", 1)
        sob_t = [t for t in sob.split() if t not in PARTICULAS]
        pre_t = [t for t in pre.split() if t not in PARTICULAS]
    else:
        toks = [t for t in n.split() if t not in PARTICULAS]
        k = 2 if len(toks) >= 2 and toks[-1] in SUFIXOS else 1
        sob_t, pre_t = toks[-k:], toks[:-k]
    return sob_t, pre_t


def chave_canonica(nome):
    sob_t, pre_t = _partes(nome)
    if not sob_t:
        return ""
    sobrenome = " ".join(sob_t[-2:]) if sob_t[-1] in SUFIXOS and len(sob_t) >= 2 else sob_t[-1]
    return f"{sobrenome} {''.join(t[0] for t in pre_t if t)}".strip()


def compativel(nome_a, nome_b):
    """Mesma chave canonica e prenomes por extenso coincidentes."""
    if chave_canonica(nome_a) != chave_canonica(nome_b):
        return False
    a, b = _partes(nome_a)[1], _partes(nome_b)[1]
    if len(a) > len(b):
        a, b = b, a
    for x, y in zip(a, b):
        if x[0] != y[0]:
            return False
        if len(x) > 1 and len(y) > 1 and x != y:
            return False
    return True


class ClienteOpenAlex:
    def __init__(self, api_key, pasta_bruta, pausa=0.12):
        self.api_key = api_key
        self.pasta = pasta_bruta
        self.pausa = pausa
        self.n_chamadas = 0
        os.makedirs(pasta_bruta, exist_ok=True)

    def get(self, caminho, params, nome_arquivo=None):
        params = dict(params)
        if self.api_key:
            params["api_key"] = self.api_key
        for tentativa in range(5):
            r = requests.get(f"{API}/{caminho}", params=params, timeout=60)
            self.n_chamadas += 1
            if r.status_code == 200:
                dados = r.json()
                if nome_arquivo:
                    with open(os.path.join(self.pasta, nome_arquivo), "w", encoding="utf-8") as f:
                        json.dump(dados, f, ensure_ascii=False)
                time.sleep(self.pausa)
                return dados
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** tentativa)
                continue
            raise RuntimeError(f"OpenAlex {r.status_code}: {r.text[:300]}")
        raise RuntimeError("OpenAlex: muitas tentativas sem sucesso")


def _inst_ids(autor):
    ids = set()
    for a in autor.get("affiliations", []) or []:
        inst = a.get("institution") or {}
        if inst.get("id"):
            ids.add(inst["id"].rsplit("/", 1)[-1])
    for inst in autor.get("last_known_institutions", []) or []:
        if inst.get("id"):
            ids.add(inst["id"].rsplit("/", 1)[-1])
    return ids


def identificar_autores(cliente, docentes, saida_csv):
    """Etapa A. docentes: lista de dicts com nome, id_lattes, orcid, nomes_citacao.
    Gera um CSV de candidatos com a coluna 'aceito' (True/False) para revisao manual."""
    linhas = []
    for d in docentes:
        vistos = set()
        variantes = [d["nome"]] + [v for v in (d.get("nomes_citacao") or "").split(";") if v.strip()]
        # 1) ORCID: identificacao direta
        if d.get("orcid"):
            res = cliente.get("authors", {"filter": f"orcid:{d['orcid']}", "per-page": 25},
                              f"autor_orcid_{d['id_lattes']}.json")
            for a in res.get("results", []):
                aid = a["id"].rsplit("/", 1)[-1]
                vistos.add(aid)
                linhas.append(_linha_candidato(d, a, "orcid", True))
        # 2) nome + afiliacao UNIFEI: capta perfis fragmentados sem ORCID
        res = cliente.get("authors", {"search": d["nome"], "filter": f"affiliations.institution.id:{UNIFEI}",
                                      "per-page": 50}, f"autor_busca_{d['id_lattes']}.json")
        for a in res.get("results", []):
            aid = a["id"].rsplit("/", 1)[-1]
            if aid in vistos:
                continue
            nomes_a = [a.get("display_name", "")] + (a.get("display_name_alternatives") or [])
            ok = any(compativel(na, v) for na in nomes_a for v in variantes)
            # ancora de contexto: nome compativel E afiliacao UNIFEI no perfil
            ancora = UNIFEI in _inst_ids(a)
            linhas.append(_linha_candidato(d, a, "nome+UNIFEI", ok and ancora))
            vistos.add(aid)
    campos = ["docente_id_lattes", "docente_nome", "openalex_author_id", "openalex_nome", "orcid_openalex",
              "works_count", "metodo", "aceito"]
    with open(saida_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    return linhas


def _linha_candidato(d, a, metodo, aceito):
    return {
        "docente_id_lattes": d["id_lattes"], "docente_nome": d["nome"],
        "openalex_author_id": a["id"].rsplit("/", 1)[-1], "openalex_nome": a.get("display_name", ""),
        "orcid_openalex": (a.get("orcid") or "").replace("https://orcid.org/", ""),
        "works_count": a.get("works_count", ""), "metodo": metodo, "aceito": aceito,
    }


def baixar_publicacoes(cliente, candidatos_csv, inicio, fim, pasta_saida):
    """Etapa B. Usa apenas os candidatos com aceito == True."""
    aceitos = {}
    with open(candidatos_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r["aceito"]).strip().lower() in ("true", "1", "sim", "verdadeiro"):
                aceitos.setdefault(r["docente_id_lattes"], []).append(r["openalex_author_id"])
    obras, autorias, mapa = {}, [], []
    for did, aids in aceitos.items():
        for aid in aids:
            mapa.append({"docente_id_lattes": did, "openalex_author_id": aid})
        cursor, pagina = "*", 0
        while cursor:
            res = cliente.get("works", {
                "filter": f"author.id:{'|'.join(aids)},publication_year:{inicio}-{fim}",
                "select": "id,doi,display_name,publication_year,type,authorships",
                "per-page": 200, "cursor": cursor}, f"obras_{did}_{pagina}.json")
            for w in res.get("results", []):
                wid = w["id"].rsplit("/", 1)[-1]
                if wid not in obras:
                    obras[wid] = w
            cursor = res.get("meta", {}).get("next_cursor")
            pagina += 1
            if not res.get("results"):
                break
    linhas_obras = []
    for wid, w in obras.items():
        auts = w.get("authorships") or []
        linhas_obras.append({
            "work_id": wid, "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower(),
            "titulo": w.get("display_name") or "", "ano": w.get("publication_year"),
            "tipo": w.get("type") or "", "n_autores": len(auts),
        })
        for pos, au in enumerate(auts):
            autor = au.get("author") or {}
            insts = au.get("institutions") or []
            linhas_obras_aut = {
                "work_id": wid, "posicao": pos,
                "author_id": (autor.get("id") or "").rsplit("/", 1)[-1],
                "author_nome": autor.get("display_name") or "",
                "orcid": (autor.get("orcid") or "").replace("https://orcid.org/", ""),
                "raw_author_name": au.get("raw_author_name") or "",
                "institution_ids": ";".join(i.get("id", "").rsplit("/", 1)[-1] for i in insts),
                "institution_nomes": ";".join(i.get("display_name", "") or "" for i in insts),
                "paises": ";".join(sorted({i.get("country_code") or "" for i in insts} - {""})),
                "unifei": any(i.get("id", "").endswith(UNIFEI) for i in insts),
                "raw_affiliation": " | ".join(au.get("raw_affiliation_strings") or []),
            }
            autorias.append(linhas_obras_aut)
    os.makedirs(pasta_saida, exist_ok=True)
    _gravar(os.path.join(pasta_saida, "obras_openalex.csv"), linhas_obras)
    _gravar(os.path.join(pasta_saida, "autorias_openalex.csv"), autorias)
    _gravar(os.path.join(pasta_saida, "autores_docentes_openalex.csv"), mapa)
    return linhas_obras, autorias, mapa


def _gravar(caminho, linhas):
    if not linhas:
        open(caminho, "w").close()
        return
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)

