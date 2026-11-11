import csv
import glob
import html
import os
import re
import unicodedata
from collections import Counter, defaultdict

from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------
# Normalizacao de nomes
# ----------------------------------------------------------------------------
PARTICULAS = {"DE", "DA", "DO", "DOS", "DAS", "E", "DI", "DEL", "VAN", "VON", "Y"}
SUFIXOS = {"JUNIOR", "JR", "FILHO", "NETO", "SOBRINHO", "SEGUNDO"}


def sem_acentos(s):
    s = html.unescape(s or "")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalizar(s):
    """Maiusculas, sem acentos/entidades, so letras, virgula e espaco."""
    s = sem_acentos(s).upper()
    s = re.sub(r"[^A-Z, ]", " ", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    return re.sub(r"\s+", " ", s).strip(" ,")


def chave_canonica(nome):
    """Sobrenome final (com sufixo, se houver) + iniciais dos prenomes, sem particulas.
    Aceita 'SOUZA, A. D.' e 'Adler Diniz de Souza'."""
    n = normalizar(nome)
    if not n:
        return ""
    if "," in n:
        sob, pre = n.split(",", 1)
        sob_t = [t for t in sob.split() if t not in PARTICULAS]
        pre_t = [t for t in pre.split() if t not in PARTICULAS]
    else:
        toks = [t for t in n.split() if t not in PARTICULAS]
        if not toks:
            return ""
        k = 2 if len(toks) >= 2 and toks[-1] in SUFIXOS else 1
        sob_t, pre_t = toks[-k:], toks[:-k]
    if not sob_t:
        return ""
    if sob_t[-1] in SUFIXOS and len(sob_t) >= 2:
        sobrenome = " ".join(sob_t[-2:])
    else:
        sobrenome = sob_t[-1]
    iniciais = "".join(t[0] for t in pre_t if t)
    return f"{sobrenome} {iniciais}".strip()


def trigramas(s):
    s = f"  {s} "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def similaridade(a, b):
    ta, tb = trigramas(a), trigramas(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def prenomes(nome):
    n = normalizar(nome)
    if "," in n:
        sob, pre = n.split(",", 1)
        return [t for t in pre.split() if t not in PARTICULAS]
    toks = [t for t in n.split() if t not in PARTICULAS]
    k = 2 if len(toks) >= 2 and toks[-1] in SUFIXOS else 1
    return toks[:-k]


def compativel(nome_citado, nome_completo):
    """Mesma chave canonica nao basta: um prenome escrito por extenso precisa coincidir."""
    a, b = prenomes(nome_citado), prenomes(nome_completo)
    if len(a) > len(b):
        return False
    for x, y in zip(a, b):
        if len(x) > 1 and x != y:
            return False
        if x[0] != y[0]:
            return False
    return True


def limpar(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


# ----------------------------------------------------------------------------
# Classificacao das secoes
# ----------------------------------------------------------------------------
def tipo_producao(rotulo):
    r = normalizar(rotulo)
    if r.startswith("ARTIGOS COMPLETOS"):
        return "publicacao", "artigo_periodico"
    if r.startswith("TRABALHOS COMPLETOS PUBLICADOS EM ANAIS"):
        return "publicacao", "trabalho_completo_evento"
    if r.startswith("LIVROS PUBLICADOS"):
        return "publicacao", "livro"
    if r.startswith("CAPITULOS DE LIVROS"):
        return "publicacao", "capitulo_livro"
    return None, None


def subtipo_formacao(rotulo):
    r = normalizar(rotulo)
    if "INICIACAO" in r:
        return "ic"
    if "CONCLUSAO DE CURSO DE GRADUACAO" in r:
        return "tcc"
    if "QUALIFICAC" in r and "DOUTORADO" in r:
        return "qualificacao_doutorado"
    if "QUALIFICAC" in r and "MESTRADO" in r:
        return "qualificacao_mestrado"
    if "MESTRADO" in r:
        return "mestrado"
    if "DOUTORADO" in r:
        return "doutorado"
    if "POS DOUTORADO" in r or "POS-DOUTORADO" in r:
        return "pos_doutorado"
    if "ESPECIALIZACAO" in r or "APERFEICOAMENTO" in r:
        return "especializacao"
    if "CONCURSO" in r:
        return "concurso_publico"
    return "outra"


ANO_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def primeiro_ano(texto, padroes):
    for p in padroes:
        m = re.search(p, texto)
        if m:
            return int(m.group(1))
    anos = [int(a) for a in ANO_RE.findall(texto) if 1950 <= int(a) <= 2030]
    return anos[-1] if anos else None


# ----------------------------------------------------------------------------
# Leitura de um curriculo
# ----------------------------------------------------------------------------
def ler_html(caminho):
    bruto = open(caminho, "rb").read()
    for enc in ("utf-8", "latin-1"):
        try:
            txt = bruto.decode(enc)
            if enc == "utf-8" and "Ã" in txt[:20000]:
                continue
            return txt
        except UnicodeDecodeError:
            continue
    return bruto.decode("latin-1", errors="replace")


def campo_identificacao(soup, rotulo):
    for b in soup.find_all("b"):
        if limpar(b.get_text()).startswith(rotulo):
            cel = b.find_parent("div", class_="layout-cell-3")
            if cel:
                prox = cel.find_next_sibling("div", class_="layout-cell-9")
                if prox:
                    return limpar(prox.get_text(" "))
    return ""


def extrair_docente(soup, texto):
    d = {}
    d["nome"] = campo_identificacao(soup, "Nome") or ""
    d["nomes_citacao"] = campo_identificacao(soup, "Nome em citações bibliográficas")
    m = re.search(r"lattes\.cnpq\.br/(\d{16})", texto)
    d["id_lattes"] = m.group(1) if m else ""
    m = re.search(r"orcid\.org/(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", texto)
    d["orcid"] = m.group(1) if m else ""
    m = re.search(r"atualização do currículo em (\d{2}/\d{2}/\d{4})", texto)
    d["data_atualizacao_lattes"] = m.group(1) if m else ""

    # Formacao: doutorado mais recente (primeiro listado)
    d["doutorado"], d["doutorado_instituicao"], d["doutorado_ano"] = "", "", ""
    ancora = soup.find("a", attrs={"name": "FormacaoAcademicaTitulacao"})
    if ancora:
        bloco = ancora.find_parent("div", class_="title-wrapper") or ancora.parent
        t = bloco.get_text("|")
        t = re.sub(r"\s+", " ", t)
        m = re.search(r"Doutorado( em andamento)? em ([^|]+?)\s*(?:\|\s*)?\.\s*\|\s*([^|]+?)\.\s*\|", t)
        if m:
            curso = re.sub(r"^(Doutorado em\s+)+", "", limpar(m.group(2)))
            d["doutorado"] = curso + (" (em andamento)" if m.group(1) else "")
            d["doutorado_instituicao"] = limpar(m.group(3))
            m2 = re.search(r"Ano de obtenção: (\d{4})", t[m.end():m.end() + 3000])
            d["doutorado_ano"] = m2.group(1) if (m2 and not m.group(1)) else ""

    # Atuacao profissional: inicio do vinculo com a UNIFEI
    d["ano_ingresso_unifei"] = ""
    ancora = soup.find("a", attrs={"name": "AtuacaoProfissional"})
    if ancora:
        bloco = ancora.find_parent("div", class_="title-wrapper") or ancora.parent
        insts = bloco.find_all("div", class_="inst_back")
        for inst in insts:
            if "ITAJUB" in normalizar(inst.get_text()):
                partes = []
                for el in inst.find_all_next():
                    if el in insts and el is not inst:
                        break
                    if el.name == "div" and "inst_back" in (el.get("class") or []) and el is not inst:
                        break
                    if el.name == "div" and "layout-cell-pad-5" in (el.get("class") or []):
                        partes.append(limpar(el.get_text(" ")))
                    if len(partes) > 400:
                        break
                inicios = []
                for i, p in enumerate(partes):
                    m = re.match(r"^(\d{4}) - (Atual|\d{4})$", p)
                    if m and i + 1 < len(partes) and "Vínculo:" in partes[i + 1]:
                        vinc = partes[i + 1]
                        if "Servidor" in vinc or "Professor" in vinc:
                            inicios.append(int(m.group(1)))
                if inicios:
                    d["ano_ingresso_unifei"] = str(min(inicios))
                break
    return d


def participantes_de_lista(trecho):
    nomes = [limpar(x) for x in trecho.split(";")]
    return [n for n in nomes if n and len(n) > 1]


def extrair_registros(soup):
    """Percorre o documento em ordem e devolve registros brutos."""
    regs = []
    h1 = inst = cita = ""
    alvo = lambda t: (
        t.name == "h1"
        or (t.name == "div" and ({"inst_back", "cita-artigos"} & set(t.get("class") or [])))
        or (t.name == "div" and "layout-cell-11" in (t.get("class") or []))
        or (t.name == "a" and (t.get("name") or "").startswith("PP_"))
    )
    for el in soup.find_all(alvo):
        cls = set(el.get("class") or [])
        if el.name == "h1":
            h1 = limpar(el.get_text())
            inst = cita = ""
            continue
        if "inst_back" in cls:
            inst = limpar(el.get_text())
            cita = ""
            continue
        if "cita-artigos" in cls:
            cita = limpar(el.get_text())
            continue
        if el.name == "a":
            if not normalizar(h1).startswith(("PROJETOS", "OUTROS PROJETOS")):
                continue
            regs.append(registro_projeto(el, h1))
            continue
        # item numerado (layout-cell-11)
        H = normalizar(h1)
        if H == "PRODUCOES":
            tipo, sub = tipo_producao(cita)
            if tipo:
                regs.append(registro_producao(el, sub))
        elif H == "BANCAS" and "TRABALHOS DE CONCLUSAO" in normalizar(inst):
            regs.append(registro_banca(el, subtipo_formacao(cita)))
        elif H == "ORIENTACOES":
            situ = "concluida" if "CONCLUIDA" in normalizar(inst) else "em_andamento"
            regs.append(registro_orientacao(el, subtipo_formacao(cita), situ))
    return [r for r in regs if r]


def links_lattes(el):
    out = {}
    for a in el.find_all("a", href=True):
        m = re.search(r"lattes\.cnpq\.br/(\d{16})", a["href"])
        if m:
            out[normalizar(a.get_text())] = m.group(1)
    return out


def doi_de(el):
    a = el.find("a", class_="icone-doi")
    if a and a.get("href"):
        return re.sub(r"^https?://(dx\.)?doi\.org/", "", a["href"]).strip()
    return ""


def registro_producao(el, subtipo):
    span_ano = el.find("span", attrs={"data-tipo-ordenacao": "ano"})
    # remove spans auxiliares de ordenacao/citacao antes de pegar o texto
    for sp in el.find_all("span", class_=["informacao-artigo", "citado"]):
        sp.extract()
    for sup in el.find_all("sup"):
        sup.extract()
    texto = limpar(el.get_text(" "))
    texto = re.sub(r"\s+([;,.])", r"\1", texto).replace(";", "; ")
    texto = limpar(texto)
    # autores terminam no primeiro ' . ' do texto original (Lattes usa 'AUTOR .')
    bruto = limpar(el.get_text(" "))
    m = re.search(r"\s\.\s", bruto)
    autores_txt = bruto[: m.start()] if m else bruto.split(". ")[0]
    resto = bruto[m.end():] if m else ""
    titulo = resto.split(". ")[0] if resto else ""
    if span_ano and limpar(span_ano.get_text()).isdigit():
        ano = int(limpar(span_ano.get_text()))
    else:
        ano = primeiro_ano(resto, [r"In: [^,]*?, ((?:19|20)\d{2}),", r", ((?:19|20)\d{2})\.\s"])
    return {
        "tipo": "publicacao", "subtipo": subtipo, "ano_inicio": ano, "ano_fim": ano,
        "titulo": limpar(titulo), "doi": doi_de(el), "texto": texto, "papel": "autor",
        "participantes": participantes_de_lista(autores_txt), "links": links_lattes(el),
        "situacao": "",
    }


def registro_banca(el, subtipo):
    bruto = limpar(el.get_text(" "))
    m = re.search(r"\.?\s*Participação em banca de\s+", bruto)
    if not m:
        return None
    autores_txt = bruto[: m.start()]
    resto = bruto[m.end():]
    candidato = resto.split(". ")[0]
    ano = primeiro_ano(resto, [r"\.\s((?:19|20)\d{2})\.\s"])
    titulo = resto.split(". ")[1] if len(resto.split(". ")) > 1 else ""
    parts = participantes_de_lista(autores_txt)
    return {
        "tipo": "formacao", "subtipo": f"banca_{subtipo}", "ano_inicio": ano, "ano_fim": ano,
        "titulo": limpar(titulo), "doi": "", "texto": bruto, "papel": "membro_banca",
        "participantes": parts, "links": links_lattes(el), "situacao": "",
        "discente": limpar(candidato),
    }


def registro_orientacao(el, subtipo, situacao):
    bruto = limpar(el.get_text(" "))
    pedacos = bruto.split(". ")
    discente = pedacos[0] if pedacos else ""
    titulo = pedacos[1] if len(pedacos) > 1 else ""
    ano = primeiro_ano(bruto, [r"\.\s((?:19|20)\d{2})\.\s", r"Início:\s*((?:19|20)\d{2})"])
    papel = "coorientador" if re.search(r"Co-?orientador", bruto, re.I) else "orientador"
    return {
        "tipo": "formacao", "subtipo": f"orientacao_{subtipo}", "ano_inicio": ano, "ano_fim": ano,
        "titulo": limpar(titulo), "doi": "", "texto": bruto, "papel": papel,
        "participantes": [discente] if discente else [], "links": links_lattes(el),
        "situacao": situacao, "discente": discente,
    }


def registro_projeto(ancora, h1):
    sub = {
        "PROJETOS DE PESQUISA": "projeto_pesquisa",
        "PROJETOS DE EXTENSAO": "projeto_extensao",
        "PROJETOS DE DESENVOLVIMENTO": "projeto_desenvolvimento",
        "PROJETOS DE ENSINO": "projeto_ensino",
    }.get(normalizar(h1), "projeto_outro")
    celulas = []
    for sib in ancora.find_next_siblings():
        if sib.name == "a" and (sib.get("name") or "").startswith("PP_"):
            break
        if sib.name == "div":
            celulas.append(sib)
    textos = [limpar(c.get_text(" ")) for c in celulas]
    periodo = textos[0] if textos else ""
    titulo = textos[1] if len(textos) > 1 else limpar(ancora.get("name", "")[3:])
    corpo = " ".join(textos[2:])
    m = re.match(r"(\d{4})\s*-\s*(Atual|\d{4})?", periodo)
    ini = int(m.group(1)) if m else None
    fim = (9999 if m and m.group(2) == "Atual" else int(m.group(2)) if m and m.group(2) else ini) if m else None
    integrantes = []
    mi = re.search(r"Integrantes:\s*(.+?)(?:\.\s*(?:Financiador|Número|$)|$)", corpo)
    if mi:
        for item in mi.group(1).split(" / "):
            nome = re.sub(r"\s+-\s+(Coordenador|Integrante|Integrante\.?)\.?$", "", limpar(item))
            nome = re.sub(r"\s+-\s+[A-Za-zÀ-ú ]+\.?$", "", nome)
            if nome:
                integrantes.append(limpar(nome.rstrip(".")))
    situ = ""
    ms = re.search(r"Situação:\s*([^;.]+)", corpo)
    if ms:
        situ = limpar(ms.group(1))
    links = {}
    for c in celulas:
        links.update(links_lattes(c))
    return {
        "tipo": "projeto", "subtipo": sub, "ano_inicio": ini, "ano_fim": fim,
        "titulo": titulo, "doi": "", "texto": limpar(f"{periodo} | {titulo} | {corpo}")[:3000],
        "papel": "integrante", "participantes": integrantes, "links": links, "situacao": situ,
    }


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------
LIMIAR_TRIGRAMA = 0.8

COMPUTACAO = [  # classificacao preliminar (conferir no SIGAA)
    "ADLER DINIZ DE SOUZA", "ADRIANA PREST MATTEDI", "ALEXANDRE CARLOS BRANDAO RAMOS",
    "BRUNO GUAZZELLI BATISTA", "CARLOS MINORU TAMAKI", "ISABELA NEVES DRUMMOND",
    "JOAO BOSCO SCHUMAM CUNHA", "LAERCIO AUGUSTO BALDOCHI JUNIOR", "LAERCIO PIOLI JUNIOR",
    "LUIZ OLMES CARVALHO", "MELISE MARIA VEIGA DE PAULA", "PHYLLIPE DE SOUZA LIMA FRANCISCO",
    "RAFAEL DE MAGALHAES DIAS FRINHANI", "ROBERTO CLAUDINO DA SILVA", "RODRIGO DUARTE SEABRA",
    "SAVIO SOARES DIAS", "SIMONE DE SOUSA BORGES", "VANESSA CRISTINA OLIVEIRA DE SOUZA",
]


def carregar_lista_unifei(caminho):
    """Lista de docentes da UNIFEI (nome, instituto, campus, vinculo), copiada do SIGAA."""
    pessoas = []
    with open(caminho, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r["nome_norm"] = normalizar(r["nome"])
            r["chave"] = chave_canonica(r["nome"])
            pessoas.append(r)
    por_chave = defaultdict(list)
    for p in pessoas:
        por_chave[p["chave"]].append(p)
    return pessoas, por_chave


def casar_unifei(nome, pessoas, por_chave):
    """Rotula um colaborador como docente da UNIFEI. Nome completo: exato ou trigramas;
    nome em formato de citacao: chave canonica unica com ao menos duas iniciais."""
    n = normalizar(nome)
    if not n:
        return None, ""
    if "," not in n and len(n.split()) >= 3:
        for p in pessoas:
            if p["nome_norm"] == n:
                return p, "nome_exato"
        melhor, pm = 0.0, None
        for p in pessoas:
            if p["nome_norm"].split()[0] != n.split()[0]:
                continue
            sim = similaridade(n, p["nome_norm"])
            if sim > melhor:
                melhor, pm = sim, p
        if melhor >= LIMIAR_TRIGRAMA:
            return pm, f"trigramas_{melhor:.2f}"
        return None, ""
    k = chave_canonica(nome)
    cands = [p for p in por_chave.get(k, []) if compativel(nome, p["nome"])]
    if len(k.split(" ")[-1]) >= 2 and len(cands) == 1 and len(por_chave.get(k, [])) == 1:
        return cands[0], "chave_unica"
    return None, ""


def executar(entrada, saida, inicio=2020, fim=2025, lista_unifei=None):
    from types import SimpleNamespace
    a = SimpleNamespace(entrada=entrada, saida=saida, inicio=inicio, fim=fim)
    pessoas_unifei, chaves_unifei = carregar_lista_unifei(lista_unifei) if lista_unifei else ([], {})
    os.makedirs(a.saida, exist_ok=True)

    arquivos = sorted(glob.glob(os.path.join(a.entrada, "**", "*.html"), recursive=True))
    docentes, registros, avisos = [], [], []
    for arq in arquivos:
        texto = ler_html(arq)
        soup = BeautifulSoup(texto, "lxml")
        d = extrair_docente(soup, texto)
        d["arquivo_html"] = os.path.basename(arq)
        if not d["id_lattes"]:
            avisos.append(f"{d['arquivo_html']}: identificador Lattes não encontrado")
            continue
        d["camada_sugerida"] = "Computação" if normalizar(d["nome"]) in COMPUTACAO else "Matemática"
        for campo in ("orcid", "doutorado", "ano_ingresso_unifei", "data_atualizacao_lattes"):
            if not d[campo]:
                avisos.append(f"{d['nome']}: campo '{campo}' não encontrado no currículo")
        regs = extrair_registros(soup)
        for r in regs:
            r["docente_id_lattes"] = d["id_lattes"]
            r["docente_nome"] = d["nome"]
        registros.extend(regs)
        docentes.append(d)

    # dicionario de variantes de nome dos docentes do IMC
    por_variante = defaultdict(set)
    por_nome = {}
    ids_docentes = {d["id_lattes"]: d for d in docentes}
    for d in docentes:
        por_nome[normalizar(d["nome"])] = d["id_lattes"]
        por_variante[normalizar(d["nome"])].add(d["id_lattes"])
        for v in d["nomes_citacao"].split(";"):
            if normalizar(v):
                por_variante[normalizar(v)].add(d["id_lattes"])
    # docentes do IMC sem Lattes (presentes na lista do SIGAA): identificador provisorio
    if pessoas_unifei:
        nomes_lattes = set(por_nome)
        for p in pessoas_unifei:
            if p["instituto"] == "IMC" and p["vinculo"] == "Permanente":
                if not any(similaridade(p["nome_norm"], n) >= LIMIAR_TRIGRAMA for n in nomes_lattes):
                    pid = "SEM_LATTES_" + p["nome_norm"].replace(" ", "_")
                    por_nome[p["nome_norm"]] = pid
                    por_variante[p["nome_norm"]].add(pid)
                    ids_docentes[pid] = {"nome": p["nome"], "id_lattes": pid}
                    avisos.append(f"Docente do IMC sem currículo Lattes incluído com identificador provisório: {p['nome']}")
    nome_completo_doc = {idd: nome for nome, idd in por_nome.items()}
    chave_doc = defaultdict(set)
    for nome, idd in por_nome.items():
        chave_doc[chave_canonica(nome)].add(idd)
    ambiguas = {v for v, ids in por_variante.items() if len(ids) > 1}
    for v in sorted(ambiguas):
        avisos.append(f"Variante de nome compartilhada por mais de um docente (ignorada no casamento): '{v}'")

    def casar(nome, links, dono):
        n = normalizar(nome)
        if n in links and links[n] in ids_docentes:
            return links[n], "link_lattes"
        if n in por_variante and n not in ambiguas:
            return next(iter(por_variante[n])), "variante_citacao"
        # forma de citacao nao declarada pelo docente: chave canonica unica e prenomes compativeis
        if "," in n:
            k = chave_canonica(nome)
            cands = [i for i in chave_doc.get(k, set()) if compativel(nome, nome_completo_doc[i])]
            if len(k.split(" ")[-1]) >= 2 and len(cands) == 1:
                return cands[0], "chave_unica_docente"
        # nome completo sem virgula (integrantes de projeto): similaridade por trigramas
        if "," not in n and len(n.split()) >= 3:
            melhor, idm = 0.0, ""
            for nome_doc, idd in por_nome.items():
                if n.split()[0] != nome_doc.split()[0]:
                    continue
                sim = similaridade(n, nome_doc)
                if sim > melhor:
                    melhor, idm = sim, idd
            if melhor >= LIMIAR_TRIGRAMA:
                return idm, f"trigramas_{melhor:.2f}"
        return "", ""

    # numeracao e tabela de participantes
    linhas_part = []
    for i, r in enumerate(registros, 1):
        r["id_registro"] = f"R{i:06d}"
        dono = r["docente_id_lattes"]
        r["no_periodo"] = bool(
            r["ano_inicio"] and r["ano_inicio"] <= a.fim and (r["ano_fim"] or r["ano_inicio"]) >= a.inicio
        )
        docs = {dono}
        for p in r["participantes"]:
            idd, metodo = casar(p, r["links"], dono)
            if idd:
                docs.add(idd)
            pu, metodo_u = (None, "")
            if not idd and pessoas_unifei:
                pu, metodo_u = casar_unifei(p, pessoas_unifei, chaves_unifei)
            linhas_part.append({
                "id_registro": r["id_registro"], "docente_id_lattes": dono, "tipo": r["tipo"],
                "subtipo": r["subtipo"], "ano_inicio": r["ano_inicio"], "no_periodo": r["no_periodo"],
                "nome_raw": p, "nome_normalizado": normalizar(p), "chave_canonica": chave_canonica(p),
                "lattes_id_link": r["links"].get(normalizar(p), ""),
                "docente_imc_id": idd, "metodo_casamento": metodo,
                "e_o_proprio_docente": idd == dono,
                "unifei_nome": pu["nome"] if pu else "",
                "unifei_instituto": pu["instituto"] if pu else "",
                "unifei_campus": pu["campus"] if pu else "",
                "metodo_unifei": metodo_u,
            })
        r["docentes_imc_no_registro"] = ";".join(sorted(docs))
        r["n_participantes"] = len(r["participantes"])
        # o proprio docente precisa aparecer nas publicacoes/bancas; se nao, avisar
        if r["tipo"] in ("publicacao",) and not any(
            lp["e_o_proprio_docente"] for lp in linhas_part[-len(r["participantes"]):] if r["participantes"]
        ):
            avisos.append(f"{r['docente_nome']}: docente não reconhecido entre os autores de {r['id_registro']} ({r['subtipo']}, {r['ano_inicio']})")
        if r["ano_inicio"] is None:
            avisos.append(f"{r['docente_nome']}: ano não identificado em {r['id_registro']} ({r['subtipo']})")

    # contagens por docente no periodo
    cont = defaultdict(Counter)
    for r in registros:
        if r["no_periodo"]:
            cont[r["docente_id_lattes"]][r["subtipo"]] += 1
            cont[r["docente_id_lattes"]]["_" + r["tipo"]] += 1

    cols_doc = ["nome", "id_lattes", "orcid", "camada_sugerida", "nomes_citacao", "doutorado",
                "doutorado_instituicao", "doutorado_ano", "ano_ingresso_unifei",
                "data_atualizacao_lattes", "arquivo_html",
                "n_publicacoes_periodo", "n_formacao_periodo", "n_projetos_periodo"]
    with open(os.path.join(a.saida, "docentes.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols_doc)
        w.writeheader()
        for d in sorted(docentes, key=lambda x: normalizar(x["nome"])):
            c = cont[d["id_lattes"]]
            w.writerow({**{k: d.get(k, "") for k in cols_doc},
                        "n_publicacoes_periodo": c["_publicacao"],
                        "n_formacao_periodo": c["_formacao"],
                        "n_projetos_periodo": c["_projeto"]})

    cols_reg = ["id_registro", "docente_id_lattes", "docente_nome", "tipo", "subtipo", "papel",
                "situacao", "ano_inicio", "ano_fim", "no_periodo", "titulo", "doi", "discente",
                "n_participantes", "participantes", "docentes_imc_no_registro", "texto"]
    with open(os.path.join(a.saida, "cooperacoes.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols_reg)
        w.writeheader()
        for r in registros:
            linha = {k: r.get(k, "") for k in cols_reg}
            linha["participantes"] = " | ".join(r["participantes"])
            if linha["ano_fim"] == 9999:
                linha["ano_fim"] = "Atual"
            w.writerow(linha)

    with open(os.path.join(a.saida, "cooperacoes_participantes.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas_part[0].keys()))
        w.writeheader()
        w.writerows(linhas_part)

    # relatorio
    tot = Counter((r["tipo"], r["subtipo"]) for r in registros if r["no_periodo"])
    dd = defaultdict(int)
    for r in registros:
        if r["no_periodo"]:
            ids = r["docentes_imc_no_registro"].split(";")
            if len(ids) > 1:
                dd[r["tipo"]] += 1
    with open(os.path.join(a.saida, "relatorio_extracao.md"), "w", encoding="utf-8") as f:
        f.write(f"# Relatório de extração Lattes\n\nCurrículos lidos: {len(arquivos)}; "
                f"docentes extraídos: {len(docentes)}; registros: {len(registros)} "
                f"({sum(tot.values())} no período {a.inicio}–{a.fim}).\n\n")
        f.write("## Registros no período por subtipo\n\n| tipo | subtipo | n |\n|---|---|---|\n")
        for (t, s), n in sorted(tot.items()):
            f.write(f"| {t} | {s} | {n} |\n")
        f.write("\n## Registros no período com dois ou mais docentes do IMC identificados\n\n")
        for t, n in sorted(dd.items()):
            f.write(f"- {t}: {n}\n")
        f.write("\n(Contagem por currículo: um mesmo trabalho declarado por dois docentes aparece duas vezes; "
                "a deduplicação é feita na etapa de integração.)\n\n")
        if pessoas_unifei:
            ext = [lp for lp in linhas_part if lp["no_periodo"] and not lp["docente_imc_id"]]
            rot = Counter(lp["unifei_instituto"] for lp in ext if lp["unifei_instituto"])
            met = Counter(lp["metodo_unifei"].split("_")[0] for lp in ext if lp["metodo_unifei"])
            f.write("## Colaboradores externos rotulados como docentes da UNIFEI (no período)\n\n")
            f.write(f"Participações externas no período: {len(ext)}; rotuladas como docente da UNIFEI: "
                    f"{sum(rot.values())}.\n\n| instituto | participações |\n|---|---|\n")
            for k, v in rot.most_common():
                f.write(f"| {k} | {v} |\n")
            f.write(f"\nMétodos: {dict(met)}. Rótulos por 'chave_unica' (nome em formato de citação) são "
                    "candidatos e devem ser confirmados na integração com o OpenAlex.\n\n")
        f.write(f"## Avisos ({len(avisos)})\n\n")
        for av in avisos:
            f.write(f"- {av}\n")
    print(f"{len(docentes)} docentes, {len(registros)} registros, {len(avisos)} avisos -> {a.saida}")

