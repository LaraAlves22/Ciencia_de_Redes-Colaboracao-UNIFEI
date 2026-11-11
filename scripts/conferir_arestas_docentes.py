#!/usr/bin/env python3
"""
conferir_arestas_docentes.py - Classifica a evidencia de cada aresta entre docentes do IMC.

Para cada par de docentes, percorre os eventos de cooperacao que o ligam e registra a evidencia
mais forte encontrada:
  1. dois_curriculos      o evento foi declarado no Lattes dos dois docentes;
  2. link_lattes          um so curriculo, com link para o Lattes do outro docente;
  3. variante_citacao     um so curriculo, nome igual a uma variante de citacao declarada pelo outro;
  4. trigramas            um so curriculo, nome completo com similaridade de trigramas >= limiar;
  5. chave_unica_docente  um so curriculo, casamento apenas pela chave canonica (mais fragil).
Docentes sem Lattes (SEM_LATTES_*) so podem ter evidencia de um curriculo.

Saida: CSV com uma linha por aresta, para a conferencia manual comecar pelas mais frageis.

Uso:
  python scripts/conferir_arestas_docentes.py --lattes saida_lattes --redes redes --saida conferencia_arestas_DD.csv
"""
import argparse
import os
from collections import defaultdict

import pandas as pd

ORDEM = ["dois_curriculos", "link_lattes", "variante_citacao", "trigramas", "chave_unica_docente"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattes", required=True)
    ap.add_argument("--redes", required=True, help="pasta com eventos.csv e nos.csv (notebook 03)")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    part = pd.read_csv(os.path.join(a.lattes, "cooperacoes_participantes.csv"), dtype=str, keep_default_na=False)
    coop = pd.read_csv(os.path.join(a.lattes, "cooperacoes.csv"), dtype=str, keep_default_na=False)
    ev = pd.read_csv(os.path.join(a.redes, "eventos.csv"), dtype=str, keep_default_na=False)
    nos = pd.read_csv(os.path.join(a.redes, "nos.csv"), dtype=str, keep_default_na=False)
    nome = dict(zip(nos.id, nos.rotulo))
    dono = dict(zip(coop.id_registro, coop.docente_id_lattes))
    # metodo pelo qual o docente X foi reconhecido no registro R
    metodo = {}
    for r in part.itertuples():
        if r.docente_imc_id and not r.e_o_proprio_docente == "True":
            m = r.metodo_casamento.split("_")[0] if r.metodo_casamento.startswith("trigramas") else r.metodo_casamento
            metodo[(r.id_registro, r.docente_imc_id)] = "trigramas" if m == "trigramas" else r.metodo_casamento
    pares = defaultdict(lambda: {"eventos": 0, "melhor": None, "exemplo": ""})
    for e in ev[ev.fonte == "lattes"].itertuples():
        docs = sorted(d for d in e.docentes.split(";") if d)
        if len(docs) < 2:
            continue
        regs = [x for x in e.registros.split(";") if x]
        donos = {dono.get(x) for x in regs}
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                p, q = docs[i], docs[j]
                if p in donos and q in donos:
                    ev_m = "dois_curriculos"
                else:
                    # docente que nao declarou o evento: melhor metodo pelo qual foi reconhecido em
                    # algum registro do evento; se nenhum dos dois declarou, vale o mais fragil dos dois
                    def melhor(d):
                        ms = [metodo.get((x, d)) for x in regs if dono.get(x) != d]
                        ms = [m for m in ms if m in ORDEM]
                        return min(ms, key=ORDEM.index) if ms else "chave_unica_docente"
                    nao_donos = [d for d in (p, q) if d not in donos]
                    ev_m = max((melhor(d) for d in nao_donos), key=ORDEM.index)
                reg = pares[(p, q)]
                reg["eventos"] += 1
                if reg["melhor"] is None or ORDEM.index(ev_m) < ORDEM.index(reg["melhor"]):
                    reg["melhor"], reg["exemplo"] = ev_m, f"{e.tipo} {e.ano}: {(e.discente or e.titulo)[:80]}"
    linhas = [{"docente_a": nome.get(p, p), "docente_b": nome.get(q, q), "id_a": p, "id_b": q,
               "eventos": v["eventos"], "evidencia_mais_forte": v["melhor"], "exemplo": v["exemplo"],
               "conferido_manualmente": ""} for (p, q), v in pares.items()]
    df = pd.DataFrame(linhas)
    df["_o"] = df["evidencia_mais_forte"].map(ORDEM.index)
    df.sort_values(["_o", "eventos"], ascending=[False, True]).drop(columns="_o").to_csv(
        a.saida, index=False, encoding="utf-8-sig")
    print(len(df), "arestas entre docentes")
    print(df["evidencia_mais_forte"].value_counts().to_string())


if __name__ == "__main__":
    main()
