#!/usr/bin/env python3
"""
atualizar_rede_v2.py - Aplica a rede ja integrada (saida do notebook 03 com OpenAlex) as duas
mudancas de metodo feitas apos o parecer da V1, sem precisar refazer a coleta no OpenAlex:

1. M2 com cada grupo (P, F, J) entrando pela MEDIA de seus tipos, o que garante w_ij em (0, 1].
   O M2 e recalculado a partir das colunas n_<tipo> de cada arquivo de arestas.
2. Orientandos sem vinculo UNIFEI no registro passam a receber a categoria da instituicao do
   curso informada no proprio registro de orientacao do Lattes (nacional ou estrangeira).

Os mesmos ajustes ja estao no codigo do notebook 03; este script so evita repetir a etapa B do
notebook 02 (consulta a API do OpenAlex) para atualizar a rede usada no artigo.

Uso:
  python scripts/atualizar_rede_v2.py --redes redes_corrigido --lattes saida_lattes --saida redes_v2
"""
import argparse
import os
import shutil
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from integracao import (ESTRANGEIRA_RE, PESOS_M2, TAMANHO_GRUPO, TIPO_GRUPO,  # noqa: E402
                        instituicao_orientacao, normalizar)


def recalcular_m2(arestas, pesos=PESOS_M2):
    tipos = [c[2:] for c in arestas.columns if c.startswith("n_") and c != "n_eventos"]
    tot = defaultdict(lambda: defaultdict(float))
    for r in arestas.itertuples(index=False):
        d = r._asdict()
        for t in tipos:
            tot[d["source"]][t] += d["n_" + t]
            tot[d["target"]][t] += d["n_" + t]
    m2 = []
    for r in arestas.itertuples(index=False):
        d = r._asdict()
        s = 0.0
        for t in tipos:
            v = d["n_" + t]
            if v > 0:
                g = TIPO_GRUPO[t]
                s += pesos[g] / TAMANHO_GRUPO[g] * 2 * v / (tot[d["source"]][t] + tot[d["target"]][t])
        m2.append(s)
    out = arestas.copy()
    out["M2"] = m2
    return out


def instituicao_dos_orientandos(cooperacoes):
    """normalizar(discente) -> categoria lida dos registros de orientacao sem mencao a UNIFEI."""
    ori = cooperacoes[(cooperacoes["tipo"] == "formacao") & cooperacoes["subtipo"].str.startswith("orientacao")
                      & (cooperacoes["no_periodo"] == "True")]
    cats = defaultdict(Counter)
    for r in ori.itertuples():
        t = normalizar(r.texto)
        if "UNIVERSIDADE FEDERAL DE ITAJUBA" in t or "UNIFEI" in t:
            cats[normalizar(r.discente)]["outro_vinculo_unifei"] += 1
            continue
        inst = instituicao_orientacao(r.texto)
        if inst:
            cats[normalizar(r.discente)]["estrangeira" if ESTRANGEIRA_RE.search(inst) else "nacional"] += 1
    return {k: c.most_common(1)[0][0] for k, c in cats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redes", required=True, help="pasta com nos.csv, arestas.csv... (saida do notebook 03)")
    ap.add_argument("--lattes", required=True, help="pasta saida_lattes (notebook 01)")
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    os.makedirs(a.saida, exist_ok=True)
    coop = pd.read_csv(os.path.join(a.lattes, "cooperacoes.csv"), dtype=str, keep_default_na=False)
    cat_orientando = instituicao_dos_orientandos(coop)
    relatorio = {}
    for nome in sorted(os.listdir(a.redes)):
        src, dst = os.path.join(a.redes, nome), os.path.join(a.saida, nome)
        if nome.startswith("arestas") and nome.endswith(".csv"):
            ar = pd.read_csv(src)
            novo = recalcular_m2(ar)
            novo.to_csv(dst, index=False, encoding="utf-8-sig")
            relatorio[nome] = {"M2_max_antes": float(ar["M2"].max()), "M2_max_depois": float(novo["M2"].max())}
        elif nome.startswith("nos") and nome.endswith(".csv"):
            nos = pd.read_csv(src, dtype=str, keep_default_na=False)
            mudou = Counter()
            for i, r in nos.iterrows():
                if r["tipo_no"] != "X" or r["categoria"] != "nao_identificado" or not r["id"].startswith("N:"):
                    continue
                c = cat_orientando.get(r["id"][2:])
                if c:
                    nos.at[i, "categoria"] = c
                    nos.at[i, "metodo_identificacao"] = (r["metodo_identificacao"] + ";discente_instituicao_registro").strip(";")
                    mudou[c] += 1
            nos.to_csv(dst, index=False, encoding="utf-8-sig")
            relatorio[nome] = {"reclassificados": dict(mudou),
                               "categorias": nos[nos.tipo_no == "X"].categoria.value_counts().to_dict()}
        elif os.path.isfile(src):
            shutil.copy(src, dst)
    for k, v in relatorio.items():
        print(k, v)


if __name__ == "__main__":
    main()
