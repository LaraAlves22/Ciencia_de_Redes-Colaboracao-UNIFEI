#!/usr/bin/env python3
"""Figuras e tabelas LaTeX das seções de modelos de referência (AT3) e robustez (AT4)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8.5, "axes.titleweight": "bold",
                     "legend.frameon": False, "legend.fontsize": 7, "pdf.fonttype": 42})
vir = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))
COR = {"Baseline": "#111111", "ER": "#8c8c8c", "WS": "#1f77b4", "BA": "#ff7f0e", "HK": "#2ca02c", "Configuração": "#9467bd", "LFR": "#8c564b"}
PROPS = ["d", "diam", "cc_med", "cc_glob", "Q", "kmax", "r", "S"]
NOMES = {"d": r"$\langle d\rangle$", "diam": "Diâm.", "cc_med": "CA\nméd.", "cc_glob": "CA\nglob.", "Q": "$Q$", "kmax": r"$k_{\max}$", "r": "$r$", "S": "$S$"}

def fig_at3():
    r = json.load(open("resultados/at3/resumo.json")); base = r["baseline"]; rep = pd.read_csv("resultados/at3/replicas.csv"); gr = np.load("resultados/at3/graus.npz")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.1, 1]})
    a = ax[0]
    for nm, ck in (("ER", "ER"), ("WS", "WS"), ("BA", "BA"), ("HK", "HK"), ("CM", "Configuração"), ("LFR", "LFR")):
        k = np.sort(gr[nm]); y = 1 - np.arange(len(k)) / len(k)
        a.step(k, y, where="post", color=COR[ck], lw=1.1, ls="--" if nm == "CM" else "-", label=ck)
    k = np.sort(gr["baseline"]); a.plot(k, 1 - np.arange(len(k)) / len(k), "o", ms=2.2, color=COR["Baseline"], label="Baseline", zorder=5)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("Grau $k$"); a.set_ylabel("$P(K\\geq k)$"); a.set_title("(a) Distribuição de grau (CCDF)")
    a.legend(loc="upper right", ncol=2, handlelength=1.4, columnspacing=1)
    b = ax[1]; ordem = ["ER", "WS", "BA", "HK", "LFR", "Configuração"]
    M = np.array([[min(abs(rep[rep.modelo == m][p].mean() - base[p]) / abs(base[p]), 1) for p in PROPS] for m in ordem]) * 100
    im = b.imshow(M, cmap="Blues_r", vmin=0, vmax=100, aspect="auto")
    b.set_xticks(range(len(PROPS))); b.set_xticklabels([NOMES[p] for p in PROPS]); b.set_yticks(range(len(ordem))); b.set_yticklabels(ordem)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            b.text(j, i, f"{M[i, j]:.0f}".replace(".", ","), ha="center", va="center", fontsize=7, color="white" if M[i, j] < 45 else "black")
    b.set_title("(b) Erro relativo à baseline (%)"); b.spines[:].set_visible(False); b.tick_params(length=0)
    fig.tight_layout(); [fig.savefig(f"resultados/at3/fig_modelos.{e}", dpi=300) for e in ("pdf", "png")]

def fig_at4():
    c = np.load("resultados/at4/curvas.npz"); nD = 48; ks = np.arange(nD + 1) / nD * 100
    g = pd.read_csv("resultados/at4/ataque_geral_modelos.csv"); fs = np.round(np.arange(0, 0.2001, 0.005), 3)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.9))
    estilos = {"grau_DD": ("#d62728", "Grau (D–D)"), "intermediacao": ("#9467bd", "Intermediação"), "exclusivos": ("#ff7f0e", "Externos exclusivos")}
    for a, m, tit, yl in ((ax[0], "SD", "(a) Integração interna", "$S_D$ (fração dos 48 docentes)"), (ax[1], "cob", "(b) Alcance externo", "Externos com vínculo (fração)")):
        arr = c[f"arr_{m}"]; a.fill_between(ks, np.percentile(arr, 2.5, 0), np.percentile(arr, 97.5, 0), color="#9a9a9a", alpha=.35, lw=0)
        a.plot(ks, np.median(arr, 0), color="#555555", lw=1.4, label="Aleatória (mediana; IC 95%)")
        for s, (col, lab) in estilos.items(): a.plot(ks, c[f"dir_{s}_{m}"], color=col, lw=1.3, label=lab)
        a.axhline(0.5 * 0.8541666 if m == "SD" else 0.5, color="k", ls=":", lw=.8)
        a.set_xlabel("Docentes afastados (%)"); a.set_ylabel(yl); a.set_title(tit); a.yaxis.set_major_formatter(vir); a.set_ylim(0, 1.02)
    h, l = ax[0].get_legend_handles_labels()
    a = ax[2]
    for nm in ("Baseline", "Configuração", "HK", "BA", "ER", "WS"):
        d = g[g.modelo == nm]; M = np.array([[r[f"atk_{f}"] for f in fs] for _, r in d.iterrows()]); M = M / M[:, [0]]
        a.plot(fs * 100, M.mean(0), color=COR[nm], lw=1.4 if nm == "Baseline" else 1.1, ls="--" if nm == "Configuração" else "-", label=nm)
    d = g[g.modelo == "Baseline"]; M = np.array([[r[f"rand_{f}"] for f in fs] for _, r in d.iterrows()]); a.plot(fs * 100, M.mean(0), color="k", ls=":", lw=1, label="Baseline, aleatória")
    a.set_xlabel("Nós removidos (% de $N$)"); a.set_ylabel("$S/S_0$ (maior componente)"); a.set_title("(c) Rede completa, por grau"); a.yaxis.set_major_formatter(vir)
    a.legend(loc="center right", fontsize=6.3); a.set_ylim(0, 1.02)
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7); fig.tight_layout(rect=(0, 0.07, 1, 1)); [fig.savefig(f"resultados/at4/fig_robustez.{e}", dpi=300) for e in ("pdf", "png")]

def f3(x): return f"{x:.3f}".replace(".", "{,}")
def tabela_at3():
    r = json.load(open("resultados/at3/resumo.json")); base = r["baseline"]; rep = pd.read_csv("resultados/at3/replicas.csv")
    cols = ["L", "d", "diam", "cc_med", "cc_glob", "Q", "kmax", "r", "S", "DKS", "erro"]
    ordem = [("ER", "ER"), ("WS", f"WS ($p{{=}}{r['escolhidos']['p_ws']:g}$)".replace(".", "{,}")), ("BA", "BA"), ("HK", f"HK ($p{{=}}{r['escolhidos']['p_hk']:g}$)".replace(".", "{,}")),
             ("LFR", f"LFR ($\\mu{{=}}{r['escolhidos']['mu_lfr']:g}$)".replace(".", "{,}")), ("Configuração", "Configuração")]
    melhor = {}
    for c in PROPS:
        dif = {o[0]: abs(rep[rep.modelo == o[0]][c].mean() - base[c]) for o in ordem}; mn = min(dif.values()); melhor[c] = {k for k, v in dif.items() if v <= mn + 5e-4}
    melhor["DKS"] = {min(ordem, key=lambda o: rep[rep.modelo == o[0]]["DKS"].mean())[0]}; melhor["erro"] = {min(ordem, key=lambda o: rep[rep.modelo == o[0]]["erro"].mean())[0]}
    lin = [f"Baseline & {int(base['L']):,}".replace(",", ".") + f" & {f3(base['d'])} & {f3(base['diam'])} & {f3(base['cc_med'])} & {f3(base['cc_glob'])} & {f3(base['Q'])} & {f3(base['kmax'])} & {f3(base['r'])} & {f3(base['S'])} & --- & --- \\\\", "\\midrule"]
    for key, rot in ordem:
        d = rep[rep.modelo == key]; cel = []
        for c in cols:
            if c == "L": cel.append(f"{d[c].mean():,.0f}".replace(",", ".")); continue
            t = f"{f3(d[c].mean())}\\,$\\pm$\\,{f3(d[c].std())}"; cel.append(f"\\textbf{{{t}}}" if key in melhor.get(c, set()) else t)
        lin.append(f"{rot} & " + " & ".join(cel) + " \\\\")
    cab = ("\\begin{table*}[t]\n\\centering\\footnotesize\n\\caption{Baseline (maior componente da rede completa, $N=2.010$) e modelos de referência calibrados para o mesmo $N$ e $L$. Média $\\pm$ desvio-padrão de 30 réplicas; "
           "$\\langle d\\rangle$ e diâmetro no maior componente de cada réplica; CA: coeficiente de agrupamento; $Q$: modularidade (Louvain); $r$: assortatividade por grau; $S$: fração do maior componente; "
           "$D_{KS}$: distância KS entre as distribuições de grau; Erro: média dos oito erros relativos (limitados a 1). Em negrito, o valor mais próximo da baseline.}\n\\label{tab:modelos}\n"
           "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{@{}lrrrrrrrrrrr@{}}\n\\toprule\n\\textbf{Rede} & $L$ & $\\langle d\\rangle$ & Diâm. & CA méd. & CA glob. & $Q$ & $k_{\\max}$ & $r$ & $S$ & $D_{KS}$ & Erro \\\\\n\\midrule\n")
    open("resultados/at3/tab_modelos.tex", "w", encoding="utf-8").write(cab + "\n".join(lin) + "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table*}\n")

def tabela_at4():
    r = json.load(open("resultados/at4/resumo.json")); al = r["aleatoria"]; di = {d["cenario"]: d for d in r["dirigidos"]}
    q = lambda v, f=f3: f"{f(v[2])} [{f(v[0])}; {f(v[4])}]"
    inteiro = lambda x: f"{x:.0f}"
    lin = [f"Aleatória (1.000 sequências) & {q(al['k_falha_SD'], inteiro)} & {q(al['k_falha_cob'], inteiro)} & {q(al['k_falha_inter'], inteiro)} & {q(al['R_SD'])} & {q(al['R_cob'])} \\\\"]
    for k, rot in (("grau_DD", "Grau entre docentes (adaptativa)"), ("intermediacao", "Intermediação (adaptativa)"), ("exclusivos", "Externos exclusivos (adaptativa)")):
        d = di[k]; lin.append(f"{rot} & {d['k_falha_SD']} & {d['k_falha_cob']} & {d['inter_zero_k']} & {f3(d['R_SD'])} & {f3(d['R_cob'])} \\\\")
    cab = ("\\begin{table*}[t]\n\\centering\\footnotesize\n\\caption{Robustez da rede do IMC ao afastamento de docentes. $k^{*}$: número de docentes afastados (de 48) em que a função deixa de ser preservada "
           "($S_D<0{,}427$; alcance externo $<0{,}5$; nenhum vínculo entre áreas). $R$: média da curva (0 a 1). Aleatória: mediana [IC 95\\%] de 1.000 sequências.}\n\\label{tab:robustez}\n"
           "\\resizebox{0.8\\textwidth}{!}{%\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n\\textbf{Cenário} & $k^{*}_{S_D}$ & $k^{*}_{\\text{alcance}}$ & $k^{*}_{\\text{entre áreas}}$ & $R_{S_D}$ & $R_{\\text{alcance}}$ \\\\\n\\midrule\n")
    open("resultados/at4/tab_robustez.tex", "w", encoding="utf-8").write(cab + "\n".join(lin) + "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table*}\n")

if __name__ == "__main__": fig_at3(); fig_at4(); tabela_at3(); tabela_at4()
