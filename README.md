# Rede de Colaboração Científica - UNIFEI

Projeto da disciplina XMCO09 (Ciência de Redes - UNIFEI): caracterização estrutural, modelagem sintética e avaliação de robustez da rede de coautoria entre pesquisadores dos institutos da universidade (OpenAlex, 2020 a 19/08/2026).

## Estrutura do Repositório

```
├── dados_de_extracao/          # nodes.csv e edges.csv
├── distribuicao_institutos/    # Relatórios descritivos e distribuições por instituto
├── figuras_redes/              # Visualizações das redes por instituto e da rede global
├── notebooks/                  # Extração, análise por instituto e pesquisadores-ponte
├── modelagem.py                # Calibração e geração dos modelos de referência (ER, WS, BA, HK, configuração, LFR)
├── ccdf_modelos.py             # CCDF, ajuste de lei de potência (MLE) e teste de Vuong
├── aderencia_modelos.py        # Erro relativo e distância KS à baseline
├── at3_complementos.py         # Mann-Whitney/Holm, delta de Cliff, sensibilidade de parâmetros e bootstrap
├── robustez_v1.py              # Falhas aleatórias, ataques direcionados e Molloy-Reed
├── resultados_*/               # Saídas de cada etapa (modelagem, ccdf, aderência, at3, robustez)
```

## Dados

- `nodes.csv`: `id` (OpenAlex Author ID), `label` (nome) e `instituto` (siglas separadas por `;`; "Não identificado" quando não há correspondência).
- `edges.csv`: `source`, `target` e `weight` (trabalhos conjuntos no período).
- O instituto é atribuído por correspondência textual entre as afiliações brutas e o nome completo de cada instituto (português ou inglês).

## Instalação

Python 3.13 e NetworkX 3.5 foram usados nos experimentos do artigo (Python 3.10+ deve funcionar):

```
pip install networkx python-igraph pandas numpy scipy matplotlib
```

## Como Executar

A partir da raiz do repositório:

```
python modelagem.py --data dados_de_extracao --out resultados_modelagem --reps 20
python ccdf_modelos.py --data dados_de_extracao --resumo resultados_modelagem/resumo.json --out resultados_ccdf
python aderencia_modelos.py --data dados_de_extracao --modelagem resultados_modelagem --out resultados_aderencia
python at3_complementos.py --data dados_de_extracao --modelagem resultados_modelagem --out resultados_at3_complementos
python robustez_v1.py --data dados_de_extracao --modelagem resultados_modelagem --out resultados_robustez
```

As sementes são fixas (0 a 19 por modelo), então os resultados são reproduzíveis.
