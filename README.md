# Rede de Colaboração Científica - UNIFEI

Projeto desenvolvido para a disciplina XMCO09 (Ciência de Redes - UNIFEI) com foco na caracterização estrutural, modelagem sintética e avaliação de robustez da rede de coautoria entre pesquisadores dos institutos da universidade.

---

## Estrutura do Repositório

```text
├── dados_de_extracao/       # Dados brutos e tratados (nodes.csv, edges.csv)
├── distribuicao_institutos/ # Relatórios descritivos e distribuições por instituto
├── figuras_redes/           # Visualizações dos grafos setoriais e rede global
├── notebooks/               # Análises exploratórias, extração e detecção de pontes
├── modelagem_v3.py          # Calibração e geração de modelos sintéticos de referência
├── ccdf_modelos.py          # Ajuste de lei de potência (MLE) e curvas CCDF
├── aderencia_modelos.py     # Métricas de distância (erro relativo e KS) à baseline
├── at3_complementos.py      # Testes estatísticos, varredura de hiperparâmetros e bootstrap
└── robustez_v1.py           # Simulação de ataques e tolerância a falhas na rede

```

---

## Descrição dos Módulos

### Pipeline de Modelagem e Validação

* **`modelagem_v3.py`**: Carrega a componente gigante da rede da UNIFEI ($N=2.987$, $L=10.432$), particiona comunidades via algoritmo de Louvain e sintetiza réplicas dos modelos ER, WS, BA, HK, Modelo de Configuração e LFR.


* **`ccdf_modelos.py`**: Gera curvas empíricas de probabilidade complementar acumulada (CCDF) com faixas de dispersão e calcula o ajuste MLE para lei de potência discreta versus modelo log-normal (teste de Vuong).


* **`aderencia_modelos.py`**: Compara as réplicas dos modelos com a baseline a partir de erro percentual e estatística Kolmogorov-Smirnov ($D_{KS}$), exportando tabela LaTeX e matriz de calor.


* **`at3_complementos.py`**: Executa testes de Kruskal-Wallis e Mann-Whitney (com ajuste de Holm e delta de Cliff), varredura dos parâmetros $p$ (HK) e $\mu$ (LFR), além de validação por bootstrap semiparamétrico.


* **`robustez_v1.py`**: Avalia o decaimento da componente conexa sob falhas aleatórias e ataques direcionados (grau adaptativo, intermediação, força e articulações), mapeando os nós com papel de ponte.



### Notebooks e Análise Setorial

* **`notebooks/extracao_de_dados_unifei_todos_institutos.ipynb`**: Coleta, padronização e estruturação das publicações dos docentes.


* **`notebooks/redes_por_instituto.ipynb` & `plotar_redes_por_instituto.ipynb**`: Geração das métricas topológicas e gráficos específicos para cada instituto (IMC, ISEE, IEM, IEI, IEPG, IESTI, IFQ e IRN).


* **`notebooks/analise_pontes_institutos.ipynb`**: Identificação de pesquisadores centrais na integração interdisciplinar entre departamentos.



---

## Instalação

Ambiente recomendado: **Python 3.10+**. Instale os pacotes necessários via terminal:

```bash
pip install networkx pandas numpy scipy matplotlib

```

---

## Como Executar

Execute os scripts a partir da raiz do repositório mantendo o diretório `dados_de_extracao/` acessível:

```bash
# 1. Calibração e geração das réplicas sintéticas
python modelagem_v3.py --data dados_de_extracao --out resultados_modelagem --reps 20

# 2. CCDF e ajuste de cauda pesada
python ccdf_modelos.py --data dados_de_extracao --resumo resultados_modelagem/resumo.json --out resultados_ccdf

# 3. Cálculo da distância e aderência dos modelos
python aderencia_modelos.py --data dados_de_extracao --modelagem resultados_modelagem --out resultados_aderencia

# 4. Testes de hipótese, varredura de parâmetros e bootstrap
python at3_complementos.py --data dados_de_extracao --modelagem resultados_modelagem --out resultados_at3_complementos

# 5. Análise de percolação e ataque a nós críticos
python robustez_v1.py --data dados_de_extracao --modelagem resultados_modelagem --out resultados_robustez

```
