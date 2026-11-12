# Rede de Cooperação Científica dos Docentes do IMC/UNIFEI

Projeto da disciplina XMCO09 – Ciência de Redes (UNIFEI, 2026/2). Analisa a rede de cooperação em pesquisa
dos 48 docentes efetivos do Instituto de Matemática e Computação (IMC), em Itajubá, de 2020 a 2025, para apoiar a
gestão acadêmica da dimensão Pesquisa. A rede é multicamada (Matemática e Computação), multirrelacional
(publicação, orientação, coorientação, banca e projetos de pesquisa e extensão) e tem fronteira de um passo:
entram os docentes e seus colaboradores diretos, sem as cooperações entre colaboradores externos.

A condução segue a CRISP-NET. A versão anterior do estudo, sobre a UNIFEI inteira com o OpenAlex, está em
[`v1_unifei/`](v1_unifei/) (inclui os modelos de referência ER, WS, BA, HK e LFR da Seção 3.5 do artigo).

## Estrutura

```
├── notebooks/                    # pipeline completo, executável no Google Colab (ordem 01 → 04)
│   ├── 01_extrator_lattes.ipynb      # currículos Lattes (HTML) → docentes e cooperações
│   ├── 02_coleta_openalex.ipynb      # identificação dos docentes e obras 2020–2025 no OpenAlex
│   ├── 03_integracao_redes.ipynb     # desambiguação, eventos, rede e modelos de intensidade M1–M3
│   ├── 04_analises.ipynb             # métricas, testes das hipóteses, afastamento, sensibilidade, figuras
│   └── lista_docentes_unifei.csv     # docentes dos oito institutos (SIGAA, 03/11/2026)
├── src/                          # o mesmo código dos notebooks, como módulos Python
├── scripts/
│   ├── atualizar_rede_v2.py          # aplica a uma rede antiga os ajustes pós-parecer (M2 e orientandos), sem refazer o OpenAlex
│   └── conferir_arestas_docentes.py  # evidência de cada aresta entre docentes, para conferência
├── dados/
│   ├── rede/                         # nos*.csv, arestas*.csv e rede_imc.graphml (externos pseudonimizados);
│   │                                 #   variantes _estrito, _sem_massivos, _trig07 e _trig09 para a sensibilidade
│   ├── codigos_docentes.csv          # código D01–D48 usado no artigo ↔ id Lattes (público)
│   ├── conferencia_arestas_DD.csv    # evidência das 232 arestas entre docentes
│   └── lista_docentes_unifei.csv
├── resultados/                   # saídas do notebook 04 (tabelas, JSON e figuras)
└── v1_unifei/                    # estudo anterior (UNIFEI inteira)
```

## Dados

- **Fonte principal:** Currículo Lattes dos docentes, salvo em HTML em 03/11/2026 (47 currículos; um docente sem
  Lattes entra pela lista do SIGAA). Os HTMLs não são publicados aqui; para refazer a coleta, salve a página pública
  de cada currículo com "Salvar como… página completa" e compacte os `.html` em `lattes.zip`.
- **Fonte complementar:** API do OpenAlex (chave gratuita, informada no notebook 02 por `getpass`), consultada em
  04/11/2026. As respostas brutas ficam em `saida_openalex/respostas_brutas` ao rodar o notebook 02.
- **Privacidade:** os dados são públicos e profissionais. Colaboradores externos (discentes e coautores) aparecem só
  como pseudônimos `X00001…`; o mapa pseudônimo → nome (`mapa_externos_PRIVADO.csv`) é gerado pelo notebook 03 e
  **não** é versionado. No artigo, os docentes são identificados pelos códigos D01–D48.

### Dicionário de dados (`dados/rede/`)

| Arquivo | Atributo | Descrição | Tipo / faixa |
|---|---|---|---|
| `nos.csv` | `id` | id Lattes (docente) ou pseudônimo (externo) | texto |
| | `rotulo` | código do docente ou pseudônimo do externo | D01–D48; X00001–X02065 |
| | `tipo_no` | docente do IMC ou colaborador externo | {D, X} |
| | `camada` | área do docente | {Matemática, Computação, Externo} |
| | `categoria` | tipo de colaborador | docente_imc ou 5 categorias de externo |
| | `instituto`, `campus` | unidade da UNIFEI, quando identificada | 8 institutos; Itajubá ou Itabira |
| | `isolado` | sem nenhuma aresta | booleano |
| `arestas.csv` | `source`, `target` | par de nós (ao menos um docente) | texto |
| | `n_eventos` | eventos distintos do par | inteiro ≥ 1 |
| | `n_<tipo>` (8 colunas) | cooperações por tipo; coautoria fracionária 1/(a−1) | real ≥ 0 |
| | `M1`, `M2`, `M3` | intensidades (Seção 3.3 do artigo) | M1 > 0; M2 e M3 em (0, 1] |

Variantes: `_estrito` (sem casamentos só por chave canônica), `_sem_massivos` (sem trabalhos com mais de 10 autores).

## Como executar

**No Google Colab:** abra os notebooks em ordem e siga as células. Cada notebook pede por upload as saídas do anterior
(`saida_lattes.zip`, `saida_openalex.zip`, `redes.zip`).

**Localmente** (Python 3.11+):

```
pip install -r requirements.txt
python scripts/conferir_arestas_docentes.py --lattes saida_lattes --redes redes --saida conferencia_arestas_DD.csv
```

As funções de cada etapa estão em `src/` (`extrator_lattes.executar`, `integracao.executar_integracao`,
`analises.executar_analises`). Sementes fixas: permutações e réplicas usam `seed=0, 1, …`.

## Ambiente dos resultados publicados

A rede em `dados/rede/` foi gerada pelos notebooks 01 e 03 (com a saída do notebook 02 de 04/11/2026) e as análises pelo notebook 04, com 1.000 réplicas do nulo de intermediação. Os resultados em `resultados/` foram gerados em Linux x86-64 com Python 3.13 e as versões de `requirements.txt`;
`resultados/resultados.json` registra o ambiente exato (chave `ambiente`).

## Licença

Código sob a licença MIT (arquivo `LICENSE`); dados derivados em `dados/` e `resultados/` sob CC BY 4.0.

## Autores

Danilo Alexandre Pereira, Lara Bruna Alves, Paulo Victor Olympio e Vinícius de Sousa Viana — XMCO09, Prof. Rafael Frinhani.
