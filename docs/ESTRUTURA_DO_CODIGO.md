# Estrutura do código

Guia para entender onde está cada coisa e como as peças se conectam. A visão
de arquitetura (agentes, grafo e etapas) está em [`ARQUITETURA.md`](ARQUITETURA.md).

## Mapa rápido

```
src/
├── gobr/                  NÚCLEO DA LINGUAGEM — determinístico, sem LLM
│   ├── grammar.py         padrões sintáticos extraídos do manual (regex)
│   ├── ast.py             modelos Pydantic de um bloco `parametros` analisado
│   ├── parser.py          texto de um bloco `parametros` → GobrParsedRule
│   ├── extractor.py       arquivo .gobr → lista de blocos + definições usadas
│   ├── vocabulary.py      vocabulário de condições `quando` (acervo + manual)
│   ├── assinatura.py      chaves de agrupamento: mesmo corpo / mesma forma de regra
│   └── scope_text.py      escopos da AST → texto em português
├── validation/            VERIFICAÇÕES — determinísticas, sem LLM
│   └── gobr_validator.py  arquivo GOBR inteiro → relatório de erros/avisos
├── explicacao/            EXPLICAÇÃO DETERMINÍSTICA — sem LLM
│   ├── glossario.py       cada construção GOBR → português (manual cap. 5 a 7)
│   ├── rascunho.py        AST → fatos + texto completo
│   └── guarda.py          um texto preserva todos os fatos, e só eles?
├── agents/                AGENTES (um por nó do grafo)
│   ├── contratos.py       RuleSpec, Segmento, Analise, Conferencia... (Pydantic)
│   ├── percepcao.py       texto / imagem / PDF → RuleSpec (LLM com visão)
│   ├── analista.py        RuleSpec → falhas de modelagem + perguntas (sem LLM)
│   ├── recuperador.py     segmento → exemplos + manual (busca híbrida)
│   ├── gerador.py         segmento + contexto → bloco parametros (LLM)
│   ├── validador.py       GOBR + RuleSpec → erros por bloco (sem LLM)
│   ├── explicador.py      GOBR → rascunho → 1 chamada LLM → guard
│   └── comum.py           texto da resposta do LLM, extração de JSON
├── grafo/                 ORQUESTRAÇÃO (LangGraph)
│   ├── estado.py          Estado do grafo + Agentes (dependências)
│   ├── orquestrador.py    guardrails (destinos permitidos) + decisão do LLM
│   ├── grafo.py           construir_grafo(): nós, arestas, interrupt
│   ├── relatorio.py       resposta final em markdown
│   └── executar.py        linha de comando
├── interface/             ENTREGA FINAL (etapa 5)
│   ├── servico.py         ServicoConversa: grafo + memória SQLite + eventos de progresso
│   └── app.py             interface Gradio (chat, anexos, .gobr, decisões, conversas salvas)
├── prompts/               prompts dos agentes com LLM
│   ├── percepcao_prompt.py
│   ├── gerador_prompt.py
│   ├── orquestrador_prompt.py
│   └── explicador_prompt.py
├── rag/                   RECUPERAÇÃO HÍBRIDA (BM25 + embeddings, RRF)
│   ├── busca_hibrida.py   BM25, embeddings, fusão RRF, índice persistido
│   ├── manual.py          PDF do manual → um documento por seção numerada
│   ├── exemplos.py        acervo + explicações → um documento por corpo de regra
│   └── recuperador.py     buscar_manual() / buscar_exemplos() (API dos agentes)
├── pipelines/             SCRIPTS DE LOTE (python -m ...)
│   ├── prepare_references.py  acervo → regras_extraidas.jsonl + diagnóstico
│   ├── build_vocabulary.py    acervo → data/vocabulario/condicoes.json
│   ├── generate_explanations.py  regras → explicações (LLM no Colab; --sem-llm local)
│   ├── avaliar_explicacoes.py    JSONL de explicações → relatório do guard
│   ├── construir_indices.py      manual + acervo → outputs/rag/{manual,exemplos}
│   └── avaliar_recuperacao.py    consultas de referência → acerto@k e MRR
└── llm_config.py          cria o ChatOllama (modelo, URL, contexto)

data/
├── referencias/   246 arquivos .gobr reais (1.700 blocos `parametros`)
├── exemplos/      3 regras comerciais em PDF + o GOBR correspondente
├── manual/        govendas-rules.pdf (manual da linguagem)
├── vocabulario/   condicoes.json (gerado; versionado)
└── avaliacao/     linha de base das explicações + consultas de avaliação do RAG
tests/             testes unitários (rodam sem LLM e sem GPU)
outputs/rag/       índices de recuperação gerados (versionados)
notebooks/gobr_colab.ipynb  execução das partes pesadas no Colab
```

## O que roda onde

| Parte | Local (Windows) | Colab |
|---|---|---|
| `src/gobr/*`, `src/validation/*`, testes | ✅ | ✅ |
| `prepare_references`, `build_vocabulary`, `avaliar_explicacoes` | ✅ | ✅ |
| `src/explicacao/*`, `generate_explanations --sem-llm` | ✅ (~10 s para o acervo) | ✅ |
| Índices e busca (`src/rag/*`, `construir_indices`) | ✅ (~2 min em CPU com embeddings; BM25 instantâneo) | ✅ |
| Grafo e interface com LLM falso (testes) | ✅ | ✅ |
| Agentes com LLM, grafo de verdade, `generate_explanations` | ❌ (pouca VRAM) | ✅ |

## Conceitos do domínio

Um arquivo GOBR tem blocos, sempre nesta ordem:

```
version 1.0
time "televendas":                      ← grupo de vendedores
    vendedores todostelevendas ativados
carteira "carteira televendas":         ← grupo de clientes do time
    clientes carteirafixa("televendas")
parametros "Ativos":                    ← como atender um subconjunto
    da carteira("carteira televendas")  ← escopo
    quando recencia de positivacao em 60 dias   ← condição
        e adimplentes                            ← condição (continuação)
    frequencia de 14 dias               ← ação: frequência
    preferencia nos dias uteis          ← modificador
```

`time`, `carteira` e `segmentacao` são **definições**; `parametros` são as
**regras** que o agente precisa gerar. Cada bloco `parametros` vira um
`GobrParsedRule` (`src/gobr/ast.py`):

| Campo | Exemplo | Origem no texto |
|---|---|---|
| `nome` | `Ativos` | `parametros "Ativos":` |
| `escopos` | `[ParsedScope(tipo="carteira", valores=["carteira televendas"])]` | `da carteira(...)`, `do time(...)`, `da segmentacao(...)` |
| `condicoes` | `[recencia de positivacao em 60 dias, e adimplentes]` | `quando ...` e continuações `e ...` / `ou ...` |
| `frequencia` | `14 dias` | `frequencia de 14 dias` |
| `nao_agendar` | `False` | `nao agendar` |
| `modificadores` | `["preferencia nos dias uteis"]` | `preferencia`, `fixo`, `roteirizavel`, `sem choque`, `restrito` |
| `filtros_temporais` | `["nos dias uteis"]` | quando a regra **executa** (manual, cap. 5) |
| `desconhecidos` | `[]` | qualquer linha não reconhecida (nunca é descartada em silêncio) |

## Os módulos em detalhe

### `src/gobr/grammar.py`
Padrões tirados do manual, cada um com o capítulo de origem: estrutura do
arquivo (`VERSION`, `CABECALHO`, `ORDEM_CATEGORIAS`), corpo de `time`
(`TIME_VENDEDORES`), corpo de `carteira`/`segmentacao` (`CARTEIRA_CLIENTES`),
filtros temporais (`FILTRO_TEMPORAL`), frequência (`frequencia_valida`) e
modificadores (`modificador_valido`). Também define `normalizar()`, que
põe o texto em minúsculas, sem acentos e com espaços colapsados, e é usada
em todo o projeto para comparar texto. Não interpreta nada: só responde se
um trecho tem a forma certa.

### `src/gobr/parser.py` — `GobrParser.parse(texto) -> GobrParsedRule`
Lê **um** bloco `parametros`. Junta linhas com parênteses abertos
(`_linhas_logicas`), classifica cada linha (escopo, condição, filtro temporal,
frequência, `nao agendar`, modificador) e trata continuações em várias linhas:
`e adimplentes` depois de `quando`, `e no sabado` depois de `preferencia`,
`ou no maximo de 30 dias` depois de `frequencia`. O que não reconhece vai
para `desconhecidos`. Os 1.700 blocos do acervo são lidos sem nenhum
desconhecido (há um teste para isso).

### `src/gobr/extractor.py` — `GobrRuleExtractor.extract_file(caminho)`
Divide um arquivo em blocos e, para cada `parametros`, anexa as definições
que ele usa, inclusive as indiretas (`parametros → carteira → time`). Cada
bloco vira um `GobrRuleBlock` com id `regra_001_003` (arquivo + posição),
linhas de início/fim e o contexto. É a base do JSONL de referências.

### `src/gobr/vocabulary.py` — `VocabularioCondicoes`
Sabe quais condições `quando` existem. `modelo_condicao()` troca números por
`N` e listas entre aspas por `("X")`, de modo que
`recencia de positivacao em 60 dias` e `... em 90 dias` viram o mesmo modelo.
Fontes: o acervo (`data/vocabulario/condicoes.json`, 127 modelos) e os padrões
do capítulo 6 do manual (`PADROES_MANUAL`, que sozinhos cobrem 122 dos 127
modelos do acervo). `conhece()` aceita condições compostas com `e`/`ou`;
`sugestoes()` devolve exemplos reais parecidos.

### `src/validation/gobr_validator.py` — `GobrValidator.validar(texto)`
Recebe um arquivo GOBR completo e devolve um `RelatorioValidacao` com
`ok`, `erros`, `avisos` e `como_texto()` (formato pensado para devolver ao
LLM corrigir). Códigos:

| Código | Severidade | Problema |
|---|---|---|
| E001 | erro | falta `version 1.0` na primeira linha (ou repetido) |
| E002 | erro | linha fora de bloco / sem indentação |
| E003 | erro | alertas/tags misturados com parametros |
| E004 | erro | bloco sem corpo |
| E005 | erro | categorias fora da ordem time → segmentacao → carteira → parametros |
| E006 | erro | nome de bloco repetido na mesma categoria |
| E007 | erro | nenhum `time` definido |
| E008 | erro | parênteses ou aspas desbalanceados |
| E010 / E011 | erro | linha inválida no corpo de `time` / `carteira`/`segmentacao` |
| E012 | erro | carteira com mais de uma segmentação como fonte |
| E020 | erro | referência a time/carteira/segmentação inexistente (com dica de tipo) |
| E021 | erro | referência a bloco definido depois do uso |
| E030 | erro | cláusula não reconhecida em `parametros` |
| E031 | erro | `parametros` sem escopo |
| E032 | erro | `parametros` sem ação (frequência, `nao agendar` ou modificador) |
| E033 | erro | `nao agendar` junto com frequência/preferência |
| E034 | erro | frequência fora da gramática |
| E101 | erro | caractere de controle invisível |
| A001 | aviso | condição fora do vocabulário (com sugestões) |
| A002 | aviso | modificador fora da gramática conhecida |
| A003 | aviso | categoria não verificada (alertas/tags) |
| A004 | aviso | cabeçalho de bloco indentado |

Linha de comando: `python -m src.validation.gobr_validator arquivo.gobr`
(sai com código 1 se houver erro; `--json` para saída estruturada).

O validador verifica a **linguagem**. Problemas de **negócio** (faixas de
recência com buracos, falta de regra de fallback, conflito entre regras) são
papel do agente Analista (etapa 4).

### `src/explicacao/glossario.py` — tradução de cada construção
Uma função por tipo de trecho: `traduzir_condicao`, `traduzir_frequencia`,
`traduzir_modificador` e `traduzir_filtro`. Cada uma devolve uma `Traducao`
com a `frase` em português, `conhecida` (False quando não há tradução; a
frase então cita o trecho original entre aspas, sem inventar significado) e
`chaves`: expressões regulares que um texto precisa conter para ser fiel a
esse trecho (ex.: `inadimpl|com pendencia` para `inadimplentes`).

Os significados vêm do manual, por exemplo:

| GOBR | Português |
|---|---|
| `recencia de positivacao global em 90 dias` | cuja última compra com qualquer vendedor da empresa ocorreu nos últimos 90 dias |
| `sem venda` | sem venda (que nunca compraram), conforme 6.1 |
| `frequencia de inativos da empresa` | com a frequência média dos clientes inativos da empresa (7.4.3) |
| `frequencia do vendedor ou de 35 dias` | com a frequência do próprio vendedor; se ela não existir, a cada 35 dias |
| `preferencia na segunda e na quarta` | preferência de visita na segunda e na quarta; nos demais dias o cliente não é visitado (7.2) |
| `restrito a partir da segunda semana` | só entra na agenda a partir da segunda semana do mês (7.5) |

Condições compostas (`inativos para o vendedor e inadimplentes`) são
divididas nos `e`/`ou` de fora dos parênteses, mas `entre 30 dias e 60 dias`
não é quebrado: a divisão só acontece quando cada parte tem tradução.
Cobertura: todas as condições do acervo, exceto as duas anomalias
(`regra_032` e `regra_140`).

### `src/explicacao/rascunho.py` — `montar_rascunho(regra) -> Rascunho`
Junta as traduções numa explicação completa e guarda a lista de `Fato`s
(tipo, trecho GOBR, significado, chaves), os nomes que precisam aparecer
literalmente e quantos `ou` lógicos a regra tem. O texto segue o molde
"Os clientes {escopo} {condições} devem ser atendidos {frequência}.", mais
uma frase para filtros temporais e uma por modificador. É a "verdade" da
explicação: o LLM só reescreve isso.

### `src/explicacao/guarda.py` — `verificar(rascunho, texto) -> ResultadoGuarda`
Confere um texto contra o rascunho: nomes literais, todos os números da
regra presentes e nenhum número novo, chaves de cada fato, `nao agendar`
como proibição (e nenhuma proibição inventada), `ou` preservado, nenhuma
menção à implementação e nenhuma justificativa inventada ("o objetivo é...",
"estratégia", "vamos..."; erro visto no primeiro teste com o `gemma3:12b`),
nem texto muito mais longo que o rascunho. O próprio rascunho passa no guard nos 1.698 blocos
traduzíveis do acervo (há um teste para isso). Aplicado às 59 explicações
antigas, aprova só 20 (`python -m src.pipelines.avaliar_explicacoes
data/avaliacao/explicacoes_baseline_v1.jsonl`).

### `src/agents/explicador.py` — `AgenteExplicador.explicar(gobr) -> Explicacao`
1. parser + glossário → rascunho;
2. uma chamada ao LLM para reescrever o rascunho de forma natural
   (prompt em `src/prompts/explicador_prompt.py`, com o GOBR, os fatos e o
   rascunho);
3. guard; se reprovar, uma chamada de correção com os problemas apontados;
4. se reprovar de novo, devolve o próprio rascunho.

`Explicacao.metodo` diz o caminho: `llm`, `llm_corrigido` ou `rascunho`.
Regras com trechos sem tradução não vão ao LLM (ele tenderia a inventar).
O agente recebe só o GOBR (verificação cega: na etapa 4 ele confere o
Gerador sem ver a descrição original). `usar_llm=False` gera só o rascunho.

### `src/rag/` — recuperação híbrida (etapa 3)
Dois índices, ambos em `outputs/rag/<nome>/` (`documents.json`,
`manifest.json`, `vetores.npy`):

- **manual** (`manual.py`): uma entrada por **seção numerada** do manual
  (98 no total, ex.: `6.11.1 Opções para tempo`), não por página. O título
  só é aceito se continuar a numeração (evita que `5 dias` vire seção),
  cabeçalhos e rodapés de página são descartados, a indentação dos exemplos
  é mantida, e o texto começa pelo caminho da seção ("6 Regras para filtros
  de clientes > 6.11 Outros filtros úteis > ..."). Metadados: `secao`,
  `construcao` (time, carteira_segmentacao, filtro_temporal,
  filtro_cliente, parametros, alerta, tag, gramatica) e `paginas`.
- **exemplos** (`exemplos.py`): uma entrada por **corpo distinto** de regra
  do acervo (1.202; os blocos repetidos ficam em `ids_iguais`), com o GOBR,
  as definições de time/carteira usadas e a explicação: a do LLM quando
  aprovada pelo guard, senão o rascunho. O BM25 indexa também o GOBR.
  Metadados `assinatura` (forma da regra, sem números) e `construcoes`
  (`nao_agendar`, `frequencia_dinamica`, `modificador_preferencia`...).

`busca_hibrida.py` implementa o BM25 (sem dependência nova), os embeddings
(`intfloat/multilingual-e5-small`, opcional) e a fusão por Reciprocal Rank
Fusion. Sem sentence-transformers ou sem vetores, a busca cai para BM25 e o
resultado diz `modo="bm25"`.

`recuperador.py` é o que os agentes usam:
- `buscar_manual(consulta, k=4, construcao=None)` → `TrechoManual`
  (com `citacao()` para citar seção e página);
- `buscar_exemplos(consulta, k=3, exigir=(), diversificar=True)` →
  `Exemplo`; `diversificar` não devolve duas regras com a mesma forma.

### `src/agents/` e `src/grafo/` — o grafo multiagente (etapa 4)
Fluxo de uma conversa (detalhes e diagrama em `ARQUITETURA.md`):

1. **Percepção** (`AgentePercepcao.extrair(textos, imagens)`): pede ao LLM um
   JSON no formato do `RuleSpec`; valida com Pydantic e, se vier inválido,
   devolve o erro ao modelo uma vez. `ler_anexo()` lê PDF com texto (pypdf)
   ou imagem (base64 para o modelo com visão).
2. **Analista** (`AgenteAnalista.analisar(spec)`): reordena os segmentos e
   aponta buracos, sobreposições, falta de fallback, segmento sem ação,
   restrições fora do GOBR. Devolve achados, perguntas (bloqueantes ou com
   suposição) e o `RuleSpec` ajustado.
3. **Perguntar ao usuário** (nó com `interrupt()`): a resposta vira uma
   mensagem e a Percepção refaz o `RuleSpec`. Pergunta não bloqueante é
   feita uma vez; sem resposta, vira suposição na resposta final.
4. **Recuperador**: para cada segmento, 3 exemplos do acervo (segmentos
   `nao agendar` só recebem exemplos com `nao agendar`) e 2 trechos do manual.
5. **Gerador** (`AgenteGerador.gerar`): um bloco por segmento, com os
   exemplos como pares pergunta/resposta (`gerador_prompt.pedido()` é o
   formato dos pares de fine-tuning). `limpar_bloco()` tira texto extra,
   força o nome pedido e normaliza a indentação. `montar_arquivo()` junta
   `version 1.0`, o `time` e os blocos.
6. **Validador** (`AgenteValidador.validar(gobr, spec)`): erros do
   `GobrValidator` + conferência pela AST, agrupados por bloco. Se houver
   erro, o gerador refaz só aqueles blocos (até 3 gerações).
7. **Explicador**: o `AgenteExplicador` da etapa 2 em cada bloco válido.
8. **Finalizar** (`relatorio.montar_resposta`): falhas de modelagem, ajustes,
   suposições, GOBR com o status da validação e o que cada regra faz.

`Agentes.padrao()` cria os agentes reais (Ollama, modelos por papel em
`src/llm_config.py`). Os testes (`tests/test_grafo.py`) injetam LLMs falsos
e cobrem o fluxo completo, o laço de correção, o limite de tentativas, a
pausa para perguntas, a escolha do orquestrador LLM, o guardrail e uma
segunda mensagem na mesma conversa.

### `src/interface/` — memória e interface (etapa 5)
`ServicoConversa` junta o grafo e a memória, sem depender do Gradio (é o
que os testes usam):
- `abrir_checkpointer(banco)`: `SqliteSaver` num arquivo; cada conversa é
  um `thread_id`. Reiniciar o processo não perde nada, nem uma conversa
  pausada numa pergunta.
- `enviar(thread_id, texto, arquivos)`: se a conversa está pausada, a
  mensagem é a resposta (`Command(resume=...)`); senão, uma mensagem nova
  (descrição ou pedido de ajuste). É um gerador de `Evento`s: uma linha de
  progresso a cada agente que termina, e no fim a pergunta, a resposta ou o
  erro. PDF e imagem anexados passam por `ler_anexo()`; o histórico mostra
  só o nome do arquivo.
- `historico()`, `gobr()`, `decisoes()` (orquestrador, último ciclo) e
  `conversas()` (ids salvos) alimentam a tela.

O estado do grafo ganhou `conversa`, o histórico exibido (mensagens do
usuário, perguntas com a resposta e respostas finais), gravado pelos
próprios nós.

`app.py` monta a tela: chat com anexos e um bloco recolhível com o andamento
dos agentes, botão *Seguir com as suposições*, arquivo `.gobr` para baixar,
tabela de decisões do orquestrador (regra, LLM ou guardrail) e *Conversas
salvas → Retomar*. `responder()` é a função do chat, testada sem subir o
servidor.

### `src/pipelines/`
Scripts de lote, todos `python -m src.pipelines.<nome>`:
- `prepare_references`: extrai os blocos do acervo para JSONL e gera um
  diagnóstico do parser.
- `build_vocabulary`: regenera `data/vocabulario/condicoes.json`.
- `generate_explanations`: gera explicações com retomada incremental (cada
  regra vira `items/<id>.json`; reexecutar pula o que já foi feito) e
  consolida tudo em `explicacoes.jsonl`. A assinatura de cada item usa o
  modelo e um hash dos arquivos do explicador, então mudar outras partes do
  código não refaz o trabalho. Blocos com o mesmo corpo (só o nome muda)
  reaproveitam a explicação: no acervo são 1.202 chamadas ao LLM em vez de
  1.698. Com Ollama no Colab; `--sem-llm` roda local. Antes de começar,
  confere se o Ollama responde; se a conexão cair no meio (3 falhas
  seguidas), para com código de saída 3 sem gastar tentativas das regras, e
  a etapa 12 do notebook reinicia o Ollama e continua.
- `avaliar_explicacoes`: passa o guard num JSONL de explicações e mostra
  quantas foram aprovadas e o motivo das reprovações.
- `construir_indices`: gera os dois índices de `outputs/rag/`. Usa
  `data/explicacoes/explicacoes_v2.jsonl` se existir (explicações do LLM);
  `--sem-vetores` gera só o BM25.
- `avaliar_recuperacao`: roda as consultas de
  `data/avaliacao/recuperacao.json` (30 do manual, 18 de exemplos, parte
  delas tirada das descrições reais em `data/exemplos/*.pdf`) e compara
  BM25, denso e híbrido (acerto@1, acerto@k, MRR).

## Fluxo de dados atual

```
data/referencias/*.gobr
   │ GobrRuleExtractor
   ▼
blocos parametros + definições ──GobrParser──► GobrParsedRule (AST)
   │                                              │ glossário
   │ build_vocabulary                             ▼
   ▼                                       rascunho (fatos + texto)
data/vocabulario/condicoes.json                   │ AgenteExplicador (LLM, Colab) + guard
   │                                              ▼
   │                                       explicacoes.jsonl ──► construir_indices ──► outputs/rag (etapa 3); fine-tuning (etapa 6)
   │
   ▼
GobrValidator ◄── qualquer GOBR (do usuário ou gerado pelo agente)
```

## Anomalias conhecidas no acervo

O validador encontrou problemas reais em 7 arquivos; o teste
`tests/test_gobr_validator.py` fixa essa lista para detectar regressões.

| Arquivo | Problema |
|---|---|
| `regra_032.gobr` | `ramo de atividade igual("CONS. FINAL"` sem fechar parêntese |
| `regra_047.gobr` | cabeçalho `parametros` com um espaço de recuo |
| `regra_080.gobr`, `regra_083.gobr` | dois blocos `parametros` com o mesmo nome |
| `regra_140.gobr` | caractere invisível `\x01` no fim de uma condição |
| `regra_221.gobr` | aspa sobrando em `preferencia nos dias uteis e no sabado"` |
| `exemplos/V1Vidros-RegraComercial.gobr` | não começa com `version 1.0` |
