# Revisão das explicações GOBR antes da indexação

## Alterações

- Prompt mais restritivo: não traduzir `sem venda` como "nunca comprou", não descrever `nao agendar` como um estado do cliente, preservar nomes literais e operadores lógicos.
- Interpretador: público com escopos combinados como a **mesma seleção de clientes**; correção da redação de critérios frequentes.
- `ExplanationGuard`: compara a saída textual do agente com a estrutura determinística extraída da GOBR. Verifica nomes de escopos, operadores, termos de condições, frequência, não agendamento e modificadores. Essas verificações são **lexicais e estruturais**, não prova de correção semântica.
- Se um requisito não for preservado, o texto principal passa a ser uma explicação conservadora derivada da estrutura; a saída do agente é preservada em `revisao.rascunho_llm` e os motivos são informados em `revisao.problemas`.
- `pending_review` significa "sem divergência detectada automaticamente, aguarda leitura humana". `needs_review` significa "houve divergência detectada ou informação insuficiente". **Nenhum status é aprovação automática para o RAG**.

## O que foi detectado na execução anterior enviada pelo usuário

- `01_frequencia_simples`: `empresa toda` virou `empresa`.
- `02_clientes_sem_venda`: mesma perda do escopo; `sem venda` virou `nunca realizou compra`.
- `03_nao_agendar_inadimplentes`: a explicação sugeria comportamento sobre agendamentos que já existem.
- `04_preferencia_dias_uteis`: sem divergência detectada por esse verificador.
- `05_nao_agendar_filiais`: `nao agendar` virou `não pode estar agendado`.
- `06_multiplos_escopos`: a frase sugeria dois grupos separados em vez da interseção de segmentação e carteira.

## Como verificar

```powershell
python -m unittest discover -s tests -v
```

Esses testes funcionam sem GPU e sem Ollama. Para verificar o comportamento do Gemma 12B, publique o código no GitHub e execute as etapas 7 e 9 do notebook atualizado no Colab. Os resultados serão gravados no Drive em `gobr-agent/outputs/first_delivery_colab_v2`.

## Limites e próximos passos

- Regras com condições não suportadas pelo parser e ambiguidades do domínio precisam de revisão adicional.
- A revisão lexical não consegue detectar todas as contradições e suposições semânticas. Nada deve ser indexado como verdadeiro até a revisão do conteúdo.
- Esta mudança não implementa extração de múltiplos blocos `parametros`, embeddings ou RAG.
