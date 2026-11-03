# Indexação rápida das explicações no Colab

Depois de clonar o commit atual e instalar `requirements.txt` no Colab:

1. Execute `src.prepare_references`.
2. Rode `src.batch_reference_explanations` em lotes pequenos no Drive até gerar os itens desejados.
3. Rode `src.index_explanations` apontando para a pasta `items` e para uma pasta persistente do Drive.

Exemplo:

```python
subprocess.run([
    sys.executable, '-m', 'src.index_explanations',
    '--items-dir', str(DESTINO_LOTE / 'items'),
    '--output-dir', str(DESTINO_LOTE / 'index_explanations'),
    '--include-pending',
], cwd=str(PROJETO), check=True)
```

`--include-pending` é o caminho rápido para a entrega: inclui explicações concluídas que ainda têm `indexable: false`, mas preserva o status em `documents.json` e `manifest.json` como `provisional`. Depois da revisão, remova a opção e gere novamente o índice apenas com itens aprovados.

O índice contém `explanations.faiss`, `documents.json` e `manifest.json`. O modelo padrão é `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; ele é carregado em CUDA quando a GPU do Colab está disponível. O teste mínimo é recuperar resultados para consultas sobre frequência, clientes sem venda e não agendamento.
