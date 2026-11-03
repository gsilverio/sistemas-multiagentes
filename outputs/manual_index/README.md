# Índice vetorial do manual GOBR

Este diretório contém o artefato gerado a partir de `data/manual/govendas-rules.pdf`.

## Arquivos

- `manual.faiss`: índice vetorial FAISS. Contém os vetores dos trechos e permite buscar os trechos mais próximos de uma consulta.
- `documents.json`: relação entre a posição do vetor e o texto original, página, trecho e título da página.
- `manifest.json`: metadados da execução, incluindo quantidade de trechos, páginas, dimensão dos vetores, modelo de embeddings e dispositivo utilizado.

O arquivo `.faiss` não é um banco relacional com tabelas. Ele é uma estrutura especializada para busca de vizinhos mais próximos em vetores. O texto e os metadados ficam no `documents.json` porque o FAISS armazena os vetores, não o conteúdo textual completo.

## Reprodução

```powershell
python -m src.index_manual `
  --pdf data/manual/govendas-rules.pdf `
  --output-dir outputs/manual_index
```

O índice versionado nesta pasta foi gerado com `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, possui 64 trechos e foi construído usando CUDA no Colab.
