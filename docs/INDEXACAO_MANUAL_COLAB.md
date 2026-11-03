# Indexação interna do manual GOBR no Colab

O manual é uma fonte de contexto do agente, separada do índice das explicações. Nesta primeira versão ele não responde diretamente ao usuário: a consulta do usuário recupera trechos do manual para orientar a geração da regra GOBR, que continua passando pelo `GobrParser` e pelas validações determinísticas.

O texto original permanece preservado em `documents.json`. Para gerar embeddings e buscar, o indexador usa uma cópia normalizada: remove acentos, pontuação decorativa e eventuais caracteres de substituição `�`. Portanto, `frequência` e `frequencia` recuperam os mesmos trechos sem degradar o texto que poderá ser exibido ou citado depois.

Depois de instalar `requirements.txt` e clonar o projeto no Colab:

```python
MANUAL_PDF = PROJETO / 'data' / 'manual' / 'govendas-rules.pdf'
MANUAL_INDEX_DIR = Path('/content/drive/MyDrive/gobr-agent/outputs/manual_index')

subprocess.run([
    sys.executable, '-m', 'src.index_manual',
    '--pdf', str(MANUAL_PDF),
    '--output-dir', str(MANUAL_INDEX_DIR),
], cwd=str(PROJETO), check=True)
```

O resultado contém `manual.faiss`, `documents.json` e `manifest.json`. O teste de recuperação é feito sem chamar o Gemma:

```python
from src.index_manual import search_index

for consulta in ['o que significa frequência', 'como declarar não agendar', 'filtros temporais']:
    print('\nCONSULTA:', consulta)
    for resultado in search_index(MANUAL_INDEX_DIR, consulta, k=3):
        meta = resultado['metadata']
        print(round(resultado['score'], 3), 'página', meta['pagina'], resultado['text'][:300])
```

O campo `pagina` permite citar o manual posteriormente, se for criada uma rota específica para perguntas diretas. Por enquanto, o índice é apenas apoio interno do agente.
