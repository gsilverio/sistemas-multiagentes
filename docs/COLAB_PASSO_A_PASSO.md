# Guia para iniciantes: Windows, GitHub e Google Colab

## Objetivo
A única fonte de código é o repositório GitHub. No Windows editamos Python; o notebook `notebooks/gobr_colab.ipynb` clona esse repositório e executa o pipeline em GPU (quando disponível). Os resultados vão para o Google Drive.

## 1. Prepare o projeto no Windows
1. Extraia esta codebase para uma pasta local, por exemplo `D:\repos\gobr-agent`.
2. Abra a pasta no VS Code (Arquivo > Abrir Pasta). Certifique-se de estar na **raiz**, onde aparecem `src`, `data`, `notebooks` e `requirements.txt`.
3. Confira que `src/llm_config.py` usa `GOBR_LLM_MODEL` e `OLLAMA_BASE_URL`. A configuração padrão continua permitindo execução local.
4. Git (se ainda não instalado): https://git-scm.com/downloads/win

## 2. Suba a mesma pasta ao GitHub
Crie um repositório **vazio**, sem README inicial, no https://github.com/new. No terminal do VS Code, dentro da raiz do projeto:

```powershell
git init
git add .
git commit -m "Prepara pipeline GOBR para execucao no Colab"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin main
```

Se a pasta já é um repositório Git, **não repita** `git init` nem `git remote add`; confira com `git status` e `git remote -v` e faça somente os commits/push necessários. A autenticação no GitHub pode abrir o navegador ou pedir um token/gerenciador de credenciais. Não inclua senhas, tokens, `.env` ou arquivos pessoais em commits.

## 3. Abra o notebook
- Em https://colab.research.google.com clique em `Arquivo > Fazer upload de notebook` e selecione `notebooks/gobr_colab.ipynb` (ou abra o arquivo no próprio GitHub pela integração do Colab).
- Se quiser preservar anotações feitas no notebook, use `Arquivo > Salvar uma cópia no Drive`; o notebook versionado deve continuar no repositório.
- Menu `Ambiente de execução > Alterar tipo de ambiente de execução > Acelerador de hardware`: escolha GPU, quando disponível.

## 4. Execute as células, na ordem
1. Confira a GPU via `nvidia-smi`.
2. Monte o Drive e conceda acesso.
3. Defina `REPO_URL` com a URL GitHub real na célula de clone. Repositórios privados exigem autenticação; **não grave credenciais no notebook**. Para testar um repositório privado sem configurar autenticação, use o modo `ORIGEM="zip"`.
4. Instale as dependências.
5. Instale e inicie o Ollama dentro do Colab.
6. Baixe `gemma3:12b` (~8,2 GB).
7. Execute a primeira regra e confira o Drive (`Meu Drive > gobr-agent > outputs > first_delivery_colab`).
8. Verifique `ollama ps` para checar a utilização da GPU.
9. (Opcional) libere a execução dos seis exemplos.

**Importante**: nunca faça `git push` no Colab como fluxo principal. Faça alterações no Windows, envie para o GitHub, depois execute novamente a célula de clone no Colab. O conteúdo de `/content` é temporário, enquanto o Drive é persistente.

## 5. Envio da entrega
O link do repositório contém os arquivos de código e o notebook. A execução no Colab não é necessária para avaliar o GitHub, mas quem executar o notebook precisará da própria GPU/sessão e acesso ao Ollama para repetir as saídas. Os resultados da primeira entrega que já estavam versionados continuam disponíveis em `outputs/first_delivery/`.

## 6. Limites conhecidos
- O plano gratuito do Colab não garante GPU, disponibilidade de T4 ou duração da sessão; consulte a FAQ oficial: https://research.google.com/colaboratory/intl/pt-BR/faq.html.
- A instalação e o download do modelo terão de ser refeitos quando a máquina virtual do Colab for substituída.
- O trabalho ainda **não implementa indexação ou RAG**: esta adaptação prepara o ambiente para a segunda entrega.
- Alguns `.gobr` de `data/referencias/` contêm mais de um bloco `parametros`. O parser atual foi validado com os exemplos da primeira entrega, **não para processamento automático dos 246 arquivos como uma regra por arquivo**. Primeiro devemos extrair cada bloco e tratar `desconhecidos`/modelagem antes de indexar.
- `src/graph/explanation_graph.py` tem divergência de tipos herdada, não é invocado no notebook desta fase.


## Atualização da revisão das explicações (v2)
1. Faça `git add .`, `git commit -m "Revisa explicacoes GOBR"` e `git push` no Windows.
2. No Colab, abra a versão atualizada de `notebooks/gobr_colab.ipynb` pelo GitHub ou carregue o arquivo novo. **Reinicie a sessão** antes de clonar novamente o projeto para evitar que Python use módulos antigos já importados.
3. Execute as células na sequência, até os seis exemplos da etapa 9; o destino atualizado é `Meu Drive/gobr-agent/outputs/first_delivery_colab_v2`.
4. Na etapa 10, salve os metadados referentes a esta execução.
5. Verifique o objeto `revisao` em cada `explanation.json` e em `summary.json`. O status `pending_review` significa que as invariantes passaram, **não que a explicação foi aprovada para o RAG**. `needs_review` indica problemas detectados e uso de fallback, e contém `rascunho_llm` e `problemas`.

No Windows, rode `python -m unittest discover -s tests -v` para executar verificações estruturais **sem GPU, sem Ollama e sem accuracy**. A validação estrutural é conservadora: não garante interpretação semântica correta.
