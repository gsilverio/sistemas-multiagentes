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
3. Defina `REPO_URL` com a URL GitHub real e `BRANCH` (padrão `main`) na célula de clone. Repositórios privados exigem autenticação; **não grave credenciais no notebook**. Para testar um repositório privado sem configurar autenticação, use o modo `ORIGEM="zip"`.
4. Instale as dependências.
5. Instale e inicie o Ollama dentro do Colab.
6. Baixe `gemma3:12b` (~8,2 GB).
7. Verifique `ollama ps` para checar a utilização da GPU.
8. Siga as células seguintes para extrair as regras, gerar explicações em lote e montar os índices no Drive (`Meu Drive > gobr-agent > outputs`).

### Geração das explicações (etapa 2)
- Etapa 10 do notebook: gera 5 explicações de teste. Confira na etapa 11.
- Etapa 12: `EXECUTAR_MAIS = True` e rode quantas vezes precisar (200 regras por vez; `LOTE = 0` faz tudo). O que já foi feito é pulado.
- Etapa 12b: confere todas as explicações com o guard e compara com a linha de base antiga.
- No fim, baixe `explicacoes.jsonl` da pasta do Drive indicada na etapa 8 e coloque no projeto em `data/explicacoes/explicacoes_v2.jsonl`.

**Importante**: nunca faça `git push` no Colab como fluxo principal. Faça alterações no Windows, envie para o GitHub, **reinicie a sessão do Colab** e execute novamente a célula de clone. O conteúdo de `/content` é temporário, enquanto o Drive é persistente.

## 5. Limites conhecidos
- O plano gratuito do Colab não garante GPU, disponibilidade de T4 ou duração da sessão; consulte a FAQ oficial: https://research.google.com/colaboratory/intl/pt-BR/faq.html.
- A instalação e o download do modelo terão de ser refeitos quando a máquina virtual do Colab for substituída.
- `aprovada` nas explicações significa que o guard não encontrou omissões nem invenções (nomes, números, condições, `nao agendar`, `ou`). É uma rede de segurança, não prova de correção. O método `rascunho` indica que o modelo errou duas vezes e o texto determinístico foi usado.

No Windows, `python -m unittest discover -s tests -v` executa as verificações estruturais **sem GPU e sem Ollama**.
