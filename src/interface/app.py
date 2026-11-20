"""Interface Gradio do agente GOBR (etapa 5).

    python -m src.interface.app                      # http://127.0.0.1:7860
    python -m src.interface.app --share              # link público (Colab)
    python -m src.interface.app --banco outputs/conversas.sqlite

Precisa do Ollama (no Colab, ver a etapa 16 do notebook). A lógica fica em
`ServicoConversa`; aqui só a apresentação: chat com anexos, andamento dos
agentes, GOBR para baixar, decisões do orquestrador e conversas salvas.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from src.interface.servico import ServicoConversa, novo_id

RAIZ = Path(__file__).resolve().parents[2]
BANCO_PADRAO = RAIZ / "outputs" / "conversas.sqlite"
TIPOS_ANEXO = [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt"]
DIR_DOWNLOADS = Path(tempfile.gettempdir()) / "gobr-downloads"

APRESENTACAO = """# Agente GOBR
Descreva as regras comerciais de atendimento (ou anexe a imagem/PDF da tabela). O agente aponta
falhas na modelagem, pergunta o que estiver faltando e devolve as regras na DSL **GOBR**, validadas
e explicadas. Depois da resposta, peça ajustes na mesma conversa (ex.: *"mude a frequência dos
ativos para 7 dias"*)."""


def _caminhos(arquivos: list[Any] | None) -> list[str]:
    caminhos = []
    for a in arquivos or []:
        caminhos.append(a.get("path") if isinstance(a, dict) else getattr(a, "path", None) or str(a))
    return [c for c in caminhos if c]


def mensagens_do_historico(historico: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Histórico do serviço → formato `messages` do gr.Chatbot."""
    chat = []
    for item in historico:
        texto = item["texto"]
        if item.get("anexos"):
            texto = f"{texto}\n\n" + " ".join(f"📎 `{a}`" for a in item["anexos"])
        chat.append({"role": "user" if item["papel"] == "usuario" else "assistant", "content": texto})
    return chat


def tabela_decisoes(decisoes: list[dict[str, Any]]) -> str:
    if not decisoes:
        return "_Nenhuma decisão ainda._"
    linhas = ["| # | próximo | origem | motivo |", "|---|---|---|---|"]
    for i, d in enumerate(decisoes, start=1):
        origem = {"llm": "🤖 LLM", "regra": "📏 regra", "guardrail": "🛡️ guardrail"}.get(d.get("origem"), d.get("origem"))
        motivo = str(d.get("motivo", "")).replace("|", "/")
        linhas.append(f"| {i} | `{d.get('proximo')}` | {origem} | {motivo} |")
    return "\n".join(linhas)


def arquivo_gobr(servico: ServicoConversa, thread_id: str) -> str | None:
    gobr = servico.gobr(thread_id)
    if not gobr:
        return None
    DIR_DOWNLOADS.mkdir(parents=True, exist_ok=True)
    caminho = DIR_DOWNLOADS / f"regras_{thread_id}.gobr"
    caminho.write_text(gobr, encoding="utf-8")
    return str(caminho)


def responder(servico: ServicoConversa, mensagem: dict[str, Any] | str | None, chat: list, thread_id: str):
    """Gerador usado pelo Gradio: devolve (chat, caixa, arquivo, decisões, status) a cada evento."""
    if isinstance(mensagem, dict):
        texto, arquivos = mensagem.get("text") or "", _caminhos(mensagem.get("files"))
    else:
        texto, arquivos = mensagem or "", []
    chat = list(chat or [])
    exibido = texto or "(seguir com as suposições)"
    if arquivos:
        exibido += "\n\n" + " ".join(f"📎 `{Path(a).name}`" for a in arquivos)
    chat.append({"role": "user", "content": exibido})
    andamento: list[str] = []
    progresso = {"role": "assistant", "content": "", "metadata": {"title": "⚙️ Agentes trabalhando", "status": "pending"}}
    chat.append(progresso)
    yield chat, None, None, "", "⏳ Processando..."

    final = None
    for evento in servico.enviar(thread_id, texto, arquivos):
        if evento.tipo == "progresso":
            andamento.append(evento.texto)
            progresso["content"] = "\n".join(f"- {l}" for l in andamento)
            yield chat, None, None, "", "⏳ " + evento.texto
        else:
            final = evento
    progresso["metadata"] = {"title": "⚙️ Agentes", "status": "done"}
    if not andamento:
        chat.remove(progresso)
    if final is not None:
        chat.append({"role": "assistant", "content": final.texto})
    status = {"pergunta": "❓ Aguardando sua resposta (ou clique em *Seguir com as suposições*).",
              "resposta": "✅ Concluído. Peça ajustes ou comece uma nova conversa.",
              "erro": "❌ Não foi possível concluir."}.get(final.tipo if final else "erro")
    yield (chat, None, arquivo_gobr(servico, thread_id), tabela_decisoes(servico.decisoes(thread_id)), status)


def criar_app(servico: ServicoConversa, *, exemplos: list[Path] | None = None):
    import gradio as gr

    with gr.Blocks(title="Agente GOBR") as app:
        thread = gr.State(novo_id)
        gr.Markdown(APRESENTACAO)
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(type="messages", height=620, show_copy_button=True, label="Conversa")
                caixa = gr.MultimodalTextbox(file_types=TIPOS_ANEXO, file_count="multiple", show_label=False,
                                             placeholder="Descreva as regras ou anexe a tabela (PDF/imagem)...")
                with gr.Row():
                    seguir = gr.Button("Seguir com as suposições")
                    nova = gr.Button("Nova conversa", variant="secondary")
                if exemplos:
                    gr.Examples([[{"text": f"Regras comerciais da {p.stem}", "files": [str(p)]}] for p in exemplos],
                                inputs=[caixa], label="Exemplos reais")
            with gr.Column(scale=2):
                status = gr.Markdown("Pronto.")
                arquivo = gr.File(label="GOBR gerado (.gobr)")
                with gr.Accordion("Decisões do orquestrador", open=False):
                    decisoes = gr.Markdown("_Nenhuma decisão ainda._")
                with gr.Accordion("Conversas salvas", open=False):
                    lista = gr.Dropdown(label="Conversa", choices=servico.conversas(), allow_custom_value=True)
                    with gr.Row():
                        atualizar = gr.Button("Atualizar lista")
                        retomar = gr.Button("Retomar", variant="primary")
                    id_atual = gr.Markdown()

        saidas = [chat, caixa, arquivo, decisoes, status]

        def enviar(mensagem, historico, thread_id):
            yield from responder(servico, mensagem, historico, thread_id)

        def seguir_suposicoes(historico, thread_id):
            yield from responder(servico, {"text": "", "files": []}, historico, thread_id)

        def nova_conversa():
            novo = novo_id()
            return novo, [], None, "_Nenhuma decisão ainda._", "Nova conversa.", f"Conversa atual: `{novo}`"

        def retomar_conversa(thread_id):
            if not thread_id:
                return gr.update(), gr.update(), gr.update(), gr.update(), "Escolha uma conversa.", gr.update()
            pausa = servico.pausada(thread_id)
            return (thread_id, mensagens_do_historico(servico.historico(thread_id)), arquivo_gobr(servico, thread_id),
                    tabela_decisoes(servico.decisoes(thread_id)),
                    "❓ Aguardando sua resposta." if pausa else "Conversa retomada.", f"Conversa atual: `{thread_id}`")

        caixa.submit(enviar, [caixa, chat, thread], saidas)
        seguir.click(seguir_suposicoes, [chat, thread], saidas)
        nova.click(nova_conversa, None, [thread, chat, arquivo, decisoes, status, id_atual])
        atualizar.click(lambda: gr.update(choices=servico.conversas()), None, lista)
        retomar.click(retomar_conversa, lista, [thread, chat, arquivo, decisoes, status, id_atual])
        app.load(lambda t: f"Conversa atual: `{t}`", thread, id_atual)
    return app


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--banco", type=Path, default=BANCO_PADRAO, help="arquivo SQLite da memória")
    args.add_argument("--share", action="store_true", help="link público do Gradio (necessário no Colab)")
    args.add_argument("--porta", type=int, default=7860)
    args.add_argument("--orquestrador-sem-llm", action="store_true")
    args.add_argument("--explicar-com-llm", action="store_true")
    a = args.parse_args()

    from src.grafo.estado import Agentes

    agentes = Agentes.padrao(orquestrador_com_llm=not a.orquestrador_sem_llm, explicar_com_llm=a.explicar_com_llm)
    servico = ServicoConversa(agentes, banco=a.banco)
    exemplos = sorted((RAIZ / "data" / "exemplos").glob("*.pdf"))
    criar_app(servico, exemplos=exemplos).queue().launch(share=a.share, server_port=a.porta)


if __name__ == "__main__":
    main()
