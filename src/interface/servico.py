"""Serviço de conversa: o grafo + memória persistente, sem depender do Gradio.

- **Memória**: o checkpointer do LangGraph guarda o estado de cada conversa
  (`thread_id`) num arquivo SQLite (`SqliteSaver`). Fechar a página ou
  reiniciar o processo não perde nada: a conversa continua de onde parou,
  inclusive se estava pausada numa pergunta.
- **Retomada de perguntas**: `enviar()` olha se a conversa está pausada num
  `interrupt()`; se estiver, a mensagem é a resposta (`Command(resume=...)`),
  senão é uma mensagem nova (descrição ou pedido de ajuste).
- **Progresso**: `enviar()` é um gerador de `Evento`s, emitidos à medida que
  cada agente termina (`stream_mode="updates"`), para a interface mostrar o
  andamento de execuções que levam minutos.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.agents.percepcao import PercepcaoFalhou, ler_anexo
from src.grafo.estado import Agentes
from src.grafo.grafo import config_thread, construir_grafo, entrada


@dataclass
class Evento:
    tipo: Literal["progresso", "pergunta", "resposta", "erro"]
    texto: str


def abrir_checkpointer(banco: str | Path | None):
    """SQLite em `banco` (criado se não existir); `None` = só em memória."""
    if banco is None:
        return InMemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver

    caminho = Path(banco)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(caminho), check_same_thread=False))


def novo_id() -> str:
    return uuid.uuid4().hex[:12]


def _descrever(no: str, update: dict[str, Any]) -> str | None:
    """Linha de progresso para o fim de cada agente."""
    passo = (update.get("trace") or [{}])[0]
    if update.get("erro"):
        return f"❌ {no}: {update['erro']}"
    if no == "percepcao":
        return f"👁️ Percepção: {passo.get('segmentos', 0)} grupo(s) de clientes identificados."
    if no == "analista":
        return (f"🧐 Analista: {passo.get('achados', 0)} ponto(s) de atenção, {passo.get('perguntas', 0)} "
                f"pergunta(s), {passo.get('ajustes', 0)} ajuste(s) automático(s).")
    if no == "perguntar_usuario":
        return "💬 Respostas recebidas." if passo.get("respondeu") else "💬 Seguindo com as suposições."
    if no == "recuperador":
        return f"📚 Recuperador: {passo.get('exemplos', 0)} exemplo(s) do acervo e trechos do manual."
    if no == "gerador":
        return f"✍️ Gerador (tentativa {passo.get('tentativa')}): {passo.get('refeitos', 0)} bloco(s) escrito(s)."
    if no == "validador":
        if passo.get("ok"):
            return "✅ Validador: GOBR válido e conferido."
        return (f"🔁 Validador: {passo.get('blocos_com_erro', 0)} bloco(s) com erro"
                f"{' e erros gerais' if passo.get('erros_gerais') else ''}.")
    if no == "explicador":
        return f"🗣️ Explicador: {passo.get('explicados', 0)} regra(s) explicadas."
    return None


class ServicoConversa:
    def __init__(self, agentes: Agentes, *, banco: str | Path | None = None, checkpointer: Any = None):
        self.checkpointer = checkpointer if checkpointer is not None else abrir_checkpointer(banco)
        self.grafo = construir_grafo(agentes, checkpointer=self.checkpointer)

    def fechar(self) -> None:
        """Fecha a conexão com o SQLite (no Windows, o arquivo fica preso enquanto ela existir)."""
        conexao = getattr(self.checkpointer, "conn", None)
        if conexao is not None:
            conexao.close()

    # ---------- leitura do estado ----------
    def _snapshot(self, thread_id: str):
        return self.grafo.get_state(config_thread(thread_id))

    def estado(self, thread_id: str) -> dict[str, Any]:
        return dict(self._snapshot(thread_id).values or {})

    def pausada(self, thread_id: str) -> dict[str, Any] | None:
        """Valor do `interrupt()` pendente (perguntas), ou None."""
        interrupcoes = self._snapshot(thread_id).interrupts
        return interrupcoes[0].value if interrupcoes else None

    def historico(self, thread_id: str) -> list[dict[str, Any]]:
        conversa = list(self.estado(thread_id).get("conversa") or [])
        pendente = self.pausada(thread_id)
        if pendente:
            conversa.append({"papel": "agente", "texto": pendente["mensagem"], "pergunta": True})
        return conversa

    def gobr(self, thread_id: str) -> str | None:
        return self.estado(thread_id).get("gobr")

    def decisoes(self, thread_id: str) -> list[dict[str, Any]]:
        """Decisões do orquestrador desde a última mensagem nova."""
        trace = self.estado(thread_id).get("trace") or []
        inicio = max((i for i, p in enumerate(trace) if p.get("no") == "inicio" and p.get("nova_mensagem")),
                     default=0)
        return [p for p in trace[inicio:] if p.get("no") == "orquestrador"]

    def conversas(self) -> list[str]:
        """`thread_id`s salvos, da mais recente para a mais antiga."""
        vistos: dict[str, None] = {}
        for checkpoint in self.checkpointer.list(None):
            vistos.setdefault(checkpoint.config["configurable"]["thread_id"], None)
        return list(vistos)

    # ---------- envio ----------
    def enviar(self, thread_id: str, texto: str = "", arquivos: Iterable[str | Path] = ()) -> Iterator[Evento]:
        arquivos = [Path(a) for a in arquivos]
        textos_anexos, imagens = [], []
        try:
            for arquivo in arquivos:
                conteudo, imgs = ler_anexo(arquivo)
                if conteudo:
                    textos_anexos.append(conteudo)
                imagens.extend(imgs)
        except (PercepcaoFalhou, OSError, UnicodeDecodeError) as exc:
            yield Evento("erro", f"Não consegui ler o anexo: {exc}")
            return

        pendente = self.pausada(thread_id)
        if pendente:
            if imagens:
                yield Evento("progresso", "⚠️ Imagens enviadas como resposta a perguntas são ignoradas; "
                                          "descreva a resposta em texto.")
            comando: Any = Command(resume="\n\n".join(t for t in [texto.strip(), *textos_anexos] if t))
        else:
            completo = "\n\n".join(t for t in [texto.strip(), *textos_anexos] if t)
            if not completo and not imagens:
                yield Evento("erro", "Escreva as regras comerciais ou anexe uma imagem/PDF da tabela.")
                return
            comando = entrada(completo, imagens, exibir=texto.strip() or "(regras no anexo)",
                              anexos=[a.name for a in arquivos])

        for pedaco in self.grafo.stream(comando, config_thread(thread_id), stream_mode="updates"):
            for no, update in pedaco.items():
                if no.startswith("__") or not isinstance(update, dict):
                    continue
                linha = _descrever(no, update)
                if linha:
                    yield Evento("progresso", linha)

        pendente = self.pausada(thread_id)
        if pendente:
            yield Evento("pergunta", pendente["mensagem"])
            return
        estado = self.estado(thread_id)
        yield Evento("erro" if estado.get("erro") else "resposta", estado.get("resposta") or "")
