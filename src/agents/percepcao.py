"""Agente Percepção: texto, imagem ou PDF com regras comerciais → `RuleSpec`.

Usa um modelo com visão (gemma3:12b) e pede JSON; a resposta é validada
pelo Pydantic e, se vier inválida, o modelo recebe o erro e tenta de novo
uma vez. PDFs com texto são lidos com pypdf (não precisam de visão).
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.agents.comum import extrair_json, texto_da_resposta
from src.agents.contratos import RuleSpec
from src.prompts import percepcao_prompt

EXTENSOES_IMAGEM = {".png", ".jpg", ".jpeg", ".webp"}


class PercepcaoFalhou(RuntimeError):
    pass


def ler_anexo(caminho: Path) -> tuple[str, list[str]]:
    """Arquivo → (texto, imagens em base64)."""
    caminho = Path(caminho)
    sufixo = caminho.suffix.lower()
    if sufixo == ".pdf":
        from pypdf import PdfReader

        texto = "\n".join(p.extract_text() or "" for p in PdfReader(str(caminho)).pages).strip()
        if not texto:
            raise PercepcaoFalhou(f"{caminho.name}: PDF sem texto (escaneado). Envie uma imagem (PNG/JPG) da tabela.")
        return f"Conteúdo do arquivo {caminho.name}:\n{texto}", []
    if sufixo in EXTENSOES_IMAGEM:
        return "", [base64.b64encode(caminho.read_bytes()).decode("ascii")]
    return caminho.read_text(encoding="utf-8"), []


class AgentePercepcao:
    def __init__(self, llm: Any = None, *, max_correcoes: int = 1):
        self._llm = llm
        self.max_correcoes = max_correcoes

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from src.llm_config import create_llm_para

            self._llm = create_llm_para("percepcao")
        return self._llm

    def extrair(self, textos: list[str], imagens: list[str] | None = None) -> RuleSpec:
        mensagens = percepcao_prompt.mensagens(textos, imagens or [])
        erro = ""
        for _ in range(self.max_correcoes + 1):
            resposta = texto_da_resposta(self.llm.invoke(mensagens))
            try:
                return RuleSpec.model_validate(extrair_json(resposta))
            except (ValueError, ValidationError) as exc:
                erro = str(exc)[:800]
                mensagens = [*mensagens, ("ai", resposta),
                             ("human", percepcao_prompt.CORRECAO.format(erro=erro))]
        raise PercepcaoFalhou(f"O modelo não devolveu um RuleSpec válido: {erro}")
