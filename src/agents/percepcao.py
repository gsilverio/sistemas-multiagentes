"""Agente Percepção: texto, imagem ou PDF com regras comerciais → `RuleSpec`.

Usa um modelo com visão (gemma3:12b) e pede JSON; a resposta é validada
pelo Pydantic e, se vier inválida, o modelo recebe o erro e tenta de novo
uma vez. PDFs com texto são lidos com pypdf (não precisam de visão).
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.agents.comum import extrair_json, texto_da_resposta
from src.agents.contratos import RuleSpec, Segmento
from src.gobr.grammar import normalizar
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


def esquema_estrito(modelo: type) -> dict:
    """JSON Schema do modelo com TODOS os campos obrigatórios, em todos os níveis.

    No `RuleSpec` todo campo tem valor padrão, então o esquema do Pydantic não
    exige nenhum; com a saída restrita a ele, modelos pequenos fechavam o JSON
    depois de 3 campos, sem `segmentos`. Aqui o modelo precisa escrever tudo
    (usando null/[] quando não houver valor).
    """
    esquema = modelo.model_json_schema()

    def exigir(objeto: dict) -> None:
        if objeto.get("type") == "object" and "properties" in objeto:
            objeto["required"] = list(objeto["properties"])
        for valor in objeto.values():
            if isinstance(valor, dict):
                exigir(valor)
            elif isinstance(valor, list):
                for item in valor:
                    if isinstance(item, dict):
                        exigir(item)

    exigir(esquema)
    return esquema


_BLOQUEIO = re.compile(r"^\s*n[aã]o\s+agendar\s+(?P<quem>.+?)\s*\.?\s*$", re.IGNORECASE)


def promover_bloqueios(spec: RuleSpec) -> RuleSpec:
    """"Não agendar X" nas restrições gerais vira um grupo `nao agendar` (regra 2 do prompt).

    Modelos pequenos às vezes deixam esses bloqueios como restrição geral; sem
    a correção, eles sumiriam do GOBR.
    """
    restantes, novos = [], []
    existentes = {normalizar(s.nome) for s in spec.segmentos}
    for restricao in spec.restricoes_gerais:
        m = _BLOQUEIO.match(restricao)
        if not m:
            restantes.append(restricao)
            continue
        nome = restricao.strip().rstrip(".")
        if normalizar(nome) not in existentes:
            novos.append(Segmento(nome=nome, criterio=m.group("quem"), nao_agendar=True))
            spec.observacoes.append(f'"{nome}" estava como restrição geral e virou um grupo "não agendar".')
    spec.restricoes_gerais = restantes
    spec.segmentos = [*spec.segmentos, *novos]
    return spec


class AgentePercepcao:
    def __init__(self, llm: Any = None, *, max_correcoes: int = 1):
        self._llm = llm
        self.max_correcoes = max_correcoes

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from src.llm_config import create_llm_para

            # Saída presa ao esquema do RuleSpec: sem raciocínio solto nem texto fora do JSON
            # (o qwen3:4b, sem isso, escrevia milhares de tokens e estourava o contexto).
            self._llm = create_llm_para("percepcao", formato=esquema_estrito(RuleSpec))
        return self._llm

    def extrair(self, textos: list[str], imagens: list[str] | None = None) -> RuleSpec:
        mensagens = percepcao_prompt.mensagens(textos, imagens or [])
        erro = ""
        for _ in range(self.max_correcoes + 1):
            resposta = texto_da_resposta(self.llm.invoke(mensagens))
            try:
                return promover_bloqueios(RuleSpec.model_validate(extrair_json(resposta)))
            except (ValueError, ValidationError) as exc:
                erro = str(exc)[:800]
                mensagens = [*mensagens, ("ai", resposta),
                             ("human", percepcao_prompt.CORRECAO.format(erro=erro))]
        raise PercepcaoFalhou(f"O modelo não devolveu um RuleSpec válido: {erro}")
