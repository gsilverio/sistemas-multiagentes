"""Agente Percepção: texto, imagem ou PDF com regras comerciais → `RuleSpec`.

Usa um modelo com visão (gemma3:12b) e pede JSON; a resposta é validada
pelo Pydantic e, se vier inválida, o modelo recebe o erro e tenta de novo
uma vez. PDFs com texto são lidos com pypdf (não precisam de visão).

Os erros da Percepção passariam em silêncio (o Validador confere o GOBR contra
o `RuleSpec`), então duas conferências sem LLM olham o texto original, na
primeira mensagem da conversa: faixa de dias diferente do critério escrito
(`alinhar_faixas`) e números de dias ou preferências que sumiram (`omissoes`,
que pede ao modelo para completar, uma vez).
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.agents.comum import extrair_json, texto_da_resposta
from src.agents.contratos import Faixa, RuleSpec, Segmento
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


_ENTRE = re.compile(r"\bentre (?:os )?(\d+)(?: dias)? (?:e|a|ate) (\d+)")
_ULTIMOS_E_SEM = re.compile(r"\bultimos (\d+)(?: dias)?\b.*?\b(\d+)(?: dias)? sem\b")
_MAIS_DE = re.compile(r"\b(?:mais(?: de)?|acima de|superior a) (\d+) dias")
_ATE = re.compile(r"\b(?:ate|ultimos|menos de) (\d+) dias")
_COMPRA = re.compile(r"\bcompr|\bpedido|\bpositiva")


def faixa_do_texto(criterio: str) -> tuple[int, int | None] | None:
    """(mínimo, máximo) de dias quando o critério escrito não deixa dúvida; senão None."""
    t = re.sub(r"\ba cada \d+ dias", "", normalizar(criterio))
    numeros = re.findall(r"\d+", t)
    if len(numeros) == 2:
        if m := _ENTRE.search(t):
            a, b = int(m[1]), int(m[2])
        elif m := _ULTIMOS_E_SEM.search(t):
            a, b = int(m[2]), int(m[1])
        else:
            return None
        return (a, b) if a < b else None
    if len(numeros) == 1:
        if m := _MAIS_DE.search(t):
            return int(m[1]), None
        if m := _ATE.search(t):
            return 0, int(m[1])
    return None


def alinhar_faixas(spec: RuleSpec) -> RuleSpec:
    """Corrige a `faixa` que contradiz o critério escrito ("em até 45 dias" lido como 45 a 45)."""
    for s in spec.segmentos:
        escrita = faixa_do_texto(s.criterio)
        if escrita is None or (s.faixa is None and not _COMPRA.search(normalizar(s.criterio))):
            continue
        lida = (s.faixa.min_dias or 0, s.faixa.max_dias) if s.faixa else None
        if lida != escrita:
            medida = s.faixa.medida if s.faixa else "dias_sem_compra"
            s.faixa = Faixa(medida=medida, min_dias=escrita[0], max_dias=escrita[1])
            spec.observacoes.append(f'A faixa de "{s.nome}" foi ajustada ao que está escrito: '
                                    f"{s.faixa.como_texto()}.")
    return spec


def omissoes(texto: str, spec: RuleSpec) -> list[str]:
    """O que está no texto do usuário e não aparece em lugar nenhum do `RuleSpec`."""
    original, lido = normalizar(texto), normalizar(spec.model_dump_json())
    faltando = [f'"{n} dias" não aparece em nenhum grupo.'
                for n in dict.fromkeys(re.findall(r"\b(\d+) dias\b", original))
                if n != "0" and not re.search(rf"\b{n}\b", lido)]
    grupos = normalizar(" ".join(" ".join([s.criterio, s.frequencia_texto or "", *s.restricoes])
                                 for s in spec.segmentos))
    if "prefer" in original and "prefer" not in grupos:
        faltando.append('A preferência de dia da semana sumiu: coloque-a em "restricoes" do grupo a que ela se '
                        'refere, ex.: "preferência na sexta".')
    return faltando


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
                spec = self._interpretar(resposta, textos)
                return self._completar(spec, textos, mensagens, resposta)
            except (ValueError, ValidationError) as exc:
                erro = str(exc)[:800]
                mensagens = [*mensagens, ("ai", resposta),
                             ("human", percepcao_prompt.CORRECAO.format(erro=erro))]
        raise PercepcaoFalhou(f"O modelo não devolveu um RuleSpec válido: {erro}")

    @staticmethod
    def _interpretar(resposta: str, textos: list[str]) -> RuleSpec:
        spec = promover_bloqueios(RuleSpec.model_validate(extrair_json(resposta)))
        # Nas mensagens seguintes o usuário muda o que escreveu antes: o critério antigo não manda mais.
        return alinhar_faixas(spec) if len(textos) == 1 else spec

    def _completar(self, spec: RuleSpec, textos: list[str], mensagens: list, resposta: str) -> RuleSpec:
        if len(textos) != 1 or not (faltando := omissoes(textos[0], spec)):
            return spec
        pedido = percepcao_prompt.COMPLETAR.format(faltando="\n".join(f"- {f}" for f in faltando))
        try:
            nova = self._interpretar(texto_da_resposta(self.llm.invoke(
                [*mensagens, ("ai", resposta), ("human", pedido)])), textos)
        except Exception:  # noqa: BLE001 - a primeira leitura é válida; completar é só uma melhora
            return spec
        melhorou = len(omissoes(textos[0], nova)) < len(faltando) and len(nova.segmentos) >= len(spec.segmentos)
        return nova if melhorou else spec
