"""Guard de fidelidade: um texto explica TODOS os fatos do rascunho, e só eles?

Verificações (todas determinísticas, sobre texto normalizado):

1. Nomes citados (times, carteiras, segmentações, valores de listas)
   aparecem literalmente.
2. Todo número da regra aparece; nenhum número novo é introduzido
   (ex.: inventar "14 dias" numa regra sem frequência fixa).
3. Cada fato tem suas chaves (palavras-conceito do glossário) presentes.
4. `nao agendar` vira proibição de agendar, não estado do cliente; e não
   aparece proibição de agendar em regra que não tem `nao agendar`.
5. Cada `ou` lógico da regra continua no texto.
6. O texto não fala da implementação (GOBR, JSON, parser...).

Não prova correção semântica: é uma rede de segurança contra omissões e
alucinações, que são os erros observados na linha de base.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.explicacao.rascunho import Rascunho
from src.gobr.grammar import normalizar

_PROIBIR_AGENDAR = re.compile(
    r"\bnao\s+(?:(?:devem|deve|podem|pode|serao|sera|vao|sao)\s+)?(?:ser\s+)?(?:mais\s+)?agendad\w*"
    r"|\bnao\s+(?:devem|deve|podem|pode)\s+(?:ser\s+)?(?:incluid\w*|entrar|aparecer)\s+(?:n[ao]s?\s+)?agend"
    r"|\bnao\s+agendar\b|\bproibid\w*\s+(?:de\s+)?agendar|\bnao\s+e\s+permitido\s+agendar"
    r"|\bficam\s+fora\s+da\s+agenda|\bsem\s+agendamento"
)
_ESTADO_AGENDADO = re.compile(
    r"\bja\s+(?:\w+\s+)?(?:agendad|possuem\s+agendamento|tem\s+agendamento)"
    r"|\bnao\s+(?:pode|podem|deve|devem)\s+estar\s+agendad"
)
_IMPLEMENTACAO = re.compile(r"\b(?:gobr|json|parser|dsl|ast|regex|rascunho)\b|bloco\s+parametros")


def _numeros(texto: str) -> set[str]:
    t = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", texto)   # 50.000 → 50000
    t = re.sub(r"(?<=\d),0+\b", "", t)                     # 50000,00 → 50000
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+", t)}


@dataclass
class ResultadoGuarda:
    aprovado: bool
    problemas: list[str] = field(default_factory=list)

    def como_texto(self) -> str:
        return "\n".join(f"- {p}" for p in self.problemas)


def verificar(rascunho: Rascunho, texto: str, *, gobr: str | None = None) -> ResultadoGuarda:
    """Confere `texto` contra o rascunho. `gobr` (o bloco original) refina o
    conjunto de números obrigatórios; sem ele, usa os trechos dos fatos."""
    t = normalizar(texto)
    problemas: list[str] = []
    if not t:
        return ResultadoGuarda(False, ["Texto vazio."])

    for nome in rascunho.nomes_citados:
        alvo = normalizar(nome)
        if not re.search(rf"(?<!\w){re.escape(alvo)}(?!\w)", t):
            problemas.append(f'Nome não preservado literalmente: "{nome}".')

    fonte = "\n".join(f.gobr for f in rascunho.fatos) if gobr is None else _corpo(gobr)
    obrigatorios = _numeros(normalizar(fonte))
    permitidos = obrigatorios | _numeros(normalizar(rascunho.texto))
    presentes = _numeros(t)
    for n in sorted(obrigatorios - presentes, key=int):
        problemas.append(f"Número da regra ausente no texto: {n}.")
    for n in sorted(presentes - permitidos, key=int):
        problemas.append(f"Número que não existe na regra: {n}.")

    for fato in rascunho.fatos:
        for chave in fato.chaves:
            if not re.search(chave, t):
                problemas.append(f'{fato.tipo.replace("_", " ").capitalize()} não preservado(a): "{fato.gobr}".')
                break

    if rascunho.nao_agendar:
        if not _PROIBIR_AGENDAR.search(t):
            problemas.append('"nao agendar" precisa aparecer como proibição de agendar o cliente.')
        if _ESTADO_AGENDADO.search(t):
            problemas.append("O texto fala de agendamentos já existentes; a regra proíbe criar agendamentos.")
    elif _PROIBIR_AGENDAR.search(t):
        problemas.append("O texto proíbe agendar, mas a regra não tem `nao agendar`.")

    if len(re.findall(r"\bou\b", t)) < rascunho.conectores_ou:
        problemas.append("Um `ou` lógico da regra foi perdido (as condições alternativas viraram cumulativas).")

    if _IMPLEMENTACAO.search(t):
        problemas.append("O texto menciona a implementação (GOBR, JSON, parser...).")

    return ResultadoGuarda(not problemas, problemas)


def _corpo(gobr: str) -> str:
    """Bloco sem a linha de cabeçalho: o nome da regra não precisa ser repetido."""
    linhas = gobr.strip().splitlines()
    if linhas and re.match(r"\s*(?:limpar\s+)?parametros\b", linhas[0], re.IGNORECASE):
        linhas = linhas[1:]
    return "\n".join(linhas)
