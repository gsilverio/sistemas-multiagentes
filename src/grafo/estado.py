"""Estado do grafo e configuração dos agentes.

O estado guarda só tipos simples (dict, list, str), para o checkpointer
serializar sem registrar classes; os nós reconstroem os modelos Pydantic de
`src.agents.contratos` quando precisam.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional, TypedDict

MAX_TENTATIVAS_GERACAO = 3
MAX_RODADAS_PERGUNTAS = 2

# Campos derivados da conversa: zerados quando chega uma mensagem nova.
CAMPOS_DERIVADOS = {
    "spec": None, "analise": None, "contextos": None, "blocos": {}, "gobr": None, "conferencia": None,
    "tentativas": 0, "explicacoes": None, "resposta": None, "erro": None,
}


class EntradaSimples(TypedDict, total=False):
    """Formulário de entrada no LangGraph Studio."""

    pedido: str        # descrição das regras ou pedido de ajuste
    anexo: str         # opcional: caminho de um PDF/imagem, ex.: data/exemplos/RegraMaxlub.pdf


class Estado(TypedDict, total=False):
    pedido: Optional[str]                             # entrada simples (Studio); `inicio` vira mensagem
    anexo: Optional[str]
    mensagens: Annotated[list[dict], operator.add]   # {"texto": str, "imagens": [base64], "origem": str}
    processadas: int                                  # quantas mensagens já iniciaram um ciclo
    rodadas_perguntas: int
    perguntas_feitas: list[str]                       # ids já perguntados neste ciclo
    spec: Optional[dict]                              # RuleSpec da Percepção
    analise: Optional[dict]                           # Analise (com a spec ajustada)
    contextos: Optional[dict]                         # nome do segmento → ContextoSegmento
    blocos: dict                                      # nome do segmento → bloco GOBR
    gobr: Optional[str]
    conferencia: Optional[dict]                       # Conferencia do Validador
    tentativas: int                                   # gerações feitas neste ciclo
    explicacoes: Optional[list]
    resposta: Optional[str]
    erro: Optional[str]
    trace: Annotated[list[dict], operator.add]
    conversa: Annotated[list[dict], operator.add]     # histórico exibido: {"papel": usuario|agente, "texto"}


@dataclass
class Agentes:
    """Dependências do grafo. Os testes injetam agentes com LLM falso."""

    percepcao: Any = None
    analista: Any = None
    recuperador: Any = None
    gerador: Any = None
    validador: Any = None
    explicador: Any = None
    orquestrador_llm: Any = None      # None = só as regras determinísticas
    interativo: bool = True           # False = nunca pausa para perguntar (lote, avaliação)
    max_tentativas: int = MAX_TENTATIVAS_GERACAO
    max_rodadas: int = MAX_RODADAS_PERGUNTAS
    extras: dict = field(default_factory=dict)

    @classmethod
    def padrao(cls, *, orquestrador_com_llm: bool = True, explicar_com_llm: bool = False,
               interativo: bool = True) -> "Agentes":
        """Agentes reais (Ollama). Criados sob demanda: nada conecta ao Ollama aqui."""
        from src.agents.analista import AgenteAnalista
        from src.agents.explicador import AgenteExplicador
        from src.agents.gerador import AgenteGerador
        from src.agents.percepcao import AgentePercepcao
        from src.agents.recuperador import AgenteRecuperador
        from src.agents.validador import AgenteValidador

        orquestrador = None
        if orquestrador_com_llm:
            from src.llm_config import create_llm_para

            orquestrador = create_llm_para("orquestrador", formato="json")
        explicador_llm = None
        if explicar_com_llm:
            from src.llm_config import create_llm_para

            explicador_llm = create_llm_para("explicador")
        return cls(percepcao=AgentePercepcao(), analista=AgenteAnalista(), recuperador=AgenteRecuperador(),
                   gerador=AgenteGerador(), validador=AgenteValidador(),
                   explicador=AgenteExplicador(explicador_llm, usar_llm=explicar_com_llm),
                   orquestrador_llm=orquestrador, interativo=interativo)
