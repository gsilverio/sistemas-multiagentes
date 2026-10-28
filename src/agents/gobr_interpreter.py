import json

from src.llm_config import create_llm

from src.models.gobr_parsed_rule import GobrParsedRule
from src.models.scope_text import scope_subject
from src.models.semantic_interpretation import SemanticInterpretation
from src.prompts.gobr_interpreter_prompt import GOBR_INTERPRETER_PROMPT


class GobrInterpreterAgent:
    def __init__(self):
        self.model = create_llm()

        self.structured_model = self.model.with_structured_output(
            SemanticInterpretation,
            method="json_schema"
        )

        self.chain = (
            GOBR_INTERPRETER_PROMPT
            | self.structured_model
        )

    def _build_publico(
        self,
        parsed_rule: GobrParsedRule,
    ) -> str | None:
        if not parsed_rule.escopos:
            return None
        texto = scope_subject(parsed_rule.escopos)
        # Mantém a abertura de texto dos exemplos originais do agente.
        if texto.startswith("Os clientes "):
            texto = "Clientes " + texto[len("Os clientes "):]
        return texto + "."

    def interpret(
        self,
        parsed_rule: GobrParsedRule,
        contexto: list[dict[str, str]] | None = None,
    ) -> SemanticInterpretation:

        result = self.chain.invoke({
            "nome": parsed_rule.nome,

            "contexto_definicoes": (
                json.dumps(contexto, ensure_ascii=False) if contexto else "Nenhuma."
            ),

            "escopos": [
                scope.model_dump()
                for scope in parsed_rule.escopos
            ],

            "condicoes": [
                condition.model_dump()
                for condition in parsed_rule.condicoes
            ],

            "frequencia": parsed_rule.frequencia,

            "nao_agendar": parsed_rule.nao_agendar,

            "modificadores": parsed_rule.modificadores,

            "filtros_temporais": parsed_rule.filtros_temporais,

            "desconhecidos": parsed_rule.desconhecidos
        })

        result.publico = self._build_publico(parsed_rule)

        # -------------------------------------------------
        # Garantias estruturais
        # -------------------------------------------------

        # Se não existem condições na GOBR,
        # o agente não pode inventar critérios.
        if not parsed_rule.condicoes:
            result.criterios = []

        # Evita interpretações artificiais dos critérios mais comuns.
        # O modelo continua responsável pelas demais condições.
        if len(result.criterios) == len(parsed_rule.condicoes):
            for i, condicao in enumerate(parsed_rule.condicoes):
                texto = condicao.texto.strip().lower()
                if texto == "inadimplentes":
                    result.criterios[i] = "O cliente deve estar inadimplente."
                elif texto == "é o cliente filial":
                    result.criterios[i] = "O cliente deve ser filial."
                elif texto == "sem venda":
                    result.criterios[i] = "O cliente deve estar sem venda."

        # Se "nao agendar" existe na GOBR,
        # obrigatoriamente deve existir uma restrição.
        if parsed_rule.nao_agendar:
            result.restricoes = [
                "O cliente não deve ser incluído no agendamento de atendimento."
            ]

            # "Não agendar" é restrição,
            # não é uma regra de frequência/atendimento.
            if not parsed_rule.frequencia:
                result.regra_atendimento = None

        # Filtros de execucao sao preservados deterministamente; nao
        # devem ser interpretados como preferencias de visita.
        result.filtros_temporais = [
            f"A regra deve ser aplicada somente {filtro}."
            for filtro in parsed_rule.filtros_temporais
        ]

        # Se não existem modificadores,
        # o agente não pode inventá-los.
        if not parsed_rule.modificadores:
            result.modificadores = []

        # Elementos desconhecidos devem ser sinalizados.
        for desconhecido in parsed_rule.desconhecidos:
            if desconhecido not in result.incertezas:
                result.incertezas.append(desconhecido)

        return result