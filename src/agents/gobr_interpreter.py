from src.llm_config import create_llm

from src.models.gobr_parsed_rule import GobrParsedRule
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
        parsed_rule: GobrParsedRule
    ) -> str | None:

        if not parsed_rule.escopos:
            return None

        partes = []

        for index, escopo in enumerate(parsed_rule.escopos):
            tipo = escopo.tipo.lower()
            valor = escopo.valor.strip()

            if tipo == "carteira":
                texto = f'Clientes da carteira "{valor}"'

            elif tipo == "segmentacao":
                texto = f'Clientes da segmentação "{valor}"'

            elif tipo == "time":
                if valor.lower() == "todos":
                    texto = "Todos os clientes"
                elif valor.lower() in ("toda empresa", "empresa toda"):
                    texto = "Todos os clientes da empresa"
                else:
                    texto = f'Clientes do time "{valor}"'

            else:
                texto = f'Clientes de {tipo} "{valor}"'

            if index == 0:
                partes.append(texto)
            else:
                operador = escopo.operador or "e"
                # Evita "... e clientes da carteira...", que pode
                # sugerir dois públicos independentes. Faz a interseção
                # dos escopos na mesma expressão nominal.
                complemento = texto.replace("Clientes ", "", 1)
                partes.append(f"{operador} {complemento}")

        return " ".join(partes) + "."

    def interpret(
        self,
        parsed_rule: GobrParsedRule
    ) -> SemanticInterpretation:

        result = self.chain.invoke({
            "nome": parsed_rule.nome,

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

        # Se não existem modificadores,
        # o agente não pode inventá-los.
        if not parsed_rule.modificadores:
            result.modificadores = []

        # Elementos desconhecidos devem ser sinalizados.
        for desconhecido in parsed_rule.desconhecidos:
            if desconhecido not in result.incertezas:
                result.incertezas.append(desconhecido)

        return result