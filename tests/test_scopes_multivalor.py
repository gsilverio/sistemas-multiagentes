"""Testes da AST multivalor, texto conservador e integração com guard."""

import json
import unittest

from src.models.gobr_parsed_rule import ParsedScope
from src.models.scope_text import scope_subject
from src.parsers.gobr_parser import GobrParser
from src.validation.explanation_guard import ExplanationGuard


class MultivalorTests(unittest.TestCase):
    def setUp(self):
        self.parser = GobrParser()
        self.guard = ExplanationGuard()

    def test_parse_uma_lista_de_tres_carteiras(self):
        rule = self.parser.parse('''parametros "Regra":
    da carteira("construtora", "matcon", "demais_segment")
    frequencia de 14 dias''')
        self.assertEqual(1, len(rule.escopos))
        self.assertEqual(['construtora', 'matcon', 'demais_segment'], rule.escopos[0].valores)
        self.assertEqual([], rule.desconhecidos)

    def test_operador_entre_duas_clausulas_nao_dentro_da_lista(self):
        rule = self.parser.parse('''parametros "Regra":
    do time("campo", "televendas")
    e da carteira("construtora", "matcon")''')
        self.assertEqual(2, len(rule.escopos))
        self.assertEqual(['campo', 'televendas'], rule.escopos[0].valores)
        self.assertEqual(['construtora', 'matcon'], rule.escopos[1].valores)
        self.assertIsNone(rule.escopos[0].operador)
        self.assertEqual('e', rule.escopos[1].operador)
        self.assertIn(' e das carteiras ', scope_subject(rule.escopos))
        self.assertEqual([], rule.desconhecidos)

    def test_conector_ou_explicito_preservado(self):
        rule = self.parser.parse('''parametros "Regra":
    do time("campo", "televendas")
    ou da segmentacao("ativa", "prospect")''')
        self.assertEqual('ou', rule.escopos[1].operador)
        self.assertIn(' ou das segmentações ', scope_subject(rule.escopos))

    def test_listas_formatadas_sem_afirmar_logica_interna(self):
        rule = self.parser.parse('''parametros "Regra":
    do time("campo", "televendas")''')
        self.assertEqual('Os clientes dos times "campo", "televendas"', scope_subject(rule.escopos))

    def test_validator_exige_presenca_de_todos_os_nomes(self):
        rule = self.parser.parse('''parametros "Regra":
    da carteira("sul_sudeste_centrooeste", "norte_nordeste")
    frequencia de 7 dias''')
        invalid = 'Clientes da carteira "sul_sudeste_centrooeste" têm atendimento a cada 7 dias.'
        valid = 'Os clientes das carteiras "sul_sudeste_centrooeste", "norte_nordeste" têm atendimento a cada 7 dias.'
        self.assertFalse(self.guard.inspect(rule, invalid).accepted)
        self.assertTrue(self.guard.inspect(rule, valid).accepted)
        fallback = self.guard.conservative_fallback(rule)
        self.assertTrue(self.guard.inspect(rule, fallback).accepted)
        self.assertIn('"norte_nordeste"', fallback)

    def test_aceita_serializacao_de_json_antigo(self):
        old_json = '{"tipo":"time","valor":"televendas","operador":null}'
        scope = ParsedScope.model_validate_json(old_json)
        self.assertEqual(['televendas'], scope.valores)
        self.assertEqual('televendas', scope.valor)
        self.assertEqual(['televendas'], json.loads(scope.model_dump_json())['valores'])

    def test_argumento_malformado_nao_perde_escopo(self):
        rule = self.parser.parse('''parametros "Regra":
    do time("campo", televendas)''')
        self.assertEqual([], rule.escopos)
        self.assertEqual(['do time("campo", televendas)'], rule.desconhecidos)

    def test_escopo_malformado_apos_quando_nao_vira_condicao(self):
        rule = self.parser.parse('''parametros "Regra":
    quando inadimplentes
    e do time("campo", televendas)''')
        self.assertEqual(['e do time("campo", televendas)'], rule.desconhecidos)
        self.assertEqual(1, len(rule.condicoes))

    def test_nome_contendo_virgula_e_um_unico_argumento(self):
        rule = self.parser.parse('''parametros "Regra":
    do time("campo, vendas", "televendas")''')
        self.assertEqual(['campo, vendas', 'televendas'], rule.escopos[0].valores)


if __name__ == '__main__':
    unittest.main()
