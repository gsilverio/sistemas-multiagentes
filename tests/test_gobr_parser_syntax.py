"""Cobertura incremental de variacoes sintaticas presentes no acervo GOBR."""
import unittest
from pathlib import Path

from src.parsers.gobr_parser import GobrParser
from src.parsers.gobr_rule_extractor import GobrRuleExtractor


class GobrParserSyntaxTests(unittest.TestCase):
    def setUp(self):
        self.parser = GobrParser()

    def test_nao_agendar_acentuado(self):
        rule = self.parser.parse('''parametros "teste":
    da carteira("empresa toda")
    não agendar''')
        self.assertTrue(rule.nao_agendar)
        self.assertEqual([], rule.desconhecidos)

    def test_frequencia_acentuada(self):
        rule = self.parser.parse('''parametros "teste":
    frequência de 14 dias''')
        self.assertEqual('14 dias', rule.frequencia)
        self.assertEqual([], rule.desconhecidos)

    def test_nome_com_aspas_internas(self):
        rule = self.parser.parse('''parametros "CNAE ("03","47")":
    frequencia mensal()''')
        self.assertEqual('CNAE ("03","47")', rule.nome)
        self.assertEqual('mensal()', rule.frequencia)
        self.assertEqual([], rule.desconhecidos)

    def test_modificadores_documentados(self):
        rule = self.parser.parse('''parametros "teste":
    da carteira("empresa toda")
    fixo
    roteirizavel
    sem choque na rota com time("campo")
    a partir do dia 16''')
        self.assertEqual([
            'fixo', 'roteirizavel',
            'sem choque na rota com time("campo")',
            'a partir do dia 16',
        ], rule.modificadores)
        self.assertEqual([], rule.desconhecidos)

    def test_comentario_nao_e_instrucao(self):
        rule = self.parser.parse('''parametros "teste":
    # preferencia na segunda
    frequencia semanal()''')
        self.assertEqual([], rule.desconhecidos)
        self.assertEqual([], rule.modificadores)

    def test_preferencia_multilinha_conservada(self):
        rule = self.parser.parse('''parametros "teste":
    da carteira("empresa toda")
    preferencia
        maior na segunda e
        menor na terça e
        normal na quinta''')
        self.assertEqual(1, len(rule.modificadores))
        self.assertIn('maior na segunda', rule.modificadores[0])
        self.assertIn('normal na quinta', rule.modificadores[0])
        self.assertEqual([], rule.desconhecidos)

    def test_nao_aceita_escopo_multivalor_sem_suporte(self):
        rule = self.parser.parse('''parametros "teste":
    do time("campo", "televendas")''')
        # Para nao perder a cardinalidade, mantem desconhecido ate haver
        # uma estrutura propria para multiplos valores de escopo.
        self.assertEqual([], rule.escopos)
        self.assertEqual(['do time("campo", "televendas")'], rule.desconhecidos)

    def test_multilinha_escopo_multivalor_nao_se_perde(self):
        rule = self.parser.parse('''parametros "teste":
    do time("campo",
        "televendas")''')
        self.assertEqual(1, len(rule.desconhecidos))
        self.assertIn('"televendas"', rule.desconhecidos[0])

    def test_erro_real_regra_080_001_corrigido(self):
        path = Path('data/referencias/regra_080.gobr')
        if not path.exists():
            self.skipTest('Acervo nao presente')
        block = GobrRuleExtractor().extract_file(path)[0]
        rule = self.parser.parse(block.gobr)
        self.assertIn('("03","47","27")', rule.nome)
        self.assertEqual([], rule.desconhecidos)


if __name__ == '__main__':
    unittest.main()
