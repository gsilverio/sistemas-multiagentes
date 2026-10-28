"""Regressao da etapa 2C a partir dos casos reais de regra_168.gobr.

Estes testes conferem a separacao entre dias de APLICACAO e de PREFERENCIA.
Nao realizam inferencia de LLM e nao calculam accuracy.
"""
import unittest
from pathlib import Path

from src.models.gobr_parsed_rule import GobrParsedRule
from src.parsers.gobr_parser import GobrParser
from src.parsers.gobr_rule_extractor import GobrRuleExtractor
from src.validation.explanation_guard import ExplanationGuard


class PreferenciasEFiltrosTemporaisTests(unittest.TestCase):
    def setUp(self):
        self.parser = GobrParser()
        self.guard = ExplanationGuard()

    def test_preferencia_inline_multilinha_regra_168_003(self):
        caminho = Path("data/referencias/regra_168.gobr")
        if not caminho.exists():
            self.skipTest("Acervo nao incluido")
        blocos = GobrRuleExtractor().extract_file(caminho)
        bloco = next(b for b in blocos if b.id == "regra_168_003")
        regra = self.parser.parse(bloco.gobr)
        self.assertEqual([], regra.desconhecidos)
        self.assertEqual([], regra.filtros_temporais)
        self.assertEqual(3, len(regra.modificadores))
        self.assertEqual(["roteirizavel", "fixo"], regra.modificadores[1:])
        self.assertEqual(
            ["preferencia maior na segunda", "e maior na terca", "e maior na quarta",
             "e maior na quinta", "e maior na sexta"],
            regra.modificadores[0].splitlines(),
        )
        self.assertEqual("diaria()", regra.frequencia)

    def test_filtro_independente_da_preferencia_regra_168_004_a_007(self):
        caminho = Path("data/referencias/regra_168.gobr")
        if not caminho.exists():
            self.skipTest("Acervo nao incluido")
        blocos = GobrRuleExtractor().extract_file(caminho)
        mapa = {b.id: b for b in blocos}
        for numero, dia in [(4, "segunda"), (5, "terca"), (6, "quarta"), (7, "quinta")]:
            with self.subTest(dia=dia):
                regra = self.parser.parse(mapa[f"regra_168_{numero:03d}"].gobr)
                self.assertEqual([f"na {dia}"], regra.filtros_temporais)
                self.assertEqual([f"preferencia maior na {dia}"], regra.modificadores[:1])
                self.assertEqual([], regra.desconhecidos)

    def test_preferencia_nao_engole_filtro_apos_modificador(self):
        regra = self.parser.parse('''parametros "Teste":
    preferencia maior na segunda
    na terca
    frequencia diaria()''')
        self.assertEqual(["preferencia maior na segunda"], regra.modificadores)
        self.assertEqual(["na terca"], regra.filtros_temporais)
        self.assertEqual([], regra.desconhecidos)

    def test_filtros_documentados_sao_separados_dos_modificadores(self):
        regra = self.parser.parse('''parametros "Teste":
    da carteira("empresa toda")
    na ultima semana
    a partir do dia 16
    preferencia maior na segunda
    fixo''')
        self.assertEqual(["na ultima semana", "a partir do dia 16"], regra.filtros_temporais)
        self.assertEqual(["preferencia maior na segunda", "fixo"], regra.modificadores)
        self.assertEqual([], regra.desconhecidos)

    def test_validacao_nao_confunde_filtro_temporal_com_preferencia(self):
        regra = self.parser.parse('''parametros "Teste":
    da carteira("empresa toda")
    na segunda
    preferencia maior na segunda''')
        so_preferencia = ('Os clientes da carteira "empresa toda" tem ' 
                         'preferencia maior na segunda.')
        self.assertFalse(self.guard.inspect(regra, so_preferencia).accepted)
        texto = self.guard.conservative_fallback(regra)
        self.assertTrue(self.guard.inspect(regra, texto).accepted)
        self.assertIn("A regra deve ser aplicada somente na segunda.", texto)
        self.assertIn('Modificador informado: "preferencia maior na segunda".', texto)

    def test_filtro_temporal_negado_nao_aprovado(self):
        regra = self.parser.parse('''parametros "Teste":
    da carteira("empresa toda")
    na segunda''')
        self.assertFalse(self.guard.inspect(
            regra, 'A regra da carteira "empresa toda" nao se aplica na segunda.'
        ).accepted)

    def test_json_antigo_continua_valido(self):
        regra = GobrParsedRule.model_validate({"nome": "Teste", "modificadores": []})
        self.assertEqual([], regra.filtros_temporais)

    def test_seis_exemplos_da_primeira_entrega_sem_alteracao_estrutural(self):
        pasta = Path("outputs/first_delivery")
        if not pasta.exists():
            self.skipTest("Resultados da primeira entrega nao incluidos")
        cases = [
            {"id": path.parent.name, "gobr": path.read_text(encoding="utf-8")}
            for path in sorted(pasta.glob("*/input.gobr"))
        ]
        self.assertEqual(6, len(cases))
        for caso in cases:
            with self.subTest(id=caso['id']):
                regra = self.parser.parse(caso['gobr'])
                self.assertEqual([], regra.desconhecidos)
                self.assertEqual([], regra.filtros_temporais)


if __name__ == "__main__":
    unittest.main()
