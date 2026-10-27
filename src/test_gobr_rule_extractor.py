import tempfile
import unittest
from pathlib import Path

from src.parsers.gobr_rule_extractor import GobrRuleExtractor


class GobrRuleExtractorTests(unittest.TestCase):
    def setUp(self):
        self.extractor = GobrRuleExtractor()

    def _extract(self, texto: str):
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / "exemplo.gobr"
            arquivo.write_text(texto, encoding="utf-8")
            return self.extractor.extract_file(arquivo)

    def test_separa_blocos_e_mantem_linhas(self):
        regras = self._extract('''version 1.0

parametros "Primeira":
    frequencia de 7 dias

# Separador
parametros "Segunda":
    nao agendar
''')
        self.assertEqual(2, len(regras))
        self.assertEqual(["Primeira", "Segunda"], [r.nome for r in regras])
        self.assertNotIn("Segunda", regras[0].gobr)
        self.assertEqual((3, 4), (regras[0].linha_inicio, regras[0].linha_fim))
        self.assertEqual("exemplo_001", regras[0].id)

    def test_preserva_contexto_direto_e_indireto(self):
        regras = self._extract('''version 1.0

time "empresa toda":
    vendedores todos

carteira "empresa toda":
    clientes carteirafixa("empresa toda")

carteira livre "outra":
    clientes carteiraqualquer("x")

parametros "Agendamento":
    da carteira("empresa toda")
    frequencia de 7 dias
''')
        self.assertEqual(1, len(regras))
        self.assertEqual(["time", "carteira"], [d.tipo for d in regras[0].contexto])
        self.assertFalse(regras[0].referencias_nao_resolvidas)
        self.assertNotIn("vendedores todos", regras[0].gobr)

    def test_varios_argumentos_e_escopos(self):
        regras = self._extract('''segmentacao "segmento":
    quando algo

carteira livre "regional_a":
    clientes todos

carteira livre "regional_b":
    clientes todos

parametros "Regra":
    da carteira("regional_a","regional_b")
    e da segmentacao("segmento")
    frequencia de 14 dias
''')
        self.assertEqual(3, len(regras[0].contexto))

    def test_nao_remove_duplicatas_e_aceita_aspas_em_titulo(self):
        regras = self._extract('''parametros "Usar CNAE ("03","47")":
    nao agendar

parametros "Usar CNAE ("03","47")":
    nao agendar
''')
        self.assertEqual(2, len(regras))
        self.assertEqual('Usar CNAE ("03","47")', regras[0].nome)
        self.assertNotEqual(regras[0].id, regras[1].id)

    def test_referencia_externa_nao_perde_regra(self):
        regras = self._extract('''parametros "Regra":
    da carteira("inexistente")
    frequencia de 7 dias
''')
        self.assertEqual(["inexistente"], regras[0].referencias_nao_resolvidas)

    def test_acervo_referencia(self):
        diretorio = Path("data/referencias")
        if not diretorio.exists():
            self.skipTest("Diretorio de referencias nao foi fornecido")
        regras = self.extractor.extract_directory(diretorio)
        self.assertEqual(1699, len(regras))
        self.assertEqual(1699, len({r.id for r in regras}))
        self.assertTrue(all(r.gobr.startswith('parametros "') for r in regras))


if __name__ == "__main__":
    unittest.main()
