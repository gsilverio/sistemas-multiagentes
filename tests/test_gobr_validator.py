"""Validador GOBR (etapa 1): acervo, códigos de erro e correções do parser.

Roda sem LLM e sem GPU.
"""
import glob
import unittest
from pathlib import Path

from src.gobr import grammar
from src.gobr.parser import GobrParser
from src.gobr.vocabulary import VocabularioCondicoes, modelo_condicao
from src.validation.gobr_validator import GobrValidator

VALIDO = '''version 1.0

time "televendas":
    vendedores todostelevendas ativados

carteira "carteira televendas":
    clientes carteirafixa("televendas")

parametros "Ativos":
    da carteira("carteira televendas")
    quando recencia de positivacao em 60 dias
        e adimplentes
    frequencia de 14 dias
    preferencia nos dias uteis

parametros "Nao agendar inadimplentes":
    da carteira("carteira televendas")
    quando inadimplentes
    nao agendar
'''


def _codigos(texto: str) -> set[str]:
    return {p.codigo for p in GobrValidator().validar(texto).problemas}


class AcervoTests(unittest.TestCase):
    # Erros reais encontrados no acervo; qualquer mudança aqui deve ser
    # investigada (regressão no validador ou arquivo alterado).
    ESPERADOS = {
        "regra_032.gobr": {"E008"},   # parêntese aberto em ramo de atividade igual(
        "regra_047.gobr": {"A004"},   # cabeçalho parametros com recuo
        "regra_080.gobr": {"E006"},   # nome de parametros repetido
        "regra_083.gobr": {"E006"},   # nome de parametros repetido
        "regra_140.gobr": {"E101"},   # caractere \x01 invisível
        "regra_221.gobr": {"E008"},   # aspa sobrando no fim da preferência
        "V1Vidros-RegraComercial.gobr": {"E001"},  # sem version 1.0
    }

    def test_somente_anomalias_conhecidas(self):
        arquivos = sorted(glob.glob("data/referencias/*.gobr")) + sorted(glob.glob("data/exemplos/*.gobr"))
        if not arquivos:
            self.skipTest("Acervo nao incluido")
        validador = GobrValidator()
        encontrados = {}
        for arquivo in arquivos:
            relatorio = validador.validar_arquivo(Path(arquivo))
            if relatorio.problemas:
                encontrados[Path(arquivo).name] = {p.codigo for p in relatorio.problemas}
        self.assertEqual(self.ESPERADOS, encontrados)


class ValidadorTests(unittest.TestCase):
    def test_arquivo_valido(self):
        relatorio = GobrValidator().validar(VALIDO)
        self.assertTrue(relatorio.ok, relatorio.como_texto())
        self.assertEqual([], relatorio.problemas)
        self.assertEqual({"Ativos", "Nao agendar inadimplentes"}, set(relatorio.parametros))

    def test_sem_version(self):
        self.assertIn("E001", _codigos(VALIDO.replace("version 1.0\n", "")))

    def test_linha_sem_indentacao(self):
        self.assertIn("E002", _codigos(VALIDO.replace("    frequencia de 14 dias", "frequencia de 14 dias")))

    def test_ordem_das_categorias(self):
        texto = VALIDO + '\ntime "outro":\n    vendedores todos\n'
        self.assertIn("E005", _codigos(texto))

    def test_nome_repetido(self):
        texto = VALIDO + '\nparametros "Ativos":\n    da carteira("carteira televendas")\n    nao agendar\n'
        self.assertIn("E006", _codigos(texto))

    def test_sem_time(self):
        texto = 'version 1.0\n\nparametros "x":\n    do time("a")\n    nao agendar\n'
        self.assertTrue({"E007", "E020"} <= _codigos(texto))

    def test_parenteses_e_aspas(self):
        self.assertIn("E008", _codigos(VALIDO.replace('carteirafixa("televendas")', 'carteirafixa("televendas"')))
        self.assertIn("E008", _codigos(VALIDO.replace("preferencia nos dias uteis", 'preferencia nos dias uteis"')))

    def test_corpo_de_time(self):
        self.assertIn("E010", _codigos(VALIDO.replace("vendedores todostelevendas ativados", "vendedores alguns")))

    def test_corpo_de_carteira(self):
        self.assertIn("E011", _codigos(VALIDO.replace('clientes carteirafixa("televendas")', "clientes x")))

    def test_referencia_inexistente_com_dica(self):
        texto = VALIDO.replace('da carteira("carteira televendas")\n    quando recencia',
                               'da carteira("televendas")\n    quando recencia')
        relatorio = GobrValidator().validar(texto)
        e020 = [p for p in relatorio.erros if p.codigo == "E020"]
        self.assertEqual(1, len(e020))
        self.assertIn('do time("televendas")', e020[0].mensagem)

    def test_referencia_definida_depois(self):
        texto = VALIDO.replace('clientes carteirafixa("televendas")', 'clientes carteirafixa("depois")')
        texto = texto.replace('carteira "carteira televendas":', 'time "depois":\n    vendedores todos\n\ncarteira "carteira televendas":', 1)
        self.assertNotIn("E021", _codigos(texto))
        tardio = VALIDO.replace('clientes carteirafixa("televendas")', 'clientes carteirafixa("depois")')
        tardio += '\ntime "depois":\n    vendedores todos\n'
        self.assertIn("E021", _codigos(tardio))

    def test_clausula_desconhecida(self):
        self.assertIn("E030", _codigos(VALIDO.replace("    nao agendar\n", "    nao agendar\n    agendar sempre\n")))

    def test_sem_escopo_e_sem_acao(self):
        texto = VALIDO.replace('    da carteira("carteira televendas")\n    quando inadimplentes\n    nao agendar\n',
                               "    quando inadimplentes\n")
        self.assertTrue({"E031", "E032"} <= _codigos(texto))

    def test_nao_agendar_exclusivo(self):
        self.assertIn("E033", _codigos(VALIDO.replace("    nao agendar\n", "    nao agendar\n    frequencia de 7 dias\n")))

    def test_frequencia_invalida(self):
        self.assertIn("E034", _codigos(VALIDO.replace("frequencia de 14 dias", "frequencia quinzenalmente")))

    def test_caractere_invisivel(self):
        self.assertIn("E101", _codigos(VALIDO.replace("adimplentes", "adimplentes\x01")))

    def test_condicao_fora_do_vocabulario_tem_sugestao(self):
        relatorio = GobrValidator().validar(VALIDO.replace("e adimplentes", "e adimplente pagador"))
        self.assertTrue(relatorio.ok)  # Aviso não bloqueia.
        aviso = [p for p in relatorio.avisos if p.codigo == "A001"]
        self.assertEqual(1, len(aviso))
        self.assertIn("adimplentes", aviso[0].sugestoes)

    def test_modificador_fora_da_gramatica(self):
        self.assertIn("A002", _codigos(VALIDO.replace("preferencia nos dias uteis", "preferencia nas ferias")))

    def test_sem_choque_resolve_referencia(self):
        com_choque = VALIDO.replace("    preferencia nos dias uteis\n",
                                    '    preferencia nos dias uteis\n    sem choque no time("campo")\n')
        self.assertIn("E020", _codigos(com_choque))
        self.assertNotIn("E020", _codigos(com_choque.replace('time("campo")', 'time("televendas")')))

    def test_relatorio_em_texto(self):
        texto = GobrValidator().validar(VALIDO.replace("version 1.0\n", "")).como_texto()
        self.assertTrue(texto.startswith("GOBR inválido: 1 erro(s)"))
        self.assertIn("[ERRO E001]", texto)


class GramaticaEParserTests(unittest.TestCase):
    def test_frequencias_do_manual(self):
        for valor in ("de 14 dias", "semanal()", "quinzenal('01-02-2026')", "do vendedor ou de qualquer vendedor ou de 35 dias",
                      "do historico ou no minimo de 7 dias e no maximo de 30 dias", "inativos da empresa", "remover"):
            with self.subTest(valor=valor):
                self.assertTrue(grammar.frequencia_valida(valor))
        self.assertFalse(grammar.frequencia_valida("toda semana"))

    def test_continuacao_de_frequencia(self):
        regra = GobrParser().parse('parametros "x":\n    do time("t")\n    frequencia do historico\n        ou no maximo de 30 dias')
        self.assertEqual("do historico ou no maximo de 30 dias", regra.frequencia)
        self.assertEqual([], regra.condicoes)

    def test_continuacao_de_preferencia(self):
        regra = GobrParser().parse(
            'parametros "x":\n    do time("t")\n    frequencia de 7 dias\n    preferencia nos dias uteis\n        e no sabado')
        self.assertEqual(["preferencia nos dias uteis\ne no sabado"], regra.modificadores)
        self.assertEqual([], regra.condicoes)

    def test_filtros_temporais_do_manual(self):
        regra = GobrParser().parse(
            'parametros "x":\n    do time("t")\n    no dia 10\n    no segundo semestre\n    nos dias uteis\n    nao agendar')
        self.assertEqual(["no dia 10", "no segundo semestre", "nos dias uteis"], regra.filtros_temporais)
        self.assertEqual([], regra.desconhecidos)

    def test_restrito(self):
        regra = GobrParser().parse('parametros "x":\n    do time("t")\n    frequencia mensal()\n    restrito a partir da segunda semana')
        self.assertEqual(["restrito a partir da segunda semana"], regra.modificadores)
        self.assertTrue(grammar.modificador_valido(regra.modificadores[0]))


class VocabularioTests(unittest.TestCase):
    def test_modelo_idempotente(self):
        modelo = modelo_condicao('ramo de atividade igual("1", "12") e recencia de positivação em 60 dias')
        self.assertEqual('ramo de atividade igual("X") e recencia de positivacao em N dias', modelo)
        self.assertEqual(modelo, modelo_condicao(modelo))

    def test_composicao_com_e_interno(self):
        vocabulario = VocabularioCondicoes()  # Só padrões do manual.
        self.assertTrue(vocabulario.conhece("recencia de positivacao entre 30 dias e 60 dias e inadimplentes"))
        self.assertTrue(vocabulario.conhece("inativos para o vendedor ou sem venda"))
        self.assertFalse(vocabulario.conhece("clientes que compraram muito"))


if __name__ == "__main__":
    unittest.main()
