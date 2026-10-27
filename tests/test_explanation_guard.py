"""Revisão de invariantes do pipeline, sem métricas quantitativas de accuracy."""

import unittest

from src.parsers.gobr_parser import GobrParser
from src.validation.explanation_guard import ExplanationGuard


class ExplanationGuardTests(unittest.TestCase):
    def setUp(self):
        self.parser = GobrParser()
        self.guard = ExplanationGuard()

    def test_preserva_carteira_completa(self):
        rule = self.parser.parse('''parametros "Exemplo":
    da carteira("empresa toda")
    frequencia de 7 dias''')
        self.assertFalse(self.guard.inspect(rule, "Clientes da carteira empresa têm atendimento a cada 7 dias.").accepted)
        self.assertTrue(self.guard.inspect(rule, 'Clientes da carteira "empresa toda" têm atendimento a cada 7 dias.').accepted)

    def test_condicao_sem_venda_nao_vira_nunca_comprou(self):
        rule = self.parser.parse('''parametros "Exemplo":
    da carteira("empresa toda")
    quando sem venda
    frequencia de 14 dias''')
        invalid = 'Clientes da carteira "empresa toda" que nunca compraram terão atendimento a cada 14 dias.'
        self.assertFalse(self.guard.inspect(rule, invalid).accepted)
        valid = 'Os clientes da carteira "empresa toda" sem venda terão atendimento a cada 14 dias.'
        self.assertTrue(self.guard.inspect(rule, valid).accepted)

    def test_nao_agendar_e_acao_nao_estado(self):
        rule = self.parser.parse('''parametros "Exemplo":
    do time("televendas")
    quando é o cliente filial
    nao agendar''')
        invalid = 'Clientes do time televendas classificados como cliente filial não podem estar agendados.'
        self.assertFalse(self.guard.inspect(rule, invalid).accepted)
        valid = 'Clientes do time "televendas" classificados como cliente filial não devem ser agendados para atendimento.'
        self.assertTrue(self.guard.inspect(rule, valid).accepted)

    def test_cliente_filial_aceita_plural_apos_correcao_anterior(self):
        rule = self.parser.parse('''parametros "Exemplo":
    do time("televendas")
    quando é o cliente filial
    nao agendar''')
        explanation = (
            'Clientes do time "televendas" que forem filiais '
            'não devem ser agendados para atendimento.'
        )
        self.assertTrue(self.guard.inspect(rule, explanation).accepted)

    def test_conjuncao_escopos_e_condicao(self):
        rule = self.parser.parse('''parametros "Exemplo":
    da segmentacao("televendas ruptura de fornecedor")
    e da carteira("televendas consumidor")
    quando no ciclo de vida prospect com a empresa
    frequencia de 14 dias''')
        invalid = ('Clientes da segmentação televendas ruptura de fornecedor e clientes da '
                   'carteira televendas consumidor no ciclo de vida prospect com a empresa '
                   'terão atendimento a cada 14 dias.')
        self.assertFalse(self.guard.inspect(rule, invalid).accepted)
        valid = ('Os clientes da segmentação "televendas ruptura de fornecedor" e da '
                 'carteira "televendas consumidor" que estiverem no ciclo de vida prospect '
                 'com a empresa devem receber atendimento a cada 14 dias.')
        self.assertTrue(self.guard.inspect(rule, valid).accepted)

    def test_fallback_deterministico_preserva_proibicao(self):
        rule = self.parser.parse('''parametros "Exemplo":
    da carteira("empresa toda")
    quando inadimplentes
    nao agendar''')
        result = self.guard.conservative_fallback(rule)
        self.assertEqual(result, 'Os clientes da carteira "empresa toda" que forem inadimplentes não devem ser agendados para atendimento.')
        self.assertTrue(self.guard.inspect(rule, result).accepted)

    def test_semanal_preferencia_e_nome_exato(self):
        rule = self.parser.parse('''parametros "Exemplo":
    da carteira("empresa perola")
    frequencia semanal()
    preferencia nos dias uteis''')
        text = self.guard.conservative_fallback(rule)
        self.assertTrue(self.guard.inspect(rule, text).accepted)
        self.assertIn('empresa perola', text)
        self.assertIn('semanalmente', text)
        self.assertIn('dias úteis', text)

    def test_desconhecidos_impedem_aprovacao(self):
        rule = self.parser.parse('''parametros "Exemplo":
    da carteira("empresa toda")
    instrucao inventada''')
        result = self.guard.inspect(rule, 'Clientes da carteira "empresa toda".')
        self.assertFalse(result.accepted)
        self.assertIn('Cláusulas ainda não interpretadas', self.guard.conservative_fallback(rule))


if __name__ == '__main__':
    unittest.main()
