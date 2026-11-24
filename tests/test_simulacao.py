import unittest

from src.gobr.simulacao import NAO_AGENDAR, acerto_comportamento

CAB = 'version 1.0\n\ntime "televendas":\n    vendedores todostelevendas ativados\n\n'


def gobr(*blocos: tuple[str, list[str]]) -> str:
    return CAB + "\n".join(f'parametros "{nome}":\n    do time("televendas")\n'
                           + "".join(f"    {l}\n" for l in linhas) for nome, linhas in blocos)


ESPERADO = gobr(
    ("Baixados", ["quando status na receita federal definido como baixada", "nao agendar"]),
    ("Ativos", ["quando recencia de positivacao em 30 dias", "frequencia de 14 dias"]),
    ("Risco", ["quando recencia de positivacao em 45 dias", "e fora da recencia de positivacao em 30 dias",
               "frequencia de 7 dias"]),
    ("Inativos", ["quando fora da recencia de positivacao em 45 dias", "frequencia de 14 dias"]),
    ("Prospects", ["quando no ciclo de vida prospect com a empresa", "frequencia de 7 dias"]),
)


class SimulacaoTests(unittest.TestCase):
    def test_igual_acerta_tudo(self):
        r = acerto_comportamento(ESPERADO, ESPERADO)
        self.assertEqual(1.0, r["acerto"])
        self.assertEqual([], r["erros_comportamento"])

    def test_forma_equivalente_conta_como_certa(self):
        # "entre 31 e 45" em vez de "em 45 e fora de 30"; "semanal()" em vez de "7 dias";
        # prospects depois do bloqueio, em outra posição: o tratamento dos clientes é o mesmo.
        gerado = gobr(
            ("B", ["quando status na receita federal definido como baixada", "nao agendar"]),
            ("P", ["quando no ciclo de vida prospect com a empresa", "frequencia semanal()"]),
            ("A", ["quando recencia de positivacao em 30 dias", "frequencia de 14 dias"]),
            ("R", ["quando recencia de positivacao entre 31 dias e 45 dias", "frequencia de 7 dias"]),
            ("I", ["quando fora da recencia de positivacao em 45 dias", "frequencia de 14 dias"]),
        )
        self.assertEqual(1.0, acerto_comportamento(gerado, ESPERADO)["acerto"])

    def test_faixa_errada_aparece_no_cliente_afetado(self):
        gerado = ESPERADO.replace("em 45 dias", "em 60 dias")
        r = acerto_comportamento(gerado, ESPERADO)
        self.assertLess(r["acerto"], 1.0)
        erro = r["erros_comportamento"][0]
        self.assertIn("52 dias", erro["cliente"])
        self.assertEqual(("a cada 14 dias", "a cada 7 dias"), (erro["esperado"], erro["gerado"]))

    def test_ordem_errada_muda_o_tratamento(self):
        # prospect antes do bloqueio: o prospect baixado na Receita passa a ser agendado
        gerado = gobr(
            ("P", ["quando no ciclo de vida prospect com a empresa", "frequencia de 7 dias"]),
            ("B", ["quando status na receita federal definido como baixada", "nao agendar"]),
            ("A", ["quando recencia de positivacao em 30 dias", "frequencia de 14 dias"]),
            ("R", ["quando recencia de positivacao em 45 dias", "frequencia de 7 dias"]),
            ("I", ["quando fora da recencia de positivacao em 45 dias", "frequencia de 14 dias"]),
        )
        r = acerto_comportamento(gerado, ESPERADO)
        self.assertLess(r["acerto"], 1.0)
        self.assertEqual(NAO_AGENDAR, r["erros_comportamento"][0]["esperado"])

    def test_preferencia_e_regra_faltando(self):
        com_preferencia = ESPERADO.replace("frequencia de 14 dias\n\nparametros \"Risco\"",
                                           "frequencia de 14 dias\n    preferencia na sexta\n\nparametros \"Risco\"")
        self.assertLess(acerto_comportamento(com_preferencia, ESPERADO)["acerto"], 1.0)
        sem_bloqueio = ESPERADO.replace("quando status na receita federal definido como baixada\n    nao agendar",
                                        "quando inadimplentes\n    nao agendar")
        self.assertLess(acerto_comportamento(sem_bloqueio, ESPERADO)["acerto"], 1.0)

    def test_sem_gobr_e_zero(self):
        self.assertEqual(0.0, acerto_comportamento(None, ESPERADO)["acerto"])

    def test_sem_regra_equivale_a_nao_agendar(self):
        esperado = gobr(("A", ["quando recencia de positivacao em 30 dias", "frequencia de 14 dias"]),
                        ("Demais", ["nao agendar"]))
        gerado = gobr(("A", ["quando recencia de positivacao em 30 dias", "frequencia de 14 dias"]))
        self.assertEqual(1.0, acerto_comportamento(gerado, esperado)["acerto"])


class ForaDoPedidoTests(unittest.TestCase):
    def test_condicao_que_o_usuario_nao_pediu_nao_conta(self):
        ativos = 'parametros "A":\n    do time("t")\n    quando recencia de positivacao em 60 dias\n    frequencia de 7 dias\n'
        baixada = "status na receita federal definido como baixada"
        esperado = f'version 1.0\n\nparametros "B":\n    do time("t")\n    quando {baixada}\n    nao agendar\n\n{ativos}'
        self.assertLess(acerto_comportamento(f"version 1.0\n\n{ativos}", esperado)["acerto"], 1)
        self.assertEqual(1.0, acerto_comportamento(f"version 1.0\n\n{ativos}", esperado, ignorar=[baixada])["acerto"])


if __name__ == "__main__":
    unittest.main()
