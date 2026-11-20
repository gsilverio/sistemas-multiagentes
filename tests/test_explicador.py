"""Explicador (etapa 2): glossário, rascunho, guard, agente e lote.

Roda sem Ollama: o LLM é substituído por um objeto falso.
"""
import json
import tempfile
import unittest
from pathlib import Path

from src.agents.explicador import AgenteExplicador
from src.explicacao import glossario
from src.explicacao.guarda import verificar
from src.explicacao.rascunho import rascunho_de_gobr
from src.gobr.extractor import GobrRuleExtractor
from src.gobr.parser import GobrParser
from src.pipelines.generate_explanations import (
    OllamaIndisponivel, ReferenceBatchRunner, falha_de_infra, load_rules, read_previous,
)

BLOCO = '''parametros "Ativos":
    da carteira("carteira televendas")
    quando recencia de positivacao em 60 dias
        e adimplentes
    frequencia de 14 dias
    preferencia nos dias uteis'''

BOA = ('Os clientes da carteira "carteira televendas" que compraram nos últimos 60 dias e estão '
       'adimplentes devem ser atendidos a cada 14 dias, com preferência de visita nos dias úteis.')


class FakeLLM:
    def __init__(self, respostas):
        self.respostas = respostas if isinstance(respostas, list) else list(respostas)
        self.chamadas = []

    def invoke(self, mensagens):
        self.chamadas.append(mensagens)
        resposta = self.respostas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return type("Resposta", (), {"content": resposta})()


def _acervo():
    pasta = Path("data/referencias")
    return GobrRuleExtractor().extract_directory(pasta) if pasta.exists() else []


class GlossarioTests(unittest.TestCase):
    def test_cobertura_do_acervo(self):
        blocos = _acervo()
        if not blocos:
            self.skipTest("Acervo nao incluido")
        parser = GobrParser()
        sem_traducao = set()
        for bloco in blocos:
            regra = parser.parse(bloco.gobr)
            trechos = [glossario.traduzir_condicao(c.texto) for c in regra.condicoes]
            trechos += [glossario.traduzir_modificador(m) for m in regra.modificadores]
            trechos += [glossario.traduzir_filtro(f) for f in regra.filtros_temporais]
            if regra.frequencia:
                trechos.append(glossario.traduzir_frequencia(regra.frequencia))
            if any(not t.conhecida for t in trechos):
                sem_traducao.add(bloco.arquivo_origem)
        # Só as duas anomalias do acervo (parêntese aberto e caractere \x01).
        self.assertEqual({"regra_032.gobr", "regra_140.gobr"}, sem_traducao)

    def test_condicoes(self):
        casos = {
            "sem venda": "sem venda (que nunca compraram)",
            "fora da recencia de positivacao global em 90 dias":
                "cuja última compra (positivação) com qualquer vendedor da empresa não ocorreu nos últimos 90 dias",
            "recencia de positivacao do vendedor entre 30 dias e 60 dias":
                "cuja última compra (positivação) com o próprio vendedor da carteira ocorreu entre 30 dias e 60 dias atrás",
            'ramo de atividade igual("MATCON","FERRAGISTA")':
                'com ramo de atividade igual a um destes valores: "MATCON", "FERRAGISTA"',
            "no ciclo de vida prospect com a empresa":
                "que estão no ciclo de vida prospect com a empresa (ainda não compraram)",
        }
        for gobr, esperado in casos.items():
            with self.subTest(gobr=gobr):
                self.assertEqual(esperado, glossario.traduzir_condicao(gobr).frase)

    def test_condicao_composta_e_desconhecida(self):
        composta = glossario.traduzir_condicao("inativos para o vendedor ou sem venda")
        self.assertTrue(composta.conhecida)
        self.assertIn(" ou sem venda", composta.frase)
        self.assertEqual(1, glossario.conectores_ou("inativos para o vendedor ou sem venda"))
        desconhecida = glossario.traduzir_condicao("clientes muito bons")
        self.assertFalse(desconhecida.conhecida)
        self.assertIn('"clientes muito bons"', desconhecida.frase)

    def test_frequencias(self):
        self.assertEqual("a cada 14 dias", glossario.traduzir_frequencia("14 dias").frase)
        self.assertEqual("com frequência semanal (a cada 7 dias)", glossario.traduzir_frequencia("semanal()").frase)
        # "frequencia de inativos da empresa" é a frequência média dos inativos, não um público.
        self.assertEqual("com a frequência média dos clientes inativos da empresa",
                         glossario.traduzir_frequencia("inativos da empresa").frase)
        cascata = glossario.traduzir_frequencia("do vendedor ou de qualquer vendedor ou de 35 dias").frase
        self.assertIn("se ela não existir, com a frequência configurada por qualquer vendedor", cascata)
        self.assertIn("se também não existir, a cada 35 dias", cascata)
        limites = glossario.traduzir_frequencia("do historico ou no minimo de 7 dias e no maximo de 30 dias").frase
        self.assertIn("intervalo de no mínimo 7 dias e no máximo 30 dias", limites)

    def test_modificadores_e_filtros(self):
        self.assertEqual("Preferência de visita nos dias úteis (segunda a sexta) e no sábado; "
                         "nos demais dias o cliente não é visitado.",
                         glossario.traduzir_modificador("preferencia nos dias uteis e no sabado").frase)
        self.assertIn("maior na segunda-feira e maior na terça-feira",
                      glossario.traduzir_modificador("preferencia maior na segunda\ne maior na terca").frase)
        self.assertIn('com a do time "campo" (considerando apenas clientes roteirizáveis)',
                      glossario.traduzir_modificador('sem choque na rota com time("campo")').frase)
        self.assertEqual("O cliente só pode entrar na agenda a partir da segunda semana do mês.",
                         glossario.traduzir_modificador("restrito a partir da segunda semana").frase)
        self.assertEqual("a partir do dia 16 do mês", glossario.traduzir_filtro("a partir do dia 16").frase)
        self.assertFalse(glossario.traduzir_modificador("preferencia nas ferias").conhecida)


class RascunhoTests(unittest.TestCase):
    def test_texto_completo(self):
        _, rascunho = rascunho_de_gobr(BLOCO)
        self.assertEqual(
            'Os clientes da carteira "carteira televendas" cuja última compra (positivação) com algum vendedor '
            'do grupo ocorreu nos últimos 60 dias e que estão adimplentes (sem pendências financeiras em aberto) '
            'devem ser atendidos a cada 14 dias. Preferência de visita nos dias úteis (segunda a sexta); '
            'nos demais dias o cliente não é visitado.', rascunho.texto)
        self.assertEqual(["escopo", "condicao", "condicao", "frequencia", "modificador"],
                         [f.tipo for f in rascunho.fatos])
        self.assertEqual(["carteira televendas"], rascunho.nomes_citados)

    def test_rascunho_aprovado_em_todo_o_acervo(self):
        blocos = _acervo()
        if not blocos:
            self.skipTest("Acervo nao incluido")
        for bloco in blocos:
            _, rascunho = rascunho_de_gobr(bloco.gobr)
            if rascunho.desconhecidos:
                continue
            resultado = verificar(rascunho, rascunho.texto, gobr=bloco.gobr)
            if not resultado.aprovado:
                self.fail(f"{bloco.id}: {resultado.problemas}")

    def test_nome_empresa_toda_fica_literal(self):
        _, rascunho = rascunho_de_gobr('parametros "x":\n    do time("toda empresa")\n    frequencia de 15 dias')
        self.assertTrue(rascunho.texto.startswith('Os clientes do time "toda empresa" devem ser atendidos'))


class GuardaTests(unittest.TestCase):
    def setUp(self):
        _, self.rascunho = rascunho_de_gobr(BLOCO)

    def _problemas(self, texto, rascunho=None):
        return verificar(rascunho or self.rascunho, texto, gobr=BLOCO).problemas

    def test_texto_bom(self):
        self.assertEqual([], self._problemas(BOA))

    def test_numero_inventado_e_numero_perdido(self):
        # Erro real da linha de base (regra_005_003 citava "14 dias" inexistente).
        problemas = self._problemas(BOA.replace("60 dias", "90 dias"))
        self.assertIn("Número da regra ausente no texto: 60.", problemas)
        self.assertIn("Número que não existe na regra: 90.", problemas)

    def test_milhar_formatado_e_aceito(self):
        _, rascunho = rascunho_de_gobr('parametros "x":\n    do time("t")\n'
                                       '    quando media faturada no ano com a empresa maior que R$ 50000\n'
                                       '    frequencia de 7 dias')
        texto = ('Os clientes do time "t" com faturamento médio mensal no ano acima de R$ 50.000,00 '
                 'devem ser atendidos a cada 7 dias.')
        self.assertTrue(verificar(rascunho, texto).aprovado)

    def test_nome_alterado(self):
        self.assertIn('Nome não preservado literalmente: "carteira televendas".',
                      self._problemas(BOA.replace('da carteira "carteira televendas"', 'do time "televendas"')))

    def test_condicao_invertida(self):
        problemas = self._problemas(BOA.replace("adimplentes", "inadimplentes"))
        self.assertIn('Condicao não preservado(a): "adimplentes".', problemas)

    def test_ou_perdido(self):
        gobr = 'parametros "x":\n    do time("t")\n    quando inativos para o vendedor ou sem venda\n    frequencia de 7 dias'
        _, rascunho = rascunho_de_gobr(gobr)
        texto = 'Os clientes do time "t" inativos para o vendedor e sem venda devem ser atendidos a cada 7 dias.'
        self.assertTrue(any("`ou`" in p for p in verificar(rascunho, texto).problemas))

    def test_nao_agendar(self):
        gobr = 'parametros "x":\n    do time("televendas")\n    quando inadimplentes\n    nao agendar'
        _, rascunho = rascunho_de_gobr(gobr)
        bom = 'Os clientes do time "televendas" que estão inadimplentes não devem ser agendados.'
        self.assertTrue(verificar(rascunho, bom).aprovado)
        estado = 'Os clientes do time "televendas" inadimplentes não podem estar agendados.'
        self.assertFalse(verificar(rascunho, estado).aprovado)
        self.assertIn("O texto proíbe agendar, mas a regra não tem `nao agendar`.",
                      self._problemas(BOA + " Eles não devem ser agendados."))

    def test_justificativa_inventada_reprovada(self):
        # Saídas reais do gemma3:12b no primeiro teste no Colab (regra_001_001 e 001_002).
        gobr = ('parametros "x":\n    da carteira("empresa toda")\n    quando inativos para o vendedor\n'
                '    frequencia de inativos da empresa')
        _, rascunho = rascunho_de_gobr(gobr)
        colab = ('Essa regra define como vamos abordar os clientes da carteira "empresa toda" que estão inativos '
                 'para o vendedor. Ela garante que esses clientes sejam contatados com a mesma frequência que a '
                 'média dos inativos da empresa. Assim, asseguramos um acompanhamento consistente, mesmo para '
                 'aqueles que não compram diretamente com o vendedor há algum tempo. O objetivo é manter o '
                 'relacionamento e identificar possíveis oportunidades de retomada.')
        problemas = verificar(rascunho, colab, gobr=gobr).problemas
        self.assertTrue(any("justificativa" in p for p in problemas))
        bom = ('Os clientes da carteira "empresa toda" que estão inativos para o vendedor devem ser atendidos '
               'com a frequência média dos clientes inativos da empresa.')
        self.assertEqual([], verificar(rascunho, bom, gobr=gobr).problemas)
        longo = bom + " Esses clientes são acompanhados de perto pela equipe comercial em todas as visitas." * 4
        self.assertTrue(any("caracteres" in p for p in verificar(rascunho, longo, gobr=gobr).problemas))

    def test_mencao_a_implementacao(self):
        self.assertTrue(any("implementação" in p for p in self._problemas("Esta regra GOBR: " + BOA)))


class AgenteTests(unittest.TestCase):
    def test_primeira_resposta_aprovada(self):
        llm = FakeLLM(["<think>pensando</think>\n" + BOA])
        explicacao = AgenteExplicador(llm).explicar(BLOCO)
        self.assertEqual(("llm", True, BOA), (explicacao.metodo, explicacao.aprovada, explicacao.texto))
        self.assertEqual(1, len(llm.chamadas))
        sistema, humano = llm.chamadas[0]
        self.assertEqual("system", sistema[0])
        self.assertIn(BLOCO.strip(), humano[1])
        self.assertIn("Rascunho:", humano[1])

    def test_correcao_recebe_os_problemas(self):
        llm = FakeLLM([BOA.replace("14 dias", "15 dias"), BOA])
        explicacao = AgenteExplicador(llm).explicar(BLOCO)
        self.assertEqual("llm_corrigido", explicacao.metodo)
        self.assertEqual(2, len(explicacao.tentativas_llm))
        ultima = llm.chamadas[1][-1][1]
        self.assertIn("Número que não existe na regra: 15.", ultima)

    def test_cai_no_rascunho_depois_de_duas_falhas(self):
        llm = FakeLLM(["texto ruim", "outro texto ruim"])
        explicacao = AgenteExplicador(llm).explicar(BLOCO)
        self.assertEqual("rascunho", explicacao.metodo)
        self.assertTrue(explicacao.aprovada)
        self.assertEqual(explicacao.rascunho, explicacao.texto)
        self.assertTrue(explicacao.problemas)

    def test_sem_llm_e_clausula_desconhecida_nao_chamam_modelo(self):
        llm = FakeLLM([])
        self.assertEqual("rascunho", AgenteExplicador(llm, usar_llm=False).explicar(BLOCO).metodo)
        desconhecida = AgenteExplicador(llm).explicar('parametros "x":\n    do time("t")\n    agendar sempre')
        self.assertEqual(("rascunho", False), (desconhecida.metodo, desconhecida.aprovada))
        self.assertEqual([], llm.chamadas)


class LoteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.input, self.out = root / "regras.jsonl", root / "saida"
        self.rules = [{"id": f"regra_001_{i:03d}", "nome": f"Caso {i}", "arquivo_origem": "regra_001.gobr",
                       "gobr": f'parametros "Caso {i}":\n    da carteira("teste")\n    frequencia de {i} dias'}
                      for i in range(1, 5)]
        self._gravar(self.rules)
        self.respostas = []
        self.llm = FakeLLM(self.respostas)
        self.runner = ReferenceBatchRunner(explicador_factory=lambda: AgenteExplicador(self.llm))

    def _gravar(self, rules):
        self.input.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rules), encoding="utf-8")

    def _rodar(self, runner=None, **kwargs):
        kwargs.setdefault("model", "gemma3:12b")
        return (runner or self.runner).run(input_path=self.input, output_dir=self.out, num_ctx=4096,
                                           versao="v-teste", **kwargs)

    def _boa(self, i):
        return f'Os clientes da carteira "teste" devem ser atendidos a cada {i} dias.'

    def test_dry_run_nao_chama_modelo_nem_grava(self):
        stats = self._rodar(limit=2, dry_run=True)
        self.assertEqual(2, stats["simuladas"])
        self.assertFalse(self.out.exists())
        self.assertEqual([], self.llm.chamadas)

    def test_retomada_e_consolidacao(self):
        self.respostas += [self._boa(1), self._boa(2)]
        self.assertEqual(2, self._rodar(limit=2)["concluidas_nesta_execucao"])
        antes = (self.out / "items/regra_001_001.json").read_bytes()
        self.respostas += [self._boa(3), "ruim", "ruim"]
        stats = self._rodar(limit=2)
        self.assertEqual((2, 2), (stats["concluidas_nesta_execucao"], stats["ignoradas_concluidas"]))
        self.assertEqual({"llm": 1, "rascunho": 1}, stats["metodos_nesta_execucao"])
        self.assertEqual(antes, (self.out / "items/regra_001_001.json").read_bytes())
        linhas = [json.loads(l) for l in (self.out / "explicacoes.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(["regra_001_001", "regra_001_002", "regra_001_003", "regra_001_004"], [l["id"] for l in linhas])
        self.assertEqual("rascunho", linhas[3]["metodo"])
        self.assertTrue(all(l["aprovada"] for l in linhas))

    def test_falha_registrada_e_retentativa_limitada(self):
        self.respostas += [RuntimeError("Ollama indisponível")] * 2 + [self._boa(2)]
        self.assertEqual(1, self._rodar(limit=1, max_attempts=2)["falhas_nesta_execucao"])
        self.assertEqual(1, self._rodar(limit=1, max_attempts=2)["falhas_nesta_execucao"])
        item = read_previous(self.out / "items/regra_001_001.json")
        self.assertEqual(("failed", 2), (item["status"], item["attempt"]))
        stats = self._rodar(limit=1, max_attempts=2)
        self.assertEqual((1, 1), (stats["ignoradas_tentativas"], stats["concluidas_nesta_execucao"]))

    def test_outro_modelo_invalida_itens(self):
        self.respostas += [self._boa(1), self._boa(1)]
        self._rodar(limit=1)
        stats = self._rodar(limit=1, model="qwen3:8b")
        self.assertEqual(1, stats["concluidas_nesta_execucao"])
        self.assertEqual("qwen3:8b", read_previous(self.out / "items/regra_001_001.json")["model"])

    def test_sem_llm_e_bloqueio_por_clausula_desconhecida(self):
        self.rules[0]["gobr"] = 'parametros "Caso 1":\n    da carteira("teste")\n    agendar sempre'
        self._gravar(self.rules)
        stats = self._rodar(ReferenceBatchRunner(), limit=0, sem_llm=True)
        self.assertEqual((3, 1), (stats["concluidas_nesta_execucao"], stats["bloqueadas_nesta_execucao"]))
        self.assertEqual("sem-llm", stats["model"])
        self.assertEqual(3, len((self.out / "explicacoes.jsonl").read_text(encoding="utf-8").splitlines()))

    def test_blocos_repetidos_reaproveitam_explicacao(self):
        repetido = dict(self.rules[0], id="regra_002_001", nome="Outro nome",
                        gobr=self.rules[0]["gobr"].replace('"Caso 1"', '"Outro nome"'))
        self._gravar([self.rules[0], repetido, self.rules[1]])
        self.respostas += [self._boa(1), self._boa(2)]
        stats = self._rodar(limit=2)
        self.assertEqual((2, 1), (stats["concluidas_nesta_execucao"], stats["reaproveitadas_nesta_execucao"]))
        self.assertEqual(2, len(self.llm.chamadas))
        item = read_previous(self.out / "items/regra_002_001.json")
        self.assertEqual(("regra_001_001", "Outro nome"), (item["reaproveitada_de"], item["explicacao"]["nome"]))
        self.assertEqual(self._boa(1), item["explicacao"]["texto"])
        # Retomada: o item reaproveitado é tratado como concluído.
        self.assertEqual(3, self._rodar(limit=2)["ignoradas_concluidas"])

    def test_ollama_fora_do_ar_interrompe_sem_gastar_tentativas(self):
        class ConnectError(Exception):  # mesmo nome da exceção do httpx
            pass

        mais = [dict(self.rules[0], id=f"regra_009_{i:03d}", gobr=f'parametros "x":\n    da carteira("teste")\n'
                     f'    frequencia de {i + 10} dias') for i in range(4)]
        self._gravar(self.rules + mais)
        self.respostas += [self._boa(1)] + [ConnectError("[Errno 111] Connection refused")] * 3
        with self.assertRaises(OllamaIndisponivel):
            self._rodar(limit=0)
        itens = {p.stem: read_previous(p) for p in (self.out / "items").glob("*.json")}
        self.assertEqual(4, len(itens))  # parou na 3ª falha seguida, não percorreu o resto
        falhas = [i for i in itens.values() if i["status"] == "failed"]
        self.assertEqual([0, 0, 0], [i["attempt"] for i in falhas])
        self.assertIn("interrompido", json.loads((self.out / "batch_summary.json").read_text(encoding="utf-8")))
        # Com o servidor de volta, as que falharam são refeitas mesmo com max_attempts=1.
        self.respostas += [self._boa(2), self._boa(3), self._boa(4)] + [self._boa(i + 10) for i in range(4)]
        stats = self._rodar(limit=0, max_attempts=1)
        self.assertEqual((7, 0), (stats["concluidas_nesta_execucao"], stats["ignoradas_tentativas"]))

    def test_checagem_previa_do_servidor(self):
        def fora():
            raise OllamaIndisponivel("sem servidor")
        with self.assertRaises(OllamaIndisponivel):
            self._rodar(limit=0, verificar_servico=fora)
        self.assertFalse(self.out.exists())
        self.assertTrue(falha_de_infra("ConnectError") and falha_de_infra("ReadTimeout"))
        self.assertFalse(falha_de_infra("ValueError", "regra estranha"))

    def test_rejeita_ids_duplicados(self):
        self._gravar([self.rules[0], self.rules[0]])
        with self.assertRaisesRegex(ValueError, "duplicado"):
            load_rules(self.input)


if __name__ == "__main__":
    unittest.main()
