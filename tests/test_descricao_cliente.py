import json
import tempfile
import unittest
from pathlib import Path

from src.explicacao.descricao_cliente import AgenteDescricaoCliente, conferir
from src.finetuning.dataset import montar
from src.pipelines import gerar_descricoes_cliente as lote
from src.rag.exemplos import documentos_de_exemplos

RAIZ = Path(__file__).resolve().parents[1]
ATIVOS = 'parametros "Ativos":\n    da carteira("carteira campo")\n    quando recencia de positivacao em 60 dias\n    frequencia de 14 dias'
INATIVOS = ('parametros "I":\n    do time("t")\n    quando inativos para o vendedor\n'
            "    frequencia de inativos da empresa")
RAMO = ('parametros "R":\n    do time("t")\n    quando ramo de atividade igual("MATCON")\n    frequencia de 30 dias\n'
        "    preferencia na segunda")


class LLMRoteiro:
    def __init__(self, *respostas):
        self.respostas, self.chamadas = list(respostas), []

    def invoke(self, mensagens):
        self.chamadas.append(mensagens)
        return self.respostas.pop(0)


class ConferenciaTests(unittest.TestCase):
    def test_aprova_descricao_fiel(self):
        self.assertEqual([], conferir(ATIVOS, "Clientes com compra nos últimos 60 dias. Atender a cada 14 dias."))
        self.assertEqual([], conferir(RAMO, 'Clientes do ramo "MATCON". Atender a cada 30 dias, de preferência '
                                            "na segunda-feira."))

    def test_numero_nome_e_acao(self):
        problemas = conferir(ATIVOS, 'Clientes da carteira "carteira campo" com compra recente. Não agendar.')
        texto = "\n".join(problemas)
        self.assertIn("Falta o número 60", texto)
        self.assertIn("não é um valor da regra", texto)
        self.assertIn("não pode dizer", texto)

    def test_frequencia_dinamica_precisa_ser_descrita(self):
        self.assertTrue(conferir(INATIVOS, "Clientes inativos para o vendedor. Atender com a frequência do vendedor."))
        self.assertEqual([], conferir(INATIVOS, "Clientes inativos para o vendedor. Atender com a frequência dos "
                                                "inativos da empresa."))

    def test_valor_e_modificador(self):
        texto = "\n".join(conferir(RAMO, "Clientes do ramo de construção. Atender a cada 30 dias."))
        self.assertIn('Falta o valor "MATCON"', texto)
        self.assertIn("preferencia na segunda", texto)


class AgenteTests(unittest.TestCase):
    def test_corrige_uma_vez(self):
        llm = LLMRoteiro("Clientes ativos. Atender a cada 15 dias.",
                         "Clientes com compra nos últimos 60 dias. Atender a cada 14 dias.")
        r = AgenteDescricaoCliente(llm).descrever(ATIVOS)
        self.assertEqual(("llm_corrigido", "Clientes com compra nos últimos 60 dias. Atender a cada 14 dias."),
                         (r.metodo, r.texto))
        self.assertIn("Falta o número 60", llm.chamadas[1][-1][1])
        papeis = [p for p, _ in llm.chamadas[0]]
        self.assertEqual("system", papeis[0])
        self.assertEqual(["human", "ai"] * 6 + ["human"], papeis[1:])   # 6 exemplos das tabelas reais

    def test_desiste(self):
        r = AgenteDescricaoCliente(LLMRoteiro("x", "y")).descrever(ATIVOS)
        self.assertEqual(("falhou", None), (r.metodo, r.texto))


class LoteTests(unittest.TestCase):
    def test_lote_retoma_e_replica_para_blocos_iguais(self):
        with tempfile.TemporaryDirectory() as tmp:
            refs, saida = Path(tmp) / "refs", Path(tmp) / "saida"
            refs.mkdir()
            (refs / "a.gobr").write_text("version 1.0\n\n" + ATIVOS + "\n\n" + ATIVOS.replace('"Ativos"', '"Copia"')
                                         + "\n", encoding="utf-8")
            respostas = ["Clientes com compra nos últimos 60 dias. Atender a cada 14 dias."]
            criar = lambda: AgenteDescricaoCliente(LLMRoteiro(*respostas))  # noqa: E731
            contagem = lote.run(refs, saida, modelo="m", agente_factory=criar, verificar_servico=lambda: None)
            self.assertEqual({"llm": 1}, contagem)
            linhas = [json.loads(l) for l in (saida / "descricoes_cliente.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(["a_001", "a_002"], [l["id"] for l in linhas])
            self.assertEqual("a_001", linhas[1]["reaproveitada_de"])
            self.assertEqual({}, lote.run(refs, saida, modelo="m", agente_factory=criar, verificar_servico=lambda: None))

    def test_ollama_fora_do_ar_interrompe(self):
        class Caiu:
            def descrever(self, gobr):
                raise ConnectionError("Connection refused")

        with tempfile.TemporaryDirectory() as tmp:
            refs = Path(tmp) / "refs"
            refs.mkdir()
            blocos = "\n\n".join(ATIVOS.replace('"Ativos"', f'"A{i}"').replace("60", str(60 + i)) for i in range(4))
            (refs / "a.gobr").write_text("version 1.0\n\n" + blocos + "\n", encoding="utf-8")
            with self.assertRaises(lote.OllamaIndisponivel):
                lote.run(refs, Path(tmp) / "saida", modelo="m", agente_factory=Caiu, verificar_servico=lambda: None)
            item = json.loads((Path(tmp) / "saida" / "items" / "a_001.json").read_text(encoding="utf-8"))
            self.assertEqual(0, item["tentativas"])           # falha de infra não conta tentativa


class UsoTests(unittest.TestCase):
    def test_rag_e_dataset_usam_a_descricao(self):
        regras = [{"id": "r1", "nome": "Ativos", "gobr": ATIVOS}]
        doc = documentos_de_exemplos(regras, {}, {"r1": "Clientes com compra nos últimos 60 dias."})[0]
        self.assertIn("Clientes com compra nos últimos 60 dias.", doc["text"])
        self.assertEqual("Clientes com compra nos últimos 60 dias.", doc["metadata"]["descricao_cliente"])

        conjuntos = montar(RAIZ / "data" / "referencias", RAIZ / "data" / "exemplos", None)
        self.assertEqual({"pdf"}, {p["fonte_descricao"] for p in conjuntos["real"]})
        maxlub = next(p for p in conjuntos["real"] if p["id"] == "RegraMaxlub_005")
        self.assertEqual("Clientes com compra entre 0 e 60 dias. Atender a cada 14 dias.", maxlub["descricao"])


if __name__ == "__main__":
    unittest.main()
