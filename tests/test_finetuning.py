import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.finetuning import avaliar as av
from src.finetuning.dataset import bloco_normalizado, montar, salvar, split_da_assinatura
from src.finetuning.modelo_hf import mensagens_do_par, para_chat, tokenizar_par
from src.gobr.parser import GobrParser
from src.prompts import gerador_prompt
from src.validation.gobr_validator import GobrValidator

RAIZ = Path(__file__).resolve().parents[1]

PAR = {
    "id": "x_001", "split": "teste", "nome": "Ativos", "escopo": 'da carteira("c")',
    "descricao": "Os clientes ativos devem ser atendidos a cada 14 dias.",
    "bloco": 'parametros "Ativos":\n    da carteira("c")\n    quando ativos para o vendedor\n    frequencia de 14 dias',
    "contexto": ['time "t":\n    vendedores todostelevendas', 'carteira "c":\n    clientes carteirafixa("t")'],
    "exemplos": [],
}


class TokenizadorFalso:
    eos_token = "</s>"

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        return "".join(f"<{m['role']}>{m['content']}" for m in chat) + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, texto, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in texto]}

    def get_vocab(self):
        return {"<|im_end|>": 1}


class ModeloFalso:
    def __init__(self, resposta):
        self.resposta = resposta
        self.chamadas = 0

    def invoke(self, mensagens):
        self.chamadas += 1
        return self.resposta(mensagens) if callable(self.resposta) else self.resposta


class ArgumentosTreinoTests(unittest.TestCase):
    def test_transformers_5_sem_warmup_ratio(self):
        from src.finetuning.treinar import argumentos_compativeis

        class V5:
            def __init__(self, output_dir, warmup_steps=0, fp16=False):
                self.kwargs = {"output_dir": output_dir, "warmup_steps": warmup_steps, "fp16": fp16}

        class V4(V5):
            def __init__(self, output_dir, warmup_ratio=0.0, group_by_length=False, fp16=False):
                self.kwargs = {"warmup_ratio": warmup_ratio, "group_by_length": group_by_length}

        v5 = argumentos_compativeis(V5, output_dir="x", warmup_ratio=0.03, group_by_length=True, fp16=True)
        self.assertEqual({"output_dir": "x", "warmup_steps": 0.03, "fp16": True}, v5.kwargs)
        v4 = argumentos_compativeis(V4, output_dir="x", warmup_ratio=0.03, group_by_length=True)
        self.assertEqual({"warmup_ratio": 0.03, "group_by_length": True}, v4.kwargs)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conjuntos = montar(RAIZ / "data" / "referencias", RAIZ / "data" / "exemplos", None)

    def test_bloco_normalizado(self):
        bruto = 'parametros "A":\n  do time("t")\n\t# comentario\n      frequencia de 7 dias\n'
        self.assertEqual('parametros "A":\n    do time("t")\n    frequencia de 7 dias', bloco_normalizado(bruto))

    def test_split_deterministico(self):
        self.assertEqual(split_da_assinatura("x"), split_da_assinatura("x"))

    def test_sem_vazamento_de_formas_e_conjuntos_preenchidos(self):
        treino, teste, real = (self.conjuntos[n] for n in ("treino", "teste", "real"))
        self.assertGreater(len(treino), 1000)
        self.assertGreater(len(teste), 40)
        self.assertGreater(len(real), 10)
        self.assertFalse({p["assinatura"] for p in teste} & {p["assinatura"] for p in treino})
        self.assertEqual(len(teste), len({p["chave_corpo"] for p in teste}))

    def test_exemplos_do_prompt_nao_sao_da_mesma_forma(self):
        com = [p for p in self.conjuntos["treino"] if p["exemplos"]]
        self.assertGreater(len(com), 300)
        for par in com[:50]:
            for nome, escopo, descricao, bloco in par["exemplos"]:
                self.assertNotEqual(par["bloco"], bloco)
                self.assertTrue(bloco.startswith("parametros "))

    def test_explicacoes_do_llm_viram_par_extra_no_treino(self):
        treino = self.conjuntos["treino"]
        alvo = next(p for p in treino if p["fonte_descricao"] == "rascunho")
        with tempfile.TemporaryDirectory() as tmp:
            arquivo = Path(tmp) / "exp.jsonl"
            arquivo.write_text(json.dumps({"id": alvo["id"], "explicacao": "Texto do LLM.", "metodo": "llm",
                                           "aprovada": True}) + "\n", encoding="utf-8")
            conjuntos = montar(RAIZ / "data" / "referencias", RAIZ / "data" / "exemplos", arquivo)
        fontes = [p["fonte_descricao"] for p in conjuntos["treino"] if p["id"] == alvo["id"]]
        self.assertEqual(["llm", "rascunho"], sorted(fontes))

    def test_salvar(self):
        with tempfile.TemporaryDirectory() as tmp:
            resumo = salvar(self.conjuntos, Path(tmp))
            self.assertEqual(0, resumo["vazamento_formas_teste_no_treino"])
            self.assertTrue((Path(tmp) / "treino.jsonl").exists())


class TokenizacaoTests(unittest.TestCase):
    def test_prompt_igual_ao_do_agente(self):
        esperado = gerador_prompt.mensagens(PAR["nome"], PAR["escopo"], PAR["descricao"])
        self.assertEqual(esperado, mensagens_do_par(PAR))

    def test_perda_so_na_resposta(self):
        t = tokenizar_par(PAR, TokenizadorFalso())
        n_prompt = sum(1 for l in t["labels"] if l == -100)
        resposta = "".join(chr(i) for i in t["input_ids"][n_prompt:])
        self.assertEqual(PAR["bloco"] + "<|im_end|>\n", resposta)
        self.assertTrue("".join(chr(i) for i in t["input_ids"][:n_prompt]).endswith("<assistant>"))

    def test_longo_demais(self):
        self.assertIsNone(tokenizar_par(PAR, TokenizadorFalso(), max_tokens=50))

    def test_para_chat(self):
        chat = para_chat([("system", "s"), ("human", "h"), ("ai", "a")])
        self.assertEqual(["system", "user", "assistant"], [m["role"] for m in chat])


class MetricasTests(unittest.TestCase):
    def setUp(self):
        self.v, self.p = GobrValidator(), GobrParser()

    def m(self, gerado):
        return av.metricas(gerado, PAR, validador=self.v, parser=self.p)

    def test_igual_a_menos_de_formatacao(self):
        r = self.m('parametros "Ativos":\n  da carteira( "c" )\n    quando ativos para o vendedor\n'
                   "    frequência de 14 dias")
        self.assertTrue(r["valido"] and r["ast_igual"] and r["numeros_ok"])

    def test_frequencia_errada(self):
        r = self.m(PAR["bloco"].replace("14", "7"))
        self.assertTrue(r["valido"])
        self.assertFalse(r["ast_igual"] or r["numeros_ok"] or r["exato"])
        self.assertAlmostEqual(2 / 3, r["f1_linhas"], places=3)

    def test_invalido_e_fora_do_vocabulario(self):
        self.assertFalse(self.m(PAR["bloco"].replace("frequencia de 14 dias", 'frequencia de "x'))["valido"])
        r = self.m(PAR["bloco"].replace("ativos para o vendedor", "clientes felizes"))
        self.assertFalse(r["vocabulario_ok"] and r["valido"])


class AvaliacaoTests(unittest.TestCase):
    def test_avaliar_grava_retoma_e_compara(self):
        pares = [PAR, dict(PAR, id="x_002", split="real")]
        with tempfile.TemporaryDirectory() as tmp:
            saida = Path(tmp)
            modelo = ModeloFalso("```\n" + PAR["bloco"] + "\n```")
            resumo = av.avaliar(modelo, pares, saida, rotulo="base")
            self.assertEqual(1.0, resumo["teste"]["ast_igual"])
            self.assertEqual(2, modelo.chamadas)
            av.avaliar(modelo, pares, saida, rotulo="base")      # retomada: nada a refazer
            self.assertEqual(2, modelo.chamadas)
            av.avaliar(ModeloFalso(PAR["bloco"].replace("14", "7")), pares, saida, rotulo="ruim")
            tabela = av.tabela_comparativa(saida)
            self.assertIn("| base | teste | 1 |", tabela)
            self.assertIn("| ruim | teste | 1 |", tabela)

    def test_exemplos_entram_no_prompt(self):
        vistos = []
        modelo = ModeloFalso(lambda msgs: (vistos.append(len(msgs)), PAR["bloco"])[1])
        exemplo = ["Ex", 'do time("t")', "desc", 'parametros "Ex":\n    do time("t")\n    nao agendar']
        with tempfile.TemporaryDirectory() as tmp:
            av.avaliar(modelo, [PAR], Path(tmp), rotulo="rag", exemplos=lambda p: [exemplo])
        self.assertEqual([4], vistos)                            # system + exemplo (2) + pedido


class PlugueTests(unittest.TestCase):
    def test_gerador_hf_pelo_llm_config(self):
        from src import llm_config

        falso = mock.MagicMock()
        with mock.patch.dict(os.environ, {"GOBR_MODELO_GERADOR": "hf:Qwen/Qwen3-4B",
                                          "GOBR_GERADOR_ADAPTADOR": "/drive/adaptador"}), \
                mock.patch("src.finetuning.modelo_hf.GeradorHF", falso), \
                mock.patch.dict(llm_config._MODELOS_HF, clear=True):
            a = llm_config.create_llm_para("gerador")
            b = llm_config.create_llm_para("gerador")
        falso.assert_called_once_with("Qwen/Qwen3-4B", "/drive/adaptador")
        self.assertIs(a, b)


if __name__ == "__main__":
    unittest.main()
