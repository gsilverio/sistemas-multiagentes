import json
import tempfile
import unittest
from pathlib import Path

from src.interface.app import criar_app, mensagens_do_historico, responder, tabela_decisoes
from src.interface.servico import ServicoConversa
from tests.test_grafo import SPEC, LLMRoteiro, agentes, gerador_correto

RAIZ = Path(__file__).resolve().parents[1]


def _sem_frequencia():
    spec = json.loads(json.dumps(SPEC))
    spec["segmentos"][0]["frequencia_dias"] = None
    return json.dumps(spec)


class ServicoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.banco = Path(self.tmp.name) / "conversas.sqlite"
        self.servicos = []

    def tearDown(self):
        for servico in self.servicos:
            servico.fechar()
        self.tmp.cleanup()

    def _servico(self, percepcao, gerador, **kw):
        servico = ServicoConversa(agentes(percepcao, gerador, **kw), banco=self.banco)
        self.servicos.append(servico)
        return servico

    def test_eventos_de_progresso_e_resposta(self):
        servico = self._servico(LLMRoteiro(json.dumps(SPEC)), LLMRoteiro(*[gerador_correto] * 3))
        eventos = list(servico.enviar("c1", "tabela da Maxlub"))
        tipos = [e.tipo for e in eventos]
        self.assertEqual("resposta", tipos[-1])
        self.assertTrue(all(t == "progresso" for t in tipos[:-1]))
        textos = "\n".join(e.texto for e in eventos)
        self.assertIn("Percepção: 3 grupo(s)", textos)
        self.assertIn("✅ Validador", textos)
        self.assertIn("```gobr", eventos[-1].texto)
        self.assertEqual(["usuario", "agente"], [m["papel"] for m in servico.historico("c1")])

    def test_memoria_sobrevive_a_um_novo_processo_inclusive_pausada(self):
        percepcao = LLMRoteiro(_sem_frequencia(), json.dumps(SPEC))
        servico = self._servico(percepcao, LLMRoteiro(*[gerador_correto] * 3), interativo=True)
        eventos = list(servico.enviar("c2", "tabela da Maxlub"))
        self.assertEqual("pergunta", eventos[-1].tipo)
        self.assertIn("não tem frequência", eventos[-1].texto)
        servico.fechar()

        # "Reinicia" o processo: novo serviço, mesmo arquivo SQLite, mesmos LLMs falsos.
        novo = self._servico(percepcao, LLMRoteiro(*[gerador_correto] * 3), interativo=True)
        self.assertIsNotNone(novo.pausada("c2"))
        self.assertTrue(novo.historico("c2")[-1].get("pergunta"))
        eventos = list(novo.enviar("c2", "Clientes ativos: a cada 14 dias"))
        self.assertEqual("resposta", eventos[-1].tipo)
        self.assertIn("frequencia de 14 dias", novo.gobr("c2"))
        papeis = [m["papel"] for m in novo.historico("c2")]
        self.assertEqual(["usuario", "agente", "usuario", "agente"], papeis)
        self.assertIn("c2", novo.conversas())

    def test_resposta_vazia_segue_com_as_suposicoes(self):
        percepcao = LLMRoteiro(json.dumps(SPEC))
        servico = self._servico(percepcao, LLMRoteiro(*[gerador_correto] * 3), interativo=True)
        self.assertEqual("pergunta", list(servico.enviar("c3", "tabela"))[-1].tipo)
        eventos = list(servico.enviar("c3", ""))
        self.assertEqual("resposta", eventos[-1].tipo)
        self.assertIn("**Assumido:**", eventos[-1].texto)
        self.assertEqual(1, len(percepcao.chamadas))           # sem resposta: não refaz a percepção

    def test_anexo_pdf_vai_para_a_percepcao_mas_nao_para_o_historico(self):
        percepcao = LLMRoteiro(json.dumps(SPEC))
        servico = self._servico(percepcao, LLMRoteiro(*[gerador_correto] * 3))
        list(servico.enviar("c4", "", [RAIZ / "data" / "exemplos" / "RegraMaxlub.pdf"]))
        enviado = percepcao.chamadas[0][1][1]
        self.assertIn("Risco de Inatividade", enviado)
        primeira = servico.historico("c4")[0]
        self.assertEqual("(regras no anexo)", primeira["texto"])
        self.assertEqual(["RegraMaxlub.pdf"], primeira["anexos"])

    def test_mensagem_vazia_sem_anexo(self):
        servico = self._servico(LLMRoteiro(), LLMRoteiro())
        self.assertEqual(["erro"], [e.tipo for e in servico.enviar("c5", "  ")])

    def test_decisoes_do_ultimo_ciclo(self):
        servico = self._servico(LLMRoteiro(json.dumps(SPEC)), LLMRoteiro(*[gerador_correto] * 3))
        list(servico.enviar("c6", "tabela"))
        decisoes = servico.decisoes("c6")
        self.assertEqual("percepcao", decisoes[0]["proximo"])
        self.assertEqual("finalizar", decisoes[-1]["proximo"])


class AppTests(unittest.TestCase):
    def test_responder_atualiza_chat_arquivo_e_decisoes(self):
        servico = ServicoConversa(agentes(LLMRoteiro(json.dumps(SPEC)), LLMRoteiro(*[gerador_correto] * 3)))
        saidas = list(responder(servico, {"text": "tabela da Maxlub", "files": []}, [], "a1"))
        chat, caixa, arquivo, decisoes, status = saidas[-1]
        self.assertEqual(["user", "assistant", "assistant"], [m["role"] for m in chat])
        self.assertEqual("done", chat[1]["metadata"]["status"])
        self.assertIn("Percepção", chat[1]["content"])
        self.assertIn("```gobr", chat[2]["content"])
        self.assertTrue(Path(arquivo).read_text(encoding="utf-8").startswith("version 1.0"))
        self.assertIn("`finalizar`", decisoes)
        self.assertIn("Concluído", status)
        self.assertGreater(len(saidas), 3)                     # houve atualizações intermediárias

    def test_auxiliares(self):
        chat = mensagens_do_historico([{"papel": "usuario", "texto": "oi", "anexos": ["a.pdf"]},
                                       {"papel": "agente", "texto": "olá"}])
        self.assertEqual("user", chat[0]["role"])
        self.assertIn("📎 `a.pdf`", chat[0]["content"])
        self.assertIn("🛡️ guardrail", tabela_decisoes([{"proximo": "x", "origem": "guardrail", "motivo": "a|b"}]))

    def test_app_monta(self):
        servico = ServicoConversa(agentes(LLMRoteiro(), LLMRoteiro()))
        app = criar_app(servico, exemplos=sorted((RAIZ / "data" / "exemplos").glob("*.pdf")))
        self.assertIsNotNone(app)


if __name__ == "__main__":
    unittest.main()
