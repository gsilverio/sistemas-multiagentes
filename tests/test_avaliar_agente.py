import json
import tempfile
import unittest
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from src.grafo.grafo import construir_grafo
from src.pipelines import avaliar_agente as av
from tests.test_grafo import CORPOS, SPEC, LLMRoteiro, agentes, gerador_correto

ESPERADO = "version 1.0\n\ntime \"televendas\":\n    vendedores todostelevendas ativados\n\n" + "\n\n".join(
    f'parametros "{nome}":\n    do time("televendas")\n' + "\n".join("    " + l for l in CORPOS[nome])
    for nome in ("Não agendar inadimplentes", "Clientes ativos", "Risco de inatividade")) + "\n"


class AvaliarAgenteTests(unittest.TestCase):
    def test_caso_completo(self):
        grafo = construir_grafo(agentes(LLMRoteiro(json.dumps(SPEC)), LLMRoteiro(*[gerador_correto] * 3)),
                                checkpointer=InMemorySaver())
        caso = {"id": "x", "texto": "tabela", "gobr_esperado_texto": ESPERADO,
                "achados_esperados": ["sem_fallback", "fora_do_gobr"], "achados_ausentes": ["buraco_faixa"]}
        r = av.avaliar_caso(grafo, caso)
        self.assertTrue(r["valido"])
        self.assertEqual(1.0, r["acerto"])
        self.assertEqual([], r["erros_comportamento"])
        self.assertEqual((3, 3, 1.0, 1.0, 1.0), (r["blocos_gerados"], r["blocos_esperados"], r["cobertura_ast"],
                                                 r["cobertura_acao"], r["achados_ok"]))
        self.assertEqual(7, r["passos"])

    def test_erro_no_caso_nao_derruba(self):
        class Quebrado:
            def invoke(self, *a, **k):
                raise RuntimeError("falhou")

        r = av.avaliar_caso(Quebrado(), {"id": "y", "texto": "t"})
        self.assertFalse(r["valido"])
        self.assertIn("falhou", r["erro"])

    def test_achados_faltando_e_indevidos(self):
        r = av.conferir_achados(["buraco_faixa"], {"achados_esperados": ["buraco_faixa", "sem_fallback"],
                                                   "achados_ausentes": ["buraco_faixa"]})
        self.assertEqual(["sem_fallback"], r["achados_faltando"])
        self.assertEqual(["buraco_faixa"], r["achados_indevidos"])
        self.assertAlmostEqual(1 / 3, r["achados_ok"], places=2)

    def test_avaliar_grava_e_tabela(self):
        grafo = construir_grafo(agentes(LLMRoteiro(json.dumps(SPEC)), LLMRoteiro(*[gerador_correto] * 3)),
                                checkpointer=InMemorySaver())
        with tempfile.TemporaryDirectory() as tmp:
            resumo = av.avaliar(grafo, [{"id": "x", "texto": "t", "gobr_esperado_texto": ESPERADO}], Path(tmp),
                                rotulo="base")
            self.assertEqual(1.0, resumo["cobertura_ast"])
            self.assertIn("| base | x | 1.0 | True | 3/3 |", av.tabela(Path(tmp)))

    def test_casos_do_repositorio_sao_coerentes(self):
        casos = json.loads(av.CASOS.read_text(encoding="utf-8"))["casos"]
        self.assertGreaterEqual(len(casos), 11)
        for caso in casos:
            esperado = av.gobr_esperado(caso)
            if esperado:
                if caso.get("gobr_esperado_texto"):  # os reais (V1Vidros) vieram sem a linha version
                    self.assertTrue(av.GobrValidator().validar(esperado).ok, caso["id"])
                r = av.comparar(esperado, esperado)
                self.assertEqual((1.0, 1.0), (r["cobertura_ast"], r["acerto"]), caso["id"])
            texto, _ = av.entrada_do_caso(caso)
            self.assertTrue(texto.strip(), caso["id"])


if __name__ == "__main__":
    unittest.main()
