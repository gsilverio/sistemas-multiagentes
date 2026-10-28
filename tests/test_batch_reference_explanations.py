"""Testes offline do executor; não iniciam Ollama nem importam LangChain."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.batch_reference_explanations import (
    ReferenceBatchRunner, load_rules, read_previous,
)
from src.models.business_explanation import BusinessExplanation, ExplanationReview
from src.models.semantic_interpretation import SemanticInterpretation


class FakeInterpreter:
    def __init__(self):
        self.received = []

    def interpret(self, parsed, contexto=None):
        self.received.append((parsed, contexto))
        return SemanticInterpretation(publico='Clientes da carteira teste.')


class FakeExplainer:
    def __init__(self, *, fail=False):
        self.calls = 0
        self.fail = fail

    def explain(self, interpretation, parsed_rule=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError('Serviço Ollama indisponível')
        return BusinessExplanation(
            titulo='Regra de negócio', explicacao='Teste simulado.',
            revisao=ExplanationReview(status='pending_review', metodo='llm'),
        )


class BatchReferenceExplanationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / 'regras.jsonl'
        self.out = self.root / 'saida'
        self.rules = [
            {
                'id': f'regra_001_{i:03d}', 'nome': f'Caso {i}',
                'arquivo_origem': 'regra_001.gobr', 'indice': i,
                'linha_inicio': i * 5, 'linha_fim': i * 5 + 2,
                'gobr': f'parametros "Caso {i}":\n    da carteira("teste")\n    frequencia semanal()',
                'contexto': [{'tipo': 'carteira', 'nome': 'teste',
                              'texto': 'carteira "teste":\n    clientes todos'}],
                'referencias_nao_resolvidas': [],
            }
            for i in range(1, 5)
        ]
        self.write_rules(self.rules)
        self.interpreter = FakeInterpreter()
        self.explainer = FakeExplainer()
        self.runner = ReferenceBatchRunner(
            interpreter_factory=lambda: self.interpreter,
            explainer_factory=lambda: self.explainer,
        )

    def write_rules(self, rules):
        with self.input.open('w', encoding='utf-8') as f:
            for r in rules:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    def run_batch(self, **kwargs):
        return self.runner.run(input_path=self.input, output_dir=self.out,
                               model='gemma3:12b', num_ctx=4096,
                               revision='abc123', **kwargs)

    def test_dry_run_nao_carrega_modelo_nem_cria_saida(self):
        stats = self.run_batch(limit=2, dry_run=True)
        self.assertTrue(stats['dry_run'])
        self.assertFalse(self.out.exists())
        self.assertEqual(0, self.explainer.calls)

    def test_retomada_em_lotes_e_preservacao_de_resultado(self):
        one = self.run_batch(limit=2)
        self.assertEqual(2, one['concluidas_nesta_execucao'])
        path = self.out / 'items/regra_001_001.json'
        old = path.read_bytes()
        two = self.run_batch(limit=2)
        self.assertEqual(2, two['concluidas_nesta_execucao'])
        self.assertEqual(2, two['ignoradas_concluidas'])
        self.assertEqual(old, path.read_bytes())
        all_done = self.run_batch(limit=2)
        self.assertEqual(4, all_done['ignoradas_concluidas'])
        self.assertEqual(0, all_done['concluidas_nesta_execucao'])
        self.assertEqual(4, self.explainer.calls)
        for i in range(1, 5):
            item = read_previous(self.out / f'items/regra_001_{i:03d}.json')
            self.assertEqual('completed', item['status'])
            self.assertFalse(item['indexable'])
            self.assertEqual('pending_review', item['explanation']['revisao']['status'])
        self.assertEqual('carteira', self.interpreter.received[0][1][0]['tipo'])

    def test_erro_registrado_e_retentativa_limitada(self):
        self.explainer.fail = True
        s = self.run_batch(limit=1, max_attempts=2)
        self.assertEqual(1, s['falhas_nesta_execucao'])
        self.assertEqual(1, read_previous(self.out / 'items/regra_001_001.json')['attempt'])
        s = self.run_batch(limit=1, max_attempts=2)
        self.assertEqual(1, s['falhas_nesta_execucao'])
        self.assertEqual(2, read_previous(self.out / 'items/regra_001_001.json')['attempt'])
        s = self.run_batch(limit=1, max_attempts=2)
        self.assertEqual(1, s['ignoradas_tentativas'])
        self.assertEqual(1, s['falhas_nesta_execucao'])  # próximo item falha

    def test_contexto_muito_grande_bloqueia_sem_omitir(self):
        self.rules[0]['contexto'][0]['texto'] = 'Z' * 500
        self.write_rules(self.rules[:1])
        stats = self.run_batch(limit=1, max_context_chars=100)
        self.assertEqual(1, stats['bloqueadas_nesta_execucao'])
        result = read_previous(self.out / 'items/regra_001_001.json')
        self.assertEqual('blocked_context', result['status'])
        self.assertIn('ultrapassa', result['reason'])
        self.assertFalse(result['indexable'])
        self.assertEqual(0, self.explainer.calls)

    def test_novo_modelo_invalida_cache_anterior(self):
        self.run_batch(limit=1)
        stats = self.runner.run(input_path=self.input, output_dir=self.out,
                                limit=1, model='outro:8b', num_ctx=4096,
                                revision='abc123')
        self.assertEqual(1, stats['concluidas_nesta_execucao'])
        self.assertEqual('outro:8b', read_previous(self.out / 'items/regra_001_001.json')['model'])

    def test_rejeita_ids_duplicados(self):
        self.write_rules([self.rules[0], self.rules[0]])
        with self.assertRaisesRegex(ValueError, 'duplicado'):
            load_rules(self.input)

    def test_parser_com_clausula_nao_suportada_nao_chama_llm(self):
        self.rules[0]['gobr'] = 'parametros "Teste":\n    expressao ainda desconhecida'
        self.write_rules(self.rules[:1])
        stats = self.run_batch(limit=1)
        self.assertEqual(1, stats['bloqueadas_nesta_execucao'])
        self.assertEqual(0, self.explainer.calls)

    def test_referencias_nao_resolvidas_impedem_geracao(self):
        self.rules[0]['referencias_nao_resolvidas'] = ['inexistente']
        self.write_rules(self.rules[:1])
        stats = self.run_batch(limit=1)
        self.assertEqual(1, stats['bloqueadas_nesta_execucao'])
        self.assertEqual(0, self.explainer.calls)


if __name__ == '__main__':
    unittest.main()
