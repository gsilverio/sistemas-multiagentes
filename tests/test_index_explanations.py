import json
import tempfile
import unittest
from pathlib import Path

from src.index_explanations import load_documents


class ExplanationIndexTests(unittest.TestCase):
    def test_default_requires_explicit_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = Path(tmp)
            (items / "approved.json").write_text(json.dumps({
                "id": "approved",
                "status": "completed",
                "indexable": True,
                "nome": "Aprovada",
                "explanation": {
                    "explicacao": "Atender clientes ativos semanalmente.",
                    "revisao": {"status": "pending_review", "metodo": "llm"},
                },
            }), encoding="utf-8")
            (items / "pending.json").write_text(json.dumps({
                "id": "pending",
                "status": "completed",
                "indexable": False,
                "nome": "Pendente",
                "explanation": {"explicacao": "Nao agendar filiais."},
            }), encoding="utf-8")
            (items / "failed.json").write_text(json.dumps({
                "id": "failed", "status": "failed", "indexable": False,
            }), encoding="utf-8")

            docs = load_documents(items)
            self.assertEqual(["approved"], [doc["id"] for doc in docs])
            self.assertFalse(docs[0]["metadata"]["provisional"])

    def test_include_pending_marks_provisional_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = Path(tmp)
            (items / "pending.json").write_text(json.dumps({
                "id": "pending",
                "status": "completed",
                "indexable": False,
                "explanation": {
                    "explicacao": "Nao agendar clientes inadimplentes.",
                    "revisao": {"status": "needs_review", "metodo": "deterministic_fallback"},
                },
            }), encoding="utf-8")

            docs = load_documents(items, include_pending=True)
            self.assertEqual(1, len(docs))
            self.assertTrue(docs[0]["metadata"]["provisional"])
            self.assertEqual("needs_review", docs[0]["metadata"]["review_status"])


if __name__ == "__main__":
    unittest.main()
