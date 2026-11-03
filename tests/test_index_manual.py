import unittest

from src.index_manual import chunk_text, documents_from_pages, normalize_for_embedding


class ManualIndexTests(unittest.TestCase):
    def test_chunk_text_respects_size_and_keeps_content(self):
        text = "Parágrafo inicial.\n" + ("conteúdo do manual " * 180)
        chunks = chunk_text(text, max_chars=500, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 500 for chunk in chunks))
        self.assertIn("Parágrafo inicial", chunks[0])
        self.assertIn("conteúdo do manual", " ".join(chunks))

    def test_documents_keep_page_and_chunk_metadata(self):
        docs = documents_from_pages(
            [(2, "Capítulo 5\nQuando houver inadimplência, não agendar.")],
            source_pdf="govendas-rules.pdf",
        )
        self.assertEqual(1, len(docs))
        self.assertEqual("manual_p002_c01", docs[0]["id"])
        self.assertEqual(2, docs[0]["metadata"]["pagina"])
        self.assertEqual("govendas-rules.pdf", docs[0]["metadata"]["arquivo_origem"])

    def test_embedding_normalization_preserves_search_terms_without_accents(self):
        normalized = normalize_for_embedding("Frequência de atendimento — não agendar")
        self.assertEqual("frequencia de atendimento nao agendar", normalized)
        self.assertNotIn("�", normalized)


if __name__ == "__main__":
    unittest.main()
