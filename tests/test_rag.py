import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.gobr.assinatura import assinatura_estrutural, chave_corpo, construcoes
from src.gobr.extractor import GobrRuleExtractor
from src.gobr.parser import GobrParser
from src.rag import busca_hibrida
from src.rag.busca_hibrida import BM25, IndiceHibrido, fusao_rrf, tokenizar
from src.rag.exemplos import documentos_de_exemplos, explicacoes_aprovadas
from src.rag.manual import documentos_do_manual, ler_pdf, secoes
from src.rag.recuperador import buscar_exemplos, buscar_manual

RAIZ = Path(__file__).resolve().parents[1]

CABECALHO_PAGINA = "6 R e g r a s p a r a f i l t r o s d e c l i e n t e s 31"
COPYRIGHT = "© 2 0 2 6 , G o v e n d a s • A L L R I G H T S R E S E R V E D ."


def _regra(id_, gobr, contexto=()):
    return {"id": id_, "nome": gobr.splitlines()[0], "arquivo_origem": "x.gobr", "gobr": gobr,
            "contexto": [{"texto": c} for c in contexto]}


ATIVOS = 'parametros "Ativos":\n    do time("televendas")\n    quando ativos para o vendedor\n    frequencia de 14 dias'
ATIVOS_30 = 'parametros "Ativos 30":\n    do time("televendas")\n    quando ativos para o vendedor\n    frequencia de 30 dias'
INADIMPLENTES = 'parametros "Bloqueio":\n    do time("televendas")\n    quando inadimplentes\n    nao agendar'


class TokenizacaoTests(unittest.TestCase):
    def test_sem_acentos_plural_e_stopwords(self):
        self.assertEqual(["cliente", "inativo", "frequencia"], tokenizar("Os clientes inativos — frequência"))
        self.assertEqual(["parametrizacao"], tokenizar("parametrizações"))

    def test_mantem_nao(self):
        self.assertIn("nao", tokenizar("não agendar"))


class BM25Tests(unittest.TestCase):
    def test_termo_raro_pesa_mais(self):
        bm = BM25(["cliente ativo frequencia", "cliente inadimplente nao agendar", "cliente ativo preferencia"])
        scores = bm.scores("nao agendar cliente")
        self.assertEqual(1, max(range(3), key=lambda i: scores[i]))

    def test_rrf_premia_quem_aparece_nas_duas_listas(self):
        total = fusao_rrf([[0, 1], [2, 1]])
        self.assertEqual(1, max(total, key=total.get))


class IndiceHibridoTests(unittest.TestCase):
    DOCS = [
        {"id": "a", "text": "clientes ativos a cada 14 dias", "metadata": {"g": "x"}},
        {"id": "b", "text": "nao agendar clientes inadimplentes", "metadata": {"g": "y"}},
        {"id": "c", "text": "preferência de visita na segunda", "metadata": {"g": "x"}},
    ]

    def test_sem_vetores_cai_para_bm25(self):
        indice = IndiceHibrido(self.DOCS)
        resultado = indice.buscar("não agendar", k=2)
        self.assertEqual("b", resultado[0]["id"])
        self.assertEqual("bm25", resultado[0]["modo"])

    def test_filtro_por_metadado(self):
        indice = IndiceHibrido(self.DOCS)
        resultado = indice.buscar("clientes", k=5, filtro=lambda d: d["metadata"]["g"] == "x")
        self.assertEqual({"a"}, {r["id"] for r in resultado})

    def test_hibrido_usa_os_vetores(self):
        # A consulta não tem nenhuma palavra em comum com "c", só o vetor aponta para ela.
        vetores = np.eye(3, dtype="float32")
        indice = IndiceHibrido(self.DOCS, vetores=vetores, modelo="falso")
        with mock.patch.object(busca_hibrida, "embeddings_disponiveis", return_value=True), \
                mock.patch.object(busca_hibrida.Codificador, "codificar",
                                  return_value=np.array([[0, 0, 1]], dtype="float32")):
            resultado = indice.buscar("dia preferido do cliente", k=1, modo="denso")
        self.assertEqual("c", resultado[0]["id"])
        self.assertEqual("denso", resultado[0]["modo"])

    def test_salvar_e_carregar(self):
        with tempfile.TemporaryDirectory() as tmp:
            pasta = Path(tmp)
            IndiceHibrido(self.DOCS, vetores=np.eye(3, dtype="float32"), modelo="m", tipo="t").salvar(pasta)
            carregado = IndiceHibrido.carregar(pasta)
            self.assertEqual(3, len(carregado.documentos))
            self.assertEqual((3, 3), carregado.vetores.shape)
            # Reconstruir sem vetores apaga o vetores.npy antigo.
            IndiceHibrido(self.DOCS[:2], tipo="t").salvar(pasta)
            self.assertFalse((pasta / "vetores.npy").exists())

    def test_vetores_incompativeis_sao_recusados(self):
        with tempfile.TemporaryDirectory() as tmp:
            pasta = Path(tmp)
            IndiceHibrido(self.DOCS, vetores=np.eye(3, dtype="float32"), modelo="m").salvar(pasta)
            (pasta / "documents.json").write_text(json.dumps(self.DOCS[:2]), encoding="utf-8")
            with self.assertRaises(ValueError):
                IndiceHibrido.carregar(pasta)


class ManualTests(unittest.TestCase):
    PAGINAS = [
        (1, "Govendas Business Rules (GoBR)\nÍndice\n1. Govendas ........ 1"),
        (5, "1 Govendas\nx\n2 A linguagem\nx\n3 Times\nx\n4 Carteiras\nx\n5 Filtros temporais\nx"),
        (35, f"{CABECALHO_PAGINA}\n{COPYRIGHT}\n6 Regras para filtros de clientes\nIntrodução.\n"
             "6.1 Outros filtros úteis\nFiltro   Descrição\n    quando adimplentes\n"
             "5 dias\n6.1.1 Opções para tempo\n<<N>> dias"),
        (36, f"{CABECALHO_PAGINA}\n{COPYRIGHT}\n<<N>> meses\n7 Regras para parâmetros\nTexto 7."),
    ]

    def test_secoes_seguem_a_numeracao_e_ignoram_ruido(self):
        resultado = secoes(self.PAGINAS)
        self.assertEqual([(1,), (2,), (3,), (4,), (5,), (6,), (6, 1), (6, 1, 1), (7,)],
                         [s["numero"] for s in resultado])
        linhas = [l for l, _ in resultado[6]["linhas"]]
        self.assertIn("5 dias", linhas)            # não virou título
        self.assertIn("    quando adimplentes", linhas)  # indentação preservada
        self.assertIn("Filtro Descrição", linhas)  # espaços repetidos colapsados
        self.assertFalse(any("R E S E R V E D" in l for s in resultado for l, _ in s["linhas"]))

    def test_documento_tem_caminho_construcao_e_paginas(self):
        docs = {d["id"]: d for d in documentos_do_manual(self.PAGINAS)}
        doc = docs["manual_6.1.1"]
        self.assertTrue(doc["text"].startswith(
            "6 Regras para filtros de clientes > 6.1 Outros filtros úteis > 6.1.1 Opções para tempo"))
        self.assertEqual("filtro_cliente", doc["metadata"]["construcao"])
        self.assertEqual([35, 36], doc["metadata"]["paginas"])
        self.assertEqual("parametros", docs["manual_7"]["metadata"]["construcao"])

    def test_secao_longa_dividida_em_partes(self):
        pagina = "6 Regras\n" + "\n".join(f"linha {i} " + "x" * 80 for i in range(40))
        docs = documentos_do_manual([(1, "1 A\nx"), (2, "2 B\ny"), (3, "3 C\nz"), (4, "4 D\nw"),
                                     (5, "5 E\nv"), (6, pagina)], tamanho_maximo=1000)
        partes = [d for d in docs if d["metadata"]["secao"] == "6"]
        self.assertGreater(len(partes), 1)
        self.assertTrue(all(len(d["text"]) <= 1000 + 20 for d in partes))
        self.assertTrue(all(d["text"].startswith("6 Regras\n") for d in partes))
        self.assertEqual("manual_6_p2", partes[1]["id"])

    def test_manual_real(self):
        docs = documentos_do_manual(ler_pdf(RAIZ / "data" / "manual" / "govendas-rules.pdf"))
        numeros = {d["metadata"]["secao"] for d in docs}
        for esperado in ("1.1", "4.6.1", "5.2", "6.11.1", "6.13", "7.3.11", "7.4.2", "8.8", "9.1.2", "10"):
            self.assertIn(esperado, numeros)
        self.assertFalse(any("A L L R I G H T S" in d["text"] for d in docs))
        self.assertTrue(all(len(d["text"]) <= 1800 for d in docs))


class AssinaturaTests(unittest.TestCase):
    def test_mesma_forma_com_numeros_diferentes(self):
        p = GobrParser()
        self.assertEqual(assinatura_estrutural(p.parse(ATIVOS)), assinatura_estrutural(p.parse(ATIVOS_30)))
        self.assertNotEqual(assinatura_estrutural(p.parse(ATIVOS)), assinatura_estrutural(p.parse(INADIMPLENTES)))

    def test_chave_corpo_ignora_o_nome(self):
        outro = ATIVOS.replace('"Ativos"', '"Outro nome"')
        self.assertEqual(chave_corpo(ATIVOS), chave_corpo(outro))

    def test_construcoes(self):
        p = GobrParser()
        self.assertIn("nao_agendar", construcoes(p.parse(INADIMPLENTES)))
        self.assertIn("frequencia_fixa", construcoes(p.parse(ATIVOS)))

    def test_acervo(self):
        p = GobrParser()
        blocos = GobrRuleExtractor().extract_directory(RAIZ / "data" / "referencias")
        self.assertEqual(1204, len({chave_corpo(b.gobr) for b in blocos}))
        self.assertLess(len({assinatura_estrutural(p.parse(b.gobr)) for b in blocos}), 500)


class ExemplosTests(unittest.TestCase):
    def test_deduplica_corpo_e_prefere_explicacao_do_llm(self):
        regras = [_regra("r1", ATIVOS, ['time "televendas":\n    vendedores todostelevendas ativados']),
                  _regra("r2", ATIVOS.replace('"Ativos"', '"Copia"')),
                  _regra("r3", INADIMPLENTES)]
        aprovadas = explicacoes_aprovadas([
            {"id": "r2", "explicacao": "Os ativos do time \"televendas\" são atendidos a cada 14 dias.",
             "metodo": "llm", "aprovada": True},
            {"id": "r3", "explicacao": "texto reprovado", "metodo": "rascunho", "aprovada": True},
        ])
        docs = {d["id"]: d for d in documentos_de_exemplos(regras, aprovadas)}
        self.assertEqual({"r1", "r3"}, set(docs))
        m = docs["r1"]["metadata"]
        self.assertEqual(["r2"], m["ids_iguais"])
        self.assertEqual("llm", m["metodo"])
        self.assertIn("a cada 14 dias", docs["r1"]["text"])
        self.assertIn("frequencia de 14 dias", docs["r1"]["texto_lexico"])  # GOBR para o BM25
        self.assertIn(m["rascunho"], docs["r1"]["texto_lexico"])
        self.assertEqual(1, len(m["contexto"]))
        self.assertEqual("rascunho", docs["r3"]["metadata"]["metodo"])

    def test_clausula_desconhecida_fica_de_fora(self):
        estranho = INADIMPLENTES.replace("quando inadimplentes", "quando cliente marciano")
        self.assertEqual([], documentos_de_exemplos([_regra("x", estranho)]))


class RecuperadorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pasta = Path(self.tmp.name)
        regras = [_regra("r1", ATIVOS), _regra("r2", ATIVOS_30), _regra("r3", INADIMPLENTES)]
        IndiceHibrido(documentos_de_exemplos(regras), tipo="gobr_exemplos").salvar(self.pasta / "exemplos")
        IndiceHibrido(documentos_do_manual(ManualTests.PAGINAS), tipo="m").salvar(self.pasta / "manual")

    def tearDown(self):
        self.tmp.cleanup()

    def test_exemplos_diversificados(self):
        resultado = buscar_exemplos("clientes ativos atendidos a cada 14 dias", k=3,
                                    diretorio=self.pasta / "exemplos")
        assinaturas = [r.assinatura for r in resultado]
        self.assertEqual(len(assinaturas), len(set(assinaturas)))
        self.assertEqual("r1", resultado[0].id)
        sem_diversificar = buscar_exemplos("clientes ativos atendidos a cada 14 dias", k=2, diversificar=False,
                                           diretorio=self.pasta / "exemplos")
        self.assertEqual({"r1", "r2"}, {r.id for r in sem_diversificar})

    def test_exemplos_exigindo_construcao(self):
        resultado = buscar_exemplos("clientes ativos", k=3, exigir=["nao_agendar"],
                                    diretorio=self.pasta / "exemplos")
        self.assertEqual(["r3"], [r.id for r in resultado])

    def test_manual_filtrado_por_construcao(self):
        resultado = buscar_manual("texto", k=3, construcao="parametros", diretorio=self.pasta / "manual")
        self.assertEqual(["7"], [r.secao for r in resultado])
        self.assertIn("seção 7", resultado[0].citacao())


if __name__ == "__main__":
    unittest.main()
