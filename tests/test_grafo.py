import json
import re
import unittest

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.agents.analista import AgenteAnalista
from src.agents.contratos import ContextoSegmento, Faixa, RuleSpec, Segmento, TrechoContexto
from src.agents.explicador import AgenteExplicador
from src.agents.gerador import AgenteGerador, limpar_bloco
from src.agents.percepcao import AgentePercepcao, PercepcaoFalhou
from src.agents.recuperador import AgenteRecuperador
from src.agents.validador import AgenteValidador, dias_da_frequencia
from src.grafo import orquestrador
from src.grafo.estado import Agentes
from src.grafo.grafo import config_thread, construir_grafo, entrada


class LLMRoteiro:
    """LLM falso: devolve as respostas na ordem; uma resposta pode ser função das mensagens."""

    def __init__(self, *respostas):
        self.respostas = list(respostas)
        self.chamadas = []

    def invoke(self, mensagens):
        self.chamadas.append(mensagens)
        if not self.respostas:
            raise AssertionError("LLM chamado mais vezes que o roteiro previa")
        r = self.respostas.pop(0)
        return r(mensagens) if callable(r) else r


SPEC = {
    "cliente": "Maxlub", "time": "televendas", "vendedores": "todostelevendas",
    "segmentos": [
        {"nome": "Clientes ativos", "criterio": "Clientes com compra entre 0 e 60 dias",
         "faixa": {"medida": "dias_sem_compra", "min_dias": 0, "max_dias": 60}, "frequencia_dias": 14},
        {"nome": "Risco de inatividade", "criterio": "Compraram nos últimos 90 mas estão há 60 sem comprar",
         "faixa": {"medida": "dias_sem_compra", "min_dias": 60, "max_dias": 90}, "frequencia_dias": 7},
        {"nome": "Não agendar inadimplentes", "criterio": "Clientes inadimplentes", "nao_agendar": True},
    ],
    "restricoes_gerais": ["Capacidade de atendimento: 35 clientes (Dia)"],
}

CORPOS = {
    "Clientes ativos": ["quando recencia de positivacao em 60 dias", "frequencia de 14 dias"],
    "Risco de inatividade": ["quando recencia de positivacao em 90 dias", "frequencia de 7 dias"],
    "Não agendar inadimplentes": ["quando inadimplentes", "nao agendar"],
    "Demais clientes": ["nao agendar"],
}


def _pedido(mensagens):
    ultima = [conteudo for papel, conteudo in mensagens if papel == "human"][-1]
    nome = re.search(r'Nome da regra: "(.*)"', ultima)
    return (nome.group(1) if nome else None), ultima


def gerador_correto(mensagens):
    nome, _ = _pedido(mensagens)
    if nome is None:  # pedido de correção: o nome está no bloco anterior
        nome = re.search(r'parametros "(.*)":', mensagens[-2][1]).group(1)
    corpo = CORPOS[nome]
    return "\n".join([f'parametros "{nome}":', '    do time("televendas")', *("    " + l for l in corpo)])


def gerador_errando_ativos_uma_vez():
    errou = {"feito": False}

    def responder(mensagens):
        nome, _ = _pedido(mensagens)
        if nome == "Clientes ativos" and not errou["feito"]:
            errou["feito"] = True
            return ('Claro! Aqui está:\n```\nparametros "Clientes ativos":\n    do time("televendas")\n'
                    "    quando recencia de positivacao em 60 dias\n    frequencia de 7 dias\n```")
        return gerador_correto(mensagens)
    return responder


def buscar_exemplos_falso(consulta, **kwargs):
    return []


def buscar_manual_falso(consulta, **kwargs):
    return []


def agentes(percepcao_llm, gerador_llm, *, orquestrador_llm=None, interativo=False, recuperador=None):
    return Agentes(
        percepcao=AgentePercepcao(percepcao_llm), analista=AgenteAnalista(),
        recuperador=recuperador or AgenteRecuperador(buscar_exemplos=buscar_exemplos_falso,
                                                     buscar_manual=buscar_manual_falso),
        gerador=AgenteGerador(gerador_llm), validador=AgenteValidador(),
        explicador=AgenteExplicador(usar_llm=False), orquestrador_llm=orquestrador_llm, interativo=interativo)


def rodar(ag, texto="tabela da Maxlub", thread="t"):
    grafo = construir_grafo(ag, checkpointer=InMemorySaver())
    config = config_thread(thread)
    return grafo, config, grafo.invoke(entrada(texto), config)


def nos(saida):
    return [p["no"] for p in saida["trace"] if p["no"] not in ("orquestrador", "inicio")]


class AnalistaTests(unittest.TestCase):
    def test_reordena_aponta_buraco_sobreposicao_e_falta_de_fallback(self):
        spec = RuleSpec(segmentos=[
            Segmento(nome="Inativos", faixa=Faixa(min_dias=90, max_dias=120), frequencia_dias=28, criterio="x"),
            Segmento(nome="Ativos", faixa=Faixa(min_dias=0, max_dias=60), frequencia_dias=14, criterio="x"),
            Segmento(nome="Bloqueio", criterio="inadimplentes", nao_agendar=True),
            Segmento(nome="Meio", faixa=Faixa(min_dias=100, max_dias=130), frequencia_dias=7, criterio="x"),
        ])
        a = AgenteAnalista().analisar(spec)
        self.assertEqual(["Bloqueio", "Ativos", "Inativos", "Meio"], [s.nome for s in a.spec.segmentos])
        codigos = {x.codigo for x in a.achados}
        self.assertTrue({"buraco_faixa", "sobreposicao_faixa", "sem_fallback", "sugestao_receita"} <= codigos)
        self.assertNotIn("sugestao_inadimpl", codigos)
        buraco = next(x for x in a.achados if x.codigo == "buraco_faixa")
        self.assertIn("60 a 90 dias", buraco.mensagem)
        self.assertTrue(any("130 dias" in p.texto for p in a.perguntas))
        self.assertTrue(a.ajustes)

    def test_faixas_vizinhas_e_nao_agendar_cobrem(self):
        a = AgenteAnalista().analisar(RuleSpec(segmentos=[
            Segmento(nome="A", faixa=Faixa(max_dias=30), frequencia_dias=30, criterio="x"),
            Segmento(nome="B", faixa=Faixa(min_dias=31, max_dias=60), frequencia_dias=15, criterio="x"),
            Segmento(nome="C", faixa=Faixa(min_dias=60), nao_agendar=True, criterio="x")]))
        self.assertFalse({"buraco_faixa", "sem_fallback"} & {x.codigo for x in a.achados})

    def test_restricao_que_um_bloqueio_ja_cobre(self):
        rota = "Somente clientes que não estão na rota da semana"
        a = AgenteAnalista().analisar(RuleSpec(segmentos=[
            Segmento(nome="Ativos", faixa=Faixa(max_dias=60), frequencia_dias=14, criterio="x",
                     restricoes=[rota, "preferência na sexta"]),
            Segmento(nome="Não agendar inadimplentes", criterio="inadimplentes", nao_agendar=True),
            Segmento(nome="Não agendar clientes em rota", criterio="clientes em rota", nao_agendar=True)]))
        nomes = {s.nome: s for s in a.spec.segmentos}
        self.assertEqual(["preferência na sexta"], nomes["Ativos"].restricoes)
        self.assertEqual("clientes em rota na semana", nomes["Não agendar clientes em rota"].criterio)
        sem_bloqueio = AgenteAnalista().analisar(RuleSpec(segmentos=[
            Segmento(nome="Ativos", frequencia_dias=14, criterio="x", restricoes=[rota])]))
        self.assertEqual([rota], sem_bloqueio.spec.segmentos[0].restricoes)

    def test_segmento_sem_acao_e_pergunta_bloqueante(self):
        a = AgenteAnalista().analisar(RuleSpec(segmentos=[Segmento(nome="Prospects", criterio="prospects")]))
        self.assertEqual(["acao:Prospects"], [p.id for p in a.bloqueantes])

    def test_fallback_cobre_o_fim_das_faixas(self):
        a = AgenteAnalista().analisar(RuleSpec(segmentos=[
            Segmento(nome="Ativos", faixa=Faixa(max_dias=60), frequencia_dias=14, criterio="x"),
            Segmento(nome="Demais", fallback=True, nao_agendar=True)]))
        self.assertFalse(a.perguntas)
        self.assertNotIn("sem_fallback", {x.codigo for x in a.achados})

    def test_fallback_com_criterio_proprio_e_restricao_duplicada(self):
        a = AgenteAnalista().analisar(RuleSpec(
            segmentos=[Segmento(nome="Ativos", criterio="x", faixa=Faixa(max_dias=60), frequencia_dias=7,
                                restricoes=["Fora da rota."]),
                       Segmento(nome="Não agendar inadimplentes", criterio="inadimplentes", nao_agendar=True,
                                fallback=True),
                       Segmento(nome="Demais clientes", criterio="os demais clientes", nao_agendar=True,
                                fallback=True)],
            restricoes_gerais=["fora da rota"]))
        self.assertEqual(["Não agendar inadimplentes", "Ativos", "Demais clientes"], [s.nome for s in a.spec.segmentos])
        self.assertEqual([], a.spec.restricoes_gerais)
        self.assertFalse([p for p in a.perguntas if p.id.startswith("restricao_geral")])

    def test_restricao_geral_vira_pergunta_e_time_coerente(self):
        a = AgenteAnalista().analisar(RuleSpec(
            time="televendas", vendedores="todosdecampo",
            segmentos=[Segmento(nome="A", criterio="x", frequencia_dias=7, fallback=True)],
            restricoes_gerais=["Somente clientes que não estão na rota da semana"]))
        self.assertEqual(["restricao_geral:1"], [p.id for p in a.perguntas])
        self.assertFalse(a.perguntas[0].bloqueante)
        self.assertEqual("todostelevendas", a.spec.vendedores)
        self.assertTrue(any("todostelevendas" in x for x in a.ajustes))

    def test_nomes_repetidos_e_restricao_fora_do_gobr(self):
        a = AgenteAnalista().analisar(RuleSpec(
            segmentos=[Segmento(nome="A", nao_agendar=True, criterio="x"), Segmento(nome="A", nao_agendar=True, criterio="y")],
            restricoes_gerais=["Capacidade de 35 clientes por dia"]))
        self.assertEqual(["A", "A (2)"], [s.nome for s in a.spec.segmentos])
        self.assertIn("fora_do_gobr", {x.codigo for x in a.achados})


class AgentesTests(unittest.TestCase):
    def test_percepcao_corrige_json_invalido(self):
        llm = LLMRoteiro("não sei", "```json\n" + json.dumps(SPEC) + "\n```")
        spec = AgentePercepcao(llm).extrair(["texto"])
        self.assertEqual(3, len(spec.segmentos))
        self.assertIn("não é um JSON válido", llm.chamadas[1][-1][1])

    def test_bloqueio_nas_restricoes_gerais_vira_grupo(self):
        spec = json.loads(json.dumps(SPEC))
        spec["segmentos"] = spec["segmentos"][:2]
        spec["restricoes_gerais"] = ["Não agendar clientes em rota.", "Capacidade de 35 clientes (Dia)"]
        resultado = AgentePercepcao(LLMRoteiro(json.dumps(spec))).extrair(["texto"])
        novo = resultado.segmentos[-1]
        self.assertEqual(("Não agendar clientes em rota", "clientes em rota", True),
                         (novo.nome, novo.criterio, novo.nao_agendar))
        self.assertEqual(["Capacidade de 35 clientes (Dia)"], resultado.restricoes_gerais)
        self.assertTrue(resultado.observacoes)

    def test_percepcao_desiste_depois_da_correcao(self):
        with self.assertRaises(PercepcaoFalhou):
            AgentePercepcao(LLMRoteiro("x", "y")).extrair(["texto"])

    def test_percepcao_envia_imagem(self):
        llm = LLMRoteiro(json.dumps(SPEC))
        AgentePercepcao(llm).extrair([""], ["QUJD"])
        conteudo = llm.chamadas[0][1][1]
        self.assertEqual("image_url", conteudo[1]["type"])
        self.assertTrue(conteudo[1]["image_url"]["url"].startswith("data:image/png;base64,QUJD"))

    def test_limpar_bloco(self):
        resposta = ('Segue:\n```gobr\nparametros "outro nome":\ndo time("t")\n  frequencia de 7 dias\n```\n'
                    "Espero ter ajudado.")
        self.assertEqual('parametros "Ativos":\n    do time("t")\n    frequencia de 7 dias',
                         limpar_bloco(resposta, "Ativos"))

    def test_gerador_usa_exemplos_como_pares(self):
        llm = LLMRoteiro(gerador_correto)
        contexto = ContextoSegmento(exemplos=[TrechoContexto(
            titulo="Ex", texto='Os clientes ativos.\n---\nparametros "Ex":\n    da carteira("c")\n    frequencia de 7 dias')],
            manual=[TrechoContexto(titulo="7.4.1 Frequência", texto="de N dias")])
        AgenteGerador(llm).gerar_bloco(Segmento(nome="Clientes ativos", frequencia_dias=14), 'do time("televendas")',
                                       contexto)
        papeis = [p for p, _ in llm.chamadas[0]]
        self.assertEqual(["system", "human", "ai", "human"], papeis)
        self.assertIn('Escopo: da carteira("c")', llm.chamadas[0][1][1])
        self.assertIn("7.4.1 Frequência", llm.chamadas[0][0][1])

    def test_modificador_nao_pedido_e_erro_de_conferencia(self):
        from src.agents.validador import conferir_bloco
        from src.gobr.parser import GobrParser

        regra = GobrParser().parse('parametros "A":\n    do time("t")\n    frequencia de 7 dias\n'
                                   "    preferencia nos dias uteis")
        erros = conferir_bloco(Segmento(nome="A", frequencia_dias=7), regra)
        self.assertTrue(any("modificadores que o segmento não pede" in e for e in erros))
        self.assertFalse(conferir_bloco(Segmento(nome="A", frequencia_dias=7, restricoes=["dias úteis"]), regra))

    def test_relatorio_marca_pergunta_respondida_sem_efeito(self):
        from src.grafo.relatorio import montar_resposta

        pergunta = {"id": "p1", "texto": "E os demais?", "bloqueante": False, "suposicao": "ficam sem regra"}
        e = {"analise": {"perguntas": [pergunta]}, "gobr": "version 1.0\n", "conferencia": {"ok": True},
             "perguntas_feitas": ["p1"], "mensagens": [{"texto": "x"}, {"texto": "não agendar", "origem": "respostas"}]}
        self.assertIn("perguntei, mas a resposta não resolveu", montar_resposta(e, max_tentativas=3))

    def test_frequencias_nomeadas(self):
        self.assertEqual(7, dias_da_frequencia("semanal()"))
        self.assertEqual(14, dias_da_frequencia("de 14 dias"))
        self.assertIsNone(dias_da_frequencia("do vendedor"))

    def test_recuperador_real_em_bm25(self):
        contexto = AgenteRecuperador(modo="bm25").contexto(
            Segmento(nome="Inadimplentes", criterio="clientes inadimplentes", nao_agendar=True))
        self.assertTrue(contexto.exemplos)
        from src.gobr.grammar import normalizar

        self.assertTrue(all("nao agendar" in normalizar(e.texto) for e in contexto.exemplos))
        self.assertTrue(contexto.manual)


class ConferenciaPorComportamentoTests(unittest.TestCase):
    ATIVOS = Segmento(nome="Ativos", criterio="x", faixa=Faixa(min_dias=0, max_dias=45), frequencia_dias=15)
    RISCO = Segmento(nome="Risco", criterio="x", faixa=Faixa(min_dias=45, max_dias=90), frequencia_dias=7)
    PERDIDOS = Segmento(nome="Perdidos", criterio="x", faixa=Faixa(min_dias=90), frequencia_dias=30)
    PROSPECTS = Segmento(nome="Prospects", criterio="nunca compraram", frequencia_dias=20)

    def erros(self, segmento, *linhas, spec=None):
        from src.agents.validador import conferir_bloco
        from src.gobr.parser import GobrParser

        acao = "nao agendar" if segmento.nao_agendar else f"frequencia de {segmento.frequencia_dias} dias"
        bloco = "\n".join([f'parametros "{segmento.nome}":', '    do time("t")', *("    " + l for l in linhas),
                           "    " + acao])
        spec = spec or RuleSpec(segmentos=[self.ATIVOS, self.RISCO, self.PERDIDOS, self.PROSPECTS])
        return conferir_bloco(segmento, GobrParser().parse(bloco), spec)

    def test_formas_certas_passam(self):
        for linhas in (["quando recencia de positivacao em 90 dias"],
                       ["quando recencia de positivacao entre 45 dias e 90 dias"],
                       ["quando recencia de positivacao em 90 dias", "e fora da recencia de positivacao em 45 dias"]):
            self.assertEqual([], self.erros(self.RISCO, *linhas))
        self.assertEqual([], self.erros(self.ATIVOS, "quando recencia de positivacao em 45 dias"))
        self.assertEqual([], self.erros(self.PERDIDOS, "quando fora da recencia de positivacao em 90 dias"))

    def test_faixa_invertida(self):
        erros = self.erros(self.RISCO, "quando fora da recencia de positivacao em 90 dias")
        self.assertIn("recencia de positivacao entre 45 dias e 90 dias", erros[0])
        self.assertTrue(self.erros(self.PERDIDOS, "quando recencia de positivacao em 90 dias"))

    def test_so_a_regra_anterior_permite_omitir_o_minimo(self):
        self.assertTrue(self.erros(self.RISCO, "quando recencia de positivacao em 90 dias",
                                   spec=RuleSpec(segmentos=[self.RISCO, self.PERDIDOS])))

    def test_sem_condicao_engole_as_regras_seguintes(self):
        self.assertTrue(self.erros(self.PERDIDOS))
        self.assertEqual([], self.erros(self.PERDIDOS, spec=RuleSpec(
            segmentos=[self.ATIVOS, self.RISCO, self.PERDIDOS])))
        self.assertIn("pegaria todos", self.erros(self.PROSPECTS)[0])

    def test_nao_agendar_com_faixa(self):
        s = Segmento(nome="Antigos", criterio="x", faixa=Faixa(min_dias=180), nao_agendar=True)
        spec = RuleSpec(segmentos=[s, self.ATIVOS])
        self.assertEqual([], self.erros(s, "quando fora da recencia de positivacao em 180 dias", spec=spec))
        self.assertTrue(self.erros(s, "quando recencia de positivacao em 180 dias", spec=spec))

    def test_descricao_do_segmento_nao_inverte_a_faixa(self):
        s = Segmento(nome="B", criterio="Clientes entre 20 e 40 dias sem comprar",
                     faixa=Faixa(min_dias=20, max_dias=40), frequencia_dias=7)
        self.assertEqual("Clientes com última compra entre 20 e 40 dias atrás. Devem ser atendidos a cada 7 dias.",
                         s.descricao())
        s = s.model_copy(update={"criterio": "Clientes ouro", "faixa": Faixa(max_dias=40)})
        self.assertTrue(s.descricao().startswith("Clientes ouro (compra nos últimos 40 dias)."))
        s = s.model_copy(update={"criterio": "há mais de 40 dias sem comprar", "faixa": Faixa(min_dias=40)})
        self.assertTrue(s.descricao().startswith("Clientes sem compra há mais de 40 dias."))

    def test_prospects(self):
        self.assertIn("ciclo de vida prospect", self.erros(
            self.PROSPECTS, "quando no ciclo de vida prospect com a empresa", "e data da rota na semana")[0])
        self.assertEqual([], self.erros(self.PROSPECTS, "quando no ciclo de vida prospect com a empresa"))

    def test_bloqueio_com_condicao_de_outro_assunto(self):
        rota = Segmento(nome="Não agendar clientes em rota", criterio="clientes em rota", nao_agendar=True)
        self.assertIn("data da rota na semana", self.erros(rota, "quando sem venda")[0])
        self.assertEqual([], self.erros(rota, "quando data da rota na semana"))

    def test_blocos_errados_vistos_na_avaliacao(self):
        """Blocos que os modelos escreveram na 20b de 24/11: recusados, e o reparo acerta."""
        from src.agents.validador import conferir_bloco, reparar_bloco
        from src.gobr.parser import GobrParser

        ativos = Segmento(nome="Ativos", criterio="Última compra em até 45 dias", faixa=Faixa(max_dias=45),
                          frequencia_dias=21)
        risco = Segmento(nome="Risco", criterio="entre 45 e 90 dias sem comprar",
                         faixa=Faixa(min_dias=45, max_dias=90), frequencia_dias=7)
        rota = Segmento(nome="Não agendar clientes em rota", criterio="clientes em rota", nao_agendar=True)
        spec = RuleSpec(segmentos=[rota, ativos, risco, self.PROSPECTS])
        for segmento, linhas, certo in (
            (ativos, ["quando no ciclo de vida ativo com a empresa", "e recencia de positivacao em 45 dias",
                      "frequencia de 21 dias"], "quando recencia de positivacao em 45 dias"),
            (ativos, ["quando recencia de positivacao em 45 dias", "ou fora da recencia de positivacao em 45 dias",
                      "frequencia de 21 dias"], "quando recencia de positivacao em 45 dias"),
            (risco, ["quando fora da recencia de positivacao entre 45 dias e 90 dias", "frequencia de 7 dias"],
             "quando recencia de positivacao entre 45 dias e 90 dias"),
            (rota, ["quando sem venda", "nao agendar"], "quando data da rota na semana"),
            (self.PROSPECTS, ["quando no ciclo de vida prospect com a empresa",
                              "e fora da recencia de positivacao na semana", "frequencia de 20 dias"],
             "quando no ciclo de vida prospect com a empresa"),
        ):
            regra = GobrParser().parse("\n".join([f'parametros "{segmento.nome}":', '    do time("t")',
                                                  *("    " + l for l in linhas)]))
            self.assertTrue(conferir_bloco(segmento, regra, spec), linhas)
            reparado = reparar_bloco(segmento, regra, spec, 'do time("t")')
            self.assertIsNotNone(reparado, linhas)
            self.assertIn(certo, reparado)

    def test_condicao_a_mais_na_faixa(self):
        s = self.RISCO.model_copy(update={"criterio": "última compra entre 45 e 90 dias"})
        self.assertIn("inadimplentes", self.erros(s, "quando recencia de positivacao em 90 dias", "e inadimplentes")[0])

    def test_preferencia_pedida_e_modificador_sobrando(self):
        s = self.ATIVOS.model_copy(update={"restricoes": ["preferência às segundas e terças"]})
        base = "quando recencia de positivacao em 45 dias"
        self.assertIn("preferencia na segunda e na terca", self.erros(s, base)[0])
        self.assertEqual([], self.erros(s, base, "preferencia na segunda e na terca"))
        r = self.ATIVOS.model_copy(update={"restricoes": ["somente fora da rota"]})
        self.assertIn("`fixo`", self.erros(r, base, "fixo")[0])


class PercepcaoConferenciaTests(unittest.TestCase):
    def test_faixa_do_texto(self):
        from src.agents.percepcao import faixa_do_texto

        for texto, faixa in {
            "Última compra em até 45 dias": (0, 45), "Clientes com compra entre 0 e 60 dias": (0, 60),
            "Compraram nos últimos 90 mas estão a 60 dias sem comprar": (60, 90),
            "Última compra entre 31 e 60 dias": (31, 60), "Clientes sem pedidos a mais 90 dias": (90, None),
            "Acima de 180 dias sem comprar": (180, None), "compra nos últimos 30 dias, visitar a cada 14 dias": (0, 30),
            "Clientes inadimplentes": None, "Inativos há 60 dias": None,
        }.items():
            self.assertEqual(faixa, faixa_do_texto(texto), texto)

    def spec(self, **segmento):
        return json.dumps({"segmentos": [{"nome": "Ativos", "criterio": "Última compra em até 45 dias",
                                          "faixa": {"min_dias": 45, "max_dias": 45}, "frequencia_dias": 21, **segmento}]})

    def test_faixa_corrigida_so_na_primeira_mensagem(self):
        s = AgentePercepcao(LLMRoteiro(self.spec())).extrair(["até 45 dias, a cada 21 dias"]).segmentos[0]
        self.assertEqual((0, 45), (s.faixa.min_dias, s.faixa.max_dias))
        s = AgentePercepcao(LLMRoteiro(self.spec())).extrair(["até 45 dias", "mude"]).segmentos[0]
        self.assertEqual((45, 45), (s.faixa.min_dias, s.faixa.max_dias))

    def test_completa_o_que_sumiu(self):
        texto = "Última compra em até 45 dias: a cada 21 dias, de preferência às sextas."
        llm = LLMRoteiro(self.spec(), self.spec(restricoes=["preferência na sexta"]))
        s = AgentePercepcao(llm).extrair([texto]).segmentos[0]
        self.assertEqual(["preferência na sexta"], s.restricoes)
        self.assertIn("preferência de dia", llm.chamadas[1][-1][1])
        llm = LLMRoteiro(self.spec(), "lixo")                      # não completou: vale a primeira leitura
        self.assertEqual(21, AgentePercepcao(llm).extrair([texto]).segmentos[0].frequencia_dias)


class RetentativaTests(unittest.TestCase):
    def test_ollama_que_cai_uma_vez(self):
        from langchain_core.runnables import RunnableLambda
        from ollama import ResponseError
        from src.llm_config import com_retentativas

        chamadas = []

        def invocar(_):
            chamadas.append(1)
            if len(chamadas) == 1:
                raise ResponseError("unexpected EOF", 500)
            return "ok"
        self.assertEqual("ok", com_retentativas(RunnableLambda(invocar)).invoke([("human", "x")]))
        self.assertEqual(2, len(chamadas))


class GrafoTests(unittest.TestCase):
    def test_ponta_a_ponta(self):
        ag = agentes(LLMRoteiro(json.dumps(SPEC)), LLMRoteiro(*[gerador_correto] * 3))
        _, _, saida = rodar(ag)
        self.assertEqual(["percepcao", "analista", "recuperador", "gerador", "validador", "explicador", "finalizar"],
                         nos(saida))
        self.assertTrue(saida["conferencia"]["ok"])
        gobr = saida["gobr"]
        self.assertTrue(gobr.startswith('version 1.0\n\ntime "televendas":'))
        self.assertLess(gobr.index("Não agendar inadimplentes"), gobr.index("Clientes ativos"))  # reordenado
        r = saida["resposta"]
        self.assertIn("✅ GOBR válido", r)
        self.assertIn("mais de 90 dias sem compra", r)        # falta de fallback
        self.assertIn("Capacidade de atendimento", r)          # fora do GOBR
        self.assertIn("**Assumido:**", r)                      # pergunta virou suposição
        self.assertIn("## O que cada regra faz", r)
        self.assertEqual(3, len(saida["explicacoes"]))

    def test_validador_devolve_erro_e_gerador_corrige_so_o_bloco_errado(self):
        gerador = LLMRoteiro(*[gerador_errando_ativos_uma_vez()] * 4)
        ag = agentes(LLMRoteiro(json.dumps(SPEC)), gerador)
        _, _, saida = rodar(ag)
        self.assertEqual(2, saida["tentativas"])
        self.assertTrue(saida["conferencia"]["ok"])
        self.assertEqual(4, len(gerador.chamadas))            # 3 blocos + 1 correção
        correcao = gerador.chamadas[-1][-1][1]
        self.assertIn("frequencia de 14 dias", correcao)
        self.assertIn("CONFERÊNCIA", correcao)

    def test_ultima_tentativa_repara_sem_llm(self):
        teimoso = lambda m: gerador_correto(m).replace("quando recencia de positivacao em 90",  # noqa: E731
                                                       "quando fora da recencia de positivacao entre 60 dias e 90")
        gerador = LLMRoteiro(*[teimoso] * 4)                  # 3 blocos + 1 correção que não corrige
        _, _, saida = rodar(agentes(LLMRoteiro(json.dumps(SPEC)), gerador))
        self.assertTrue(saida["conferencia"]["ok"])
        self.assertIn("quando recencia de positivacao entre 60 dias e 90 dias", saida["gobr"])
        self.assertEqual(["Risco de inatividade"], [n for p in saida["trace"] for n in p.get("reparados") or []])

    def test_desiste_depois_do_maximo_de_tentativas(self):
        sempre_errado = lambda m: gerador_correto(m).replace("quando recencia de positivacao em 60 dias",
                                                             "quando clientes muito felizes")
        ag = agentes(LLMRoteiro(json.dumps(SPEC)), LLMRoteiro(*[sempre_errado] * 10))
        _, _, saida = rodar(ag)
        self.assertEqual(3, saida["tentativas"])
        self.assertIn("Restaram erros depois de 3 tentativa(s)", saida["resposta"])
        self.assertEqual(2, len(saida["explicacoes"]))        # o bloco com erro não é explicado

    def test_pergunta_bloqueante_pausa_e_retoma_com_a_resposta(self):
        sem_frequencia = json.loads(json.dumps(SPEC))
        sem_frequencia["segmentos"][0]["frequencia_dias"] = None
        percepcao = LLMRoteiro(json.dumps(sem_frequencia), json.dumps(SPEC))
        ag = agentes(percepcao, LLMRoteiro(*[gerador_correto] * 3), interativo=True)
        grafo, config, saida = rodar(ag)
        self.assertIn("__interrupt__", saida)
        pausa = saida["__interrupt__"][0].value
        self.assertIn('"Clientes ativos" não tem frequência', pausa["mensagem"])
        saida = grafo.invoke(Command(resume="Clientes ativos: a cada 14 dias"), config)
        self.assertNotIn("__interrupt__", saida)
        self.assertTrue(saida["conferencia"]["ok"])
        segunda = percepcao.chamadas[1][1][1]
        self.assertIn("Respostas do usuário às perguntas", segunda)
        self.assertIn("tabela da Maxlub", segunda)

    def test_bloqueante_sem_modo_interativo_pede_informacao(self):
        sem_acao = {"segmentos": [{"nome": "Prospects", "criterio": "prospects"}]}
        ag = agentes(LLMRoteiro(json.dumps(sem_acao)), LLMRoteiro())
        _, _, saida = rodar(ag)
        self.assertIsNone(saida.get("gobr"))
        self.assertIn("## Preciso de mais informações", saida["resposta"])

    def test_orquestrador_llm_pode_seguir_sem_perguntar(self):
        orq = LLMRoteiro('{"proximo": "recuperador", "motivo": "pergunta não bloqueante; sigo com a suposição"}')
        ag = agentes(LLMRoteiro(json.dumps(SPEC)), LLMRoteiro(*[gerador_correto] * 3), orquestrador_llm=orq,
                     interativo=True)
        _, _, saida = rodar(ag)
        self.assertNotIn("__interrupt__", saida)
        self.assertEqual(1, len(orq.chamadas))                # só onde havia escolha
        decisao = next(p for p in saida["trace"] if p.get("origem") == "llm")
        self.assertEqual(["perguntar_usuario", "recuperador"], decisao["permitidos"])
        self.assertIn("**Assumido:**", saida["resposta"])

    def test_guardrail_corrige_escolha_invalida_do_orquestrador(self):
        orq = LLMRoteiro('{"proximo": "finalizar", "motivo": "pressa"}')
        ag = agentes(LLMRoteiro(json.dumps(SPEC)), LLMRoteiro(*[gerador_correto] * 3), orquestrador_llm=orq,
                     interativo=True)
        _, _, saida = rodar(ag)
        self.assertIn("__interrupt__", saida)                 # a regra manda perguntar
        guarda = next(p for p in saida["trace"] if p.get("origem") == "guardrail")
        self.assertIn("não é permitido", guarda["motivo"])

    def test_falha_da_percepcao_vira_resposta(self):
        ag = agentes(LLMRoteiro("x", "y"), LLMRoteiro())
        _, _, saida = rodar(ag)
        self.assertTrue(saida["resposta"].startswith("Não consegui concluir: percepcao"))

    def test_nova_mensagem_na_mesma_conversa_refaz_o_ciclo(self):
        mudou = json.loads(json.dumps(SPEC))
        mudou["segmentos"][1]["frequencia_dias"] = 10
        CORPOS_10 = dict(CORPOS, **{"Risco de inatividade": ["quando recencia de positivacao em 90 dias",
                                                            "frequencia de 10 dias"]})

        def gerador_10(m):
            nome, _ = _pedido(m)
            return "\n".join([f'parametros "{nome}":', '    do time("televendas")', *("    " + l for l in CORPOS_10[nome])])
        percepcao = LLMRoteiro(json.dumps(SPEC), json.dumps(mudou))
        ag = agentes(percepcao, LLMRoteiro(*[gerador_correto] * 3, *[gerador_10] * 3))
        grafo, config, primeira = rodar(ag)
        segunda = grafo.invoke(entrada("mude o risco de inatividade para 10 dias"), config)
        self.assertIn("frequencia de 10 dias", segunda["gobr"])
        self.assertEqual(1, segunda["tentativas"])
        self.assertEqual(2, len(percepcao.chamadas[1][1][1].split("\n\n")))  # as duas mensagens


class StudioTests(unittest.TestCase):
    def test_entrada_simples_com_pdf(self):
        percepcao = LLMRoteiro(json.dumps(SPEC))
        ag = agentes(percepcao, LLMRoteiro(*[gerador_correto] * 3))
        grafo = construir_grafo(ag, checkpointer=InMemorySaver(), entrada_simples=True)
        self.assertEqual({"pedido", "anexo"}, set(grafo.get_input_jsonschema()["properties"]))
        saida = grafo.invoke({"pedido": "Regras da Maxlub", "anexo": "data/exemplos/RegraMaxlub.pdf"},
                             config_thread("s1"))
        self.assertTrue(saida["conferencia"]["ok"])
        self.assertIn("Risco de Inatividade", percepcao.chamadas[0][1][1])   # texto do PDF
        self.assertEqual(["RegraMaxlub.pdf"], saida["conversa"][0]["anexos"])
        self.assertIsNone(saida.get("pedido"))

    def test_anexo_inexistente_vira_resposta(self):
        grafo = construir_grafo(agentes(LLMRoteiro(), LLMRoteiro()), checkpointer=InMemorySaver(),
                                entrada_simples=True)
        saida = grafo.invoke({"pedido": "x", "anexo": "nao_existe.pdf"}, config_thread("s2"))
        self.assertTrue(saida["resposta"].startswith("Não consegui concluir: inicio"))

    def test_modulo_do_studio_carrega(self):
        from src.grafo import studio

        self.assertIn("orquestrador", studio.grafo.get_graph().nodes)


class GuardrailsTests(unittest.TestCase):
    def test_destinos(self):
        ag = Agentes(interativo=True)
        self.assertEqual(["percepcao"], orquestrador.destinos_permitidos({}, ag))
        e = {"spec": {}, "analise": {"perguntas": [{"id": "x", "texto": "?", "bloqueante": True}]}}
        self.assertEqual(["perguntar_usuario"], orquestrador.destinos_permitidos(e, ag))
        e["rodadas_perguntas"] = 2
        self.assertEqual(["finalizar"], orquestrador.destinos_permitidos(e, ag))
        e = {"spec": {}, "analise": {}, "contextos": {}, "gobr": "x", "conferencia": {"ok": False}, "tentativas": 3}
        self.assertEqual(["explicador"], orquestrador.destinos_permitidos(e, ag))
        e["erro"] = "falhou"
        self.assertEqual(["finalizar"], orquestrador.destinos_permitidos(e, ag))


if __name__ == "__main__":
    unittest.main()
