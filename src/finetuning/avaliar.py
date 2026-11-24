"""Avaliação comparativa do Gerador: modelo base × modelo treinado (× Ollama).

Para cada par de teste o modelo recebe o mesmo prompt do agente e escreve o
bloco; as métricas são determinísticas:

- `valido`: o bloco passa no `GobrValidator` (montado com `version 1.0` e as
  definições de time/carteira que ele usa; erros de referência a blocos que
  o próprio acervo não define são ignorados);
- `vocabulario_ok`: nenhuma condição fora do vocabulário (A001/A002);
- `ast_igual`: a AST do bloco gerado é igual à da referência (mesmo escopo,
  condições, frequência, `nao agendar`, modificadores e filtros, a menos de
  espaços e acentos): **a métrica principal**;
- `exato`: corpo idêntico ao da referência, linha a linha (normalizado);
- `f1_linhas`: crédito parcial pelas linhas certas;
- `numeros_ok`: os números do bloco são os mesmos da referência.

    python -m src.finetuning.avaliar --modelo hf:Qwen/Qwen3-4B --rotulo base
    python -m src.finetuning.avaliar --modelo hf:Qwen/Qwen3-4B --adaptador <dir> --rotulo treinado
    python -m src.finetuning.avaliar --modelo ollama:gemma3:12b --rotulo gemma3
    python -m src.finetuning.avaliar --comparar <dir de resultados>

`--com-exemplos` coloca no prompt 2 exemplos recuperados do treino, como o
Recuperador faz no grafo. Os resultados (um JSONL por item e um resumo)
ficam em `--saida`; refazer a avaliação pula os itens já feitos.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from src.agents.comum import texto_da_resposta
from src.agents.gerador import limpar_bloco
from src.finetuning.dataset import (CODIGOS_DE_REFERENCIA, SAIDA_PADRAO, carregar, exemplos_para,
                                    indice_de_exemplos)
from src.finetuning.modelo_hf import mensagens_do_par
from src.gobr.ast import GobrParsedRule
from src.gobr.grammar import normalizar
from src.gobr.parser import GobrParser
from src.validation.gobr_validator import GobrValidator

RAIZ = Path(__file__).resolve().parents[2]
METRICAS = ("valido", "vocabulario_ok", "ast_igual", "exato", "f1_linhas", "numeros_ok")


def _corpo(bloco: str) -> list[str]:
    return [normalizar(l) for l in bloco.strip().splitlines()[1:] if l.strip() and not l.strip().startswith("#")]


def _ast_canonica(regra: GobrParsedRule) -> dict[str, Any]:
    return {
        "escopos": sorted((e.tipo, e.operador, tuple(normalizar(v) for v in e.valores)) for e in regra.escopos),
        "condicoes": [(c.operador or "", normalizar(c.texto)) for c in regra.condicoes],
        "frequencia": normalizar(regra.frequencia or ""),
        "nao_agendar": regra.nao_agendar,
        "modificadores": sorted(normalizar(m) for m in regra.modificadores),
        "filtros": sorted(normalizar(f) for f in regra.filtros_temporais),
        "desconhecidos": [normalizar(d) for d in regra.desconhecidos],
    }


def _f1(gerado: list[str], referencia: list[str]) -> float:
    if not gerado and not referencia:
        return 1.0
    comuns = sum(min(gerado.count(l), referencia.count(l)) for l in set(gerado))
    if not comuns:
        return 0.0
    precisao, revocacao = comuns / len(gerado), comuns / len(referencia)
    return 2 * precisao * revocacao / (precisao + revocacao)


def metricas(gerado: str, par: dict[str, Any], *, validador: GobrValidator, parser: GobrParser) -> dict[str, Any]:
    arquivo = "\n\n".join(["version 1.0", *par.get("contexto", []), gerado]) + "\n"
    relatorio = validador.validar(arquivo)
    erros = [p for p in relatorio.erros if p.codigo not in CODIGOS_DE_REFERENCIA]
    vocabulario = [p for p in relatorio.avisos if p.codigo in ("A001", "A002")]
    try:
        ast_igual = _ast_canonica(parser.parse(gerado)) == _ast_canonica(parser.parse(par["bloco"]))
    except ValueError:
        ast_igual = False
    g, r = _corpo(gerado), _corpo(par["bloco"])
    numeros = lambda linhas: sorted(re.findall(r"\d+", " ".join(linhas)))  # noqa: E731
    return {
        "valido": not erros, "vocabulario_ok": not vocabulario, "ast_igual": ast_igual, "exato": g == r,
        "f1_linhas": round(_f1(g, r), 4), "numeros_ok": numeros(g) == numeros(r),
        "erros": [p.como_texto() for p in erros + vocabulario][:5],
    }


def criar_modelo(especificacao: str, adaptador: str | None = None) -> Any:
    tipo, _, nome = especificacao.partition(":")
    if tipo == "hf":
        from src.finetuning.modelo_hf import GeradorHF

        return GeradorHF(nome, adaptador)
    if tipo == "ollama":
        from src.llm_config import create_llm

        return create_llm(nome)
    raise ValueError("Use hf:<modelo do Hugging Face> ou ollama:<modelo>.")


def avaliar(modelo: Any, pares: list[dict[str, Any]], saida: Path, *, rotulo: str,
            exemplos: Callable[[dict], list] | None = None) -> dict[str, Any]:
    saida.mkdir(parents=True, exist_ok=True)
    arquivo = saida / f"{rotulo}.jsonl"
    feitos = {}
    if arquivo.exists():
        for linha in arquivo.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(linha)
            except ValueError:  # linha cortada (a sessão caiu durante a gravação): o item é refeito
                continue
            feitos[(r["split"], r["id"])] = r
        arquivo.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in feitos.values()),
                           encoding="utf-8")
    validador, parser = GobrValidator(), GobrParser()
    with arquivo.open("a", encoding="utf-8") as destino:
        for i, par in enumerate(pares, start=1):
            if (par["split"], par["id"]) in feitos:
                continue
            par = dict(par, exemplos=exemplos(par) if exemplos else [])
            inicio = time.time()
            resposta = texto_da_resposta(modelo.invoke(mensagens_do_par(par)))
            gerado = limpar_bloco(resposta, par["nome"])
            r = {"id": par["id"], "split": par["split"], "segundos": round(time.time() - inicio, 2),
                 "gerado": gerado, "referencia": par["bloco"], **metricas(gerado, par, validador=validador,
                                                                          parser=parser)}
            feitos[(par["split"], par["id"])] = r
            destino.write(json.dumps(r, ensure_ascii=False) + "\n")
            destino.flush()
            if i % 10 == 0:
                print(f"{rotulo}: {i}/{len(pares)}", flush=True)
    resumo = resumir(list(feitos.values()))
    (saida / f"{rotulo}.resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    return resumo


def resumir(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    resumo: dict[str, Any] = {}
    for split in sorted({r["split"] for r in resultados}):
        itens = [r for r in resultados if r["split"] == split]
        resumo[split] = {"n": len(itens), **{m: round(sum(float(r[m]) for r in itens) / len(itens), 3)
                                             for m in METRICAS},
                         "segundos_medio": round(sum(r["segundos"] for r in itens) / len(itens), 2)}
    return resumo


def tabela_comparativa(diretorio: Path) -> str:
    linhas = ["| modelo | conjunto | n | " + " | ".join(METRICAS) + " | s/item |",
              "|---|---|---|" + "---|" * len(METRICAS) + "---|"]
    for arquivo in sorted(diretorio.glob("*.resumo.json")):
        rotulo = arquivo.name.removesuffix(".resumo.json")
        for split, r in json.loads(arquivo.read_text(encoding="utf-8")).items():
            linhas.append(f"| {rotulo} | {split} | {r['n']} | " + " | ".join(f"{r[m]:.3f}" for m in METRICAS)
                          + f" | {r['segundos_medio']} |")
    return "\n".join(linhas)


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--modelo", help="hf:<modelo> ou ollama:<modelo>")
    args.add_argument("--adaptador", help="diretório do adaptador LoRA (só com hf:)")
    args.add_argument("--rotulo", help="nome do resultado (ex.: base, treinado)")
    args.add_argument("--dataset", type=Path, default=SAIDA_PADRAO)
    args.add_argument("--conjuntos", default="teste,real")
    args.add_argument("--limite", type=int, default=0, help="máximo de itens por conjunto (0 = todos)")
    args.add_argument("--com-exemplos", action="store_true")
    args.add_argument("--saida", type=Path, default=RAIZ / "outputs" / "finetuning" / "avaliacao")
    args.add_argument("--comparar", type=Path, help="só imprime a tabela dos resumos deste diretório")
    a = args.parse_args()
    if a.comparar:
        print(tabela_comparativa(a.comparar))
        return
    if not (a.modelo and a.rotulo):
        args.error("--modelo e --rotulo são obrigatórios (ou use --comparar)")

    pares = []
    for conjunto in a.conjuntos.split(","):
        itens = carregar(a.dataset, conjunto.strip())
        pares.extend(itens[: a.limite] if a.limite else itens)
    exemplos = None
    if a.com_exemplos:
        indice = indice_de_exemplos(carregar(a.dataset, "treino"))
        exemplos = lambda par: exemplos_para(par, indice)  # noqa: E731
    modelo = criar_modelo(a.modelo, a.adaptador)
    resumo = avaliar(modelo, pares, a.saida, rotulo=a.rotulo, exemplos=exemplos)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print(tabela_comparativa(a.saida))


if __name__ == "__main__":
    main()
