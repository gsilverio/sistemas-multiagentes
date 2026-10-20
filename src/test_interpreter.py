from src.agents.gobr_interpreter import GobrInterpreterAgent


def main():
    agent = GobrInterpreterAgent()

    gobr = """
        parametros "Risco de Inatividade":
            do time("televendas")
            quando recencia de positivacao em 45 dias
            e fora da recencia de positivacao em 30 dias
            frequencia de 7 dias
        """

    result = agent.interpret(gobr)

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()