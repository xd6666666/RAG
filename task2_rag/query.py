from task2_rag.retrieval.retriever import Retriever
from task2_rag.generation.llm import QwenLLM



def main():


    question = "什么情况下可以领取保险金？"


    retriever = Retriever()

    results = retriever.search(
        question,
        top_k=2
    )


    contexts = [

        item["content"]

        for item in results

    ]


    llm = QwenLLM()


    answer = llm.chat(

        question,

        contexts

    )


    print("\n========== ANSWER ==========\n")

    print(answer)



if __name__ == "__main__":

    main()
