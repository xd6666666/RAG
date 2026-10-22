from pathlib import Path

from task2_rag.embedding.embedder import BGEEmbedder
from task2_rag.database.milvus_store import MilvusStore
from task2_rag.database.mysql_store import MySQLStore


class Retriever:


    def __init__(self):

        print("[Retriever] 初始化...")


        self.embedder = BGEEmbedder()


        self.milvus = MilvusStore()


        self.mysql = MySQLStore()


        print(
            "[Retriever] 初始化完成"
        )



    def search(
            self,
            query: str,
            top_k: int = 5
    ):


        print(
            f"\n用户问题: {query}"
        )


        # 1. 问题向量化

        query_vector = self.embedder.encode(
            [
                query
            ]
        )[0]


        print(
            "[BGE-M3] Query向量生成完成"
        )



        # 2. Milvus检索

        results = self.milvus.search(
            query_vector,
            top_k
        )


        print(
            f"[Milvus] 返回 {len(results[0])} 个结果"
        )


        chunk_ids = []


        for item in results[0]:

            chunk_ids.append(
                item["id"]
            )



        print(
            "匹配Chunk ID:",
            chunk_ids
        )



        # 3. MySQL获取原文

        chunks = self.mysql.get_chunks_by_ids(
            chunk_ids
        )


        return chunks



if __name__ == "__main__":


    retriever = Retriever()


    results = retriever.search(

        "什么情况下可以领取保险金？",

        top_k=5

    )


    print(
        "\n========== 检索结果 =========="
    )


    for item in results:

        print("\n----------------")

        print(
            "ID:",
            item["id"]
        )

        print(
            item["content"][:500]
        )
