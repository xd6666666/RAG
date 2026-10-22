import os

from pymilvus import (
    MilvusClient,
    DataType
)



class MilvusStore:


    def __init__(self):


        self.collection_name="rag_chunks"


        self.client=MilvusClient(

            uri=os.getenv(

                "MILVUS_URI",

                "http://localhost:19530"

            )

        )


        self.dim=1024


        self.create_collection()



    def create_collection(self):


        if self.client.has_collection(

            self.collection_name

        ):

            print(
                "[Milvus] Collection 已存在"
            )

            return



        schema=self.client.create_schema(

            auto_id=False

        )


        schema.add_field(

            "id",

            DataType.INT64,

            is_primary=True

        )


        schema.add_field(

            "vector",

            DataType.FLOAT_VECTOR,

            dim=self.dim

        )



        self.client.create_collection(

            collection_name=self.collection_name,

            schema=schema

        )


        self.client.create_index(

            collection_name=self.collection_name,

            field_name="vector",

            index_type="AUTOINDEX",

            metric_type="COSINE"

        )



        print(
            "[Milvus] Collection 创建成功"
        )



    def upsert_vectors(

            self,

            chunk_ids,

            vectors

    ):


        data=[]


        for cid,vec in zip(

            chunk_ids,

            vectors

        ):


            data.append(

                {

                "id":cid,

                "vector":vec

                }

            )


        self.client.upsert(

            collection_name=self.collection_name,

            data=data

        )


        print(

            f"[Milvus] 写入 {len(data)} vectors"

        )



    def search(

            self,

            vector,

            top_k=5

    ):


        return self.client.search(

            collection_name=self.collection_name,

            data=[vector],

            limit=top_k

        )