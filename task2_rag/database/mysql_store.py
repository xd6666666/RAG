import os

import pymysql

from dotenv import load_dotenv


load_dotenv()



class MySQLStore:


    def __init__(self):


        self.conn = pymysql.connect(

            host=os.getenv(
                "MYSQL_HOST",
                "127.0.0.1"
            ),

            port=int(
                os.getenv(
                    "MYSQL_PORT",
                    3307
                )
            ),

            user=os.getenv(
                "MYSQL_USER",
                "root"
            ),

            password=os.getenv(
                "MYSQL_PASSWORD"
            ),

            database=os.getenv(
                "MYSQL_DATABASE",
                "rag_db"
            ),

            charset="utf8mb4"

        )


        print(
            "[MySQL] 连接成功"
        )


        self.create_tables()



    def create_tables(self):


        cursor = self.conn.cursor()



        cursor.execute(

        """

        CREATE TABLE IF NOT EXISTS documents
        (

            document_id VARCHAR(64) PRIMARY KEY,

            filename VARCHAR(255),

            chunk_count INT

        )

        """

        )



        cursor.execute(

        """

        CREATE TABLE IF NOT EXISTS rag_chunks
        (

            chunk_id BIGINT PRIMARY KEY,

            document_id VARCHAR(64),

            chunk_index INT,

            content TEXT,

            source VARCHAR(255)

        )

        """

        )



        self.conn.commit()

        cursor.close()


        print(
            "[MySQL] 表创建完成"
        )



    def save_document(

            self,
            document_id,
            filename,
            chunk_count

    ):


        sql = """

        INSERT INTO documents

        VALUES(%s,%s,%s)

        ON DUPLICATE KEY UPDATE

        chunk_count=%s

        """



        cursor = self.conn.cursor()


        cursor.execute(

            sql,

            (
                document_id,
                filename,
                chunk_count,
                chunk_count
            )

        )


        self.conn.commit()

        cursor.close()



    def save_chunks(
            self,
            records:list[dict]
    ):


        sql = """

        INSERT INTO rag_chunks

        (
            chunk_id,
            document_id,
            chunk_index,
            content,
            source

        )

        VALUES
        (
            %s,%s,%s,%s,%s
        )


        ON DUPLICATE KEY UPDATE

        content=VALUES(content)

        """



        data=[]


        for r in records:


            data.append(

                (

                r["chunk_id"],

                r["document_id"],

                r["chunk_index"],

                r["content"],

                r.get(
                    "source",
                    ""
                )

                )

            )



        cursor=self.conn.cursor()


        cursor.executemany(
            sql,
            data
        )


        self.conn.commit()


        cursor.close()



        print(
            f"[MySQL] 保存 {len(data)} chunks"
        )



    def count_chunks(

            self,

            document_id

    ):


        cursor=self.conn.cursor()


        cursor.execute(

        """

        SELECT COUNT(*)

        FROM rag_chunks

        WHERE document_id=%s

        """,

        (document_id,)

        )


        result=cursor.fetchone()[0]


        cursor.close()


        return result

    def get_chunks_by_ids(
            self,
            ids:list[int]
    ):


        if not ids:
            return []


        placeholders = ",".join(
            ["%s"] * len(ids)
        )


        sql = f"""

        SELECT
            chunk_id AS id,
            content,
            source

        FROM rag_chunks

        WHERE chunk_id IN ({placeholders})

        """


        cursor = self.conn.cursor(
            pymysql.cursors.DictCursor
        )


        cursor.execute(
            sql,
            ids
        )


        result = cursor.fetchall()


        cursor.close()


        return result

    def close(self):

        self.conn.close()