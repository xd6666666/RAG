
from pathlib import Path
import hashlib

from dotenv import load_dotenv

from parsing.parse_pdf import parse_pdf
from chunking.chunker import split_text
from embedding.embedder import BGEEmbedder
from database.milvus_store import MilvusStore
from database.mysql_store import MySQLStore


TASK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TASK_DIR.parent
PDF_DIR = TASK_DIR / "documents"

load_dotenv(PROJECT_DIR / ".env")


def generate_chunk_id(
    document_id: str,
    chunk_index: int,
) -> int:
    """
    生成稳定的正整数 ID，供 MySQL 和 Milvus 共用。
    """
    raw = f"{document_id}:{chunk_index}"

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8], "big"
    ) & ((1 << 63) - 1)


def ingest_pdf(
    pdf_path: Path,
    embedder: BGEEmbedder,
    milvus: MilvusStore,
    mysql: MySQLStore,
):
    print(f"\n========== {pdf_path.name} ==========")

    # 1. 生成文档 ID
    document_id = hashlib.sha256(
        pdf_path.read_bytes()
    ).hexdigest()

    # 2. MinerU 解析 PDF
    markdown_path = parse_pdf(pdf_path)

    text = markdown_path.read_text(
        encoding="utf-8"
    )

    # 3. Chunking
    chunks = split_text(
        text,
        chunk_size=1024,
        overlap=128,
    )

    if not chunks:
        raise ValueError("没有生成有效 Chunk")

    print(f"[Chunking] 共 {len(chunks)} 个 Chunk")

    # 4. BGE-M3 向量化
    # 避免一次把所有文档 Chunk 送入显存
    batch_size = 8

    # 5. 先保存文档元信息
    mysql.save_document(
        document_id,
        pdf_path.name,
        len(chunks),
    )

    for start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[
            start:start + batch_size
        ]

        vectors = embedder.encode(batch_chunks)

        records = []
        chunk_ids = []

        for offset, content in enumerate(batch_chunks):
            index = start + offset

            chunk_id = generate_chunk_id(
                document_id,
                index,
            )

            records.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_index": index,
                "content": content,
            })

            chunk_ids.append(chunk_id)

        # 6. 将 Chunk 原文写入 MySQL
        mysql.save_chunks(records)

        # 7. 将向量写入 Milvus
        milvus.upsert_vectors(
            chunk_ids,
            vectors,
        )

        print(
            f"[进度] {start + len(batch_chunks)}"
            f"/{len(chunks)}"
        )

    # 8. 核对 MySQL 的 Chunk 数量
    mysql_count = mysql.count_chunks(
        document_id
    )

    print(
        f"[验证] MySQL Chunk 数量：{mysql_count}"
    )

    print("[完成] PDF 知识入库完成")


def main():
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("documents 文件夹里没有 PDF")
        return

    print(f"找到 {len(pdf_files)} 个 PDF")

    # 模型只加载一次
    embedder = BGEEmbedder()

    # 数据库只初始化一次
    milvus = MilvusStore()
    mysql = MySQLStore()

    try:
        for pdf_path in pdf_files:
            ingest_pdf(
                pdf_path,
                embedder,
                milvus,
                mysql,
            )

    finally:
        mysql.close()


if __name__ == "__main__":
    main()
