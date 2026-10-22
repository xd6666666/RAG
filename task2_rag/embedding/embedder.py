
from pathlib import Path

import torch
from FlagEmbedding import BGEM3FlagModel


PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_DIR / "models" / "bge-m3"


class BGEEmbedder:

    def __init__(self):
        print("[BGE-M3] 正在加载模型...")

        if not MODEL_PATH.is_dir():
            raise FileNotFoundError(
                f"BGE-M3 模型不存在：{MODEL_PATH}"
            )

        self.device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = BGEM3FlagModel(
            str(MODEL_PATH),
            use_fp16=self.device.startswith("cuda"),
            devices=[self.device],
        )

        print(f"[BGE-M3] 加载成功，设备：{self.device}")

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        输入多个文本 Chunk。
        输出对应的 1024 维向量。
        """
        if not texts:
            return []

        result = self.model.encode(
            texts,
            batch_size=2,
            max_length=1024,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        vectors = result["dense_vecs"]

        if vectors.shape[1] != 1024:
            raise ValueError(
                f"向量维度错误：{vectors.shape}"
            )

        return vectors.astype("float32").tolist()
if __name__ == "__main__":

    embedder = BGEEmbedder()


    texts = [
        "什么情况下可以领取保险金",
        "本合同的保险责任是什么"
    ]


    vectors = embedder.encode(
        texts
    )


    print(
        "向量数量:",
        len(vectors)
    )


    print(
        "向量维度:",
        len(vectors[0])
    )


    print(
        "前10个值:"
    )


    print(
        vectors[0][:10]
    )