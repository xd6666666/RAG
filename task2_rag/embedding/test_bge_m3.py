
import torch
from FlagEmbedding import BGEM3FlagModel

# 本地下载好的 BGE-M3 模型
model_path = "/home/davidxie/llm-rag-project/models/bge-m3"

# 检查 GPU
print("CUDA available:", torch.cuda.is_available())

# 加载模型
model = BGEM3FlagModel(
    model_path,
    use_fp16=torch.cuda.is_available(),
    devices=["cuda:0"] if torch.cuda.is_available() else ["cpu"]
)

# 需要转换为向量的文本
sentences = [
    "新能源汽车使用电池提供动力。",
    "电动汽车依靠电池驱动车辆。",
    "今天的天气很好。"
]

# 执行向量化
result = model.encode(
    sentences,
    batch_size=2,
    max_length=512,
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False
)

embeddings = result["dense_vecs"]

print("Embedding shape:", embeddings.shape)
print("First vector:", embeddings[0][:10])
