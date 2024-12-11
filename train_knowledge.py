import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = SentenceTransformer('moka-ai/m3e-base')

# 知识库和索引存储路径
KNOWLEDGE_BASE_DIR = "knowledge_base"
INDEX_DIR = "index"
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# 分块大小（每个 JSON 文件存储的知识量）
CHUNK_SIZE = 500

def split_and_save_knowledge_base(full_knowledge):
    """将完整知识库拆分为多个 JSON 文件"""
    for i in range(0, len(full_knowledge), CHUNK_SIZE):
        chunk = full_knowledge[i:i + CHUNK_SIZE]
        chunk_file = os.path.join(KNOWLEDGE_BASE_DIR, f"kb_part_{i // CHUNK_SIZE + 1}.json")
        with open(chunk_file, "w", encoding="utf-8") as file:
            json.dump(chunk, file, ensure_ascii=False, indent=4)

def build_indices():
    """为每个知识库分块构建对应的 FAISS 索引"""
    for file_name in os.listdir(KNOWLEDGE_BASE_DIR):
        if file_name.endswith(".json"):
            file_path = os.path.join(KNOWLEDGE_BASE_DIR, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                chunk = json.load(file)

            print(f"Building index for {file_name}...")
            embeddings = np.array(EMBEDDING_MODEL.encode([item["text"] for item in chunk]))
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

            index_file = os.path.join(INDEX_DIR, f"index_{file_name.replace('.json', '.bin')}")
            faiss.write_index(index, index_file)
            print(f"Index saved to {index_file}")

if __name__ == "__main__":
    # 示例：完整知识库加载
    full_knowledge_base = [
        {"id": i, "text": f"Example knowledge {i}. 知识实例 {i}"} for i in range(1500)
    ]
    split_and_save_knowledge_base(full_knowledge_base)
    build_indices()
