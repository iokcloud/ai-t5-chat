from sentence_transformers import SentenceTransformer

model = SentenceTransformer('shibing624/m3e-base') # 专为中英文优化的轻量化模型
documents = ["你好", "Hello", "欢迎使用多语言问答系统"]
embeddings = model.encode(documents)


# 存储到本地
import pickle
with open('knowledge_base.pkl', 'wb') as f:
    pickle.dump((documents, embeddings), f)


import faiss
import numpy as np

# 示例嵌入
embeddings = np.array(model.encode(documents))
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


# 保存索引
faiss.write_index(index, 'knowledge_index.faiss')
