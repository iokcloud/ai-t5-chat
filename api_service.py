import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import MT5Tokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 禁用联网请求
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 知识库和索引存储路径
KNOWLEDGE_BASE_DIR = "knowledge_base"
INDEX_DIR = "index"

# 模型初始化
EMBEDDING_MODEL = SentenceTransformer('moka-ai/m3e-base')
TOKENIZER = MT5Tokenizer.from_pretrained("google/mt5-small", legacy=False)
GENERATE_MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

# 加载所有分块知识库和索引
def load_all_knowledge_and_indices():
    knowledge_bases = []
    indices = []
    for file_name in os.listdir(KNOWLEDGE_BASE_DIR):
        if file_name.endswith(".json"):
            # 加载知识库
            with open(os.path.join(KNOWLEDGE_BASE_DIR, file_name), "r", encoding="utf-8") as file:
                knowledge_bases.append(json.load(file))
            # 加载对应索引
            index_file = os.path.join(INDEX_DIR, f"index_{file_name.replace('.json', '.bin')}")
            index = faiss.read_index(index_file)
            indices.append(index)
    return knowledge_bases, indices

KNOWLEDGE_BASES, INDICES = load_all_knowledge_and_indices()

# 搜索知识库
def search_knowledge(query, top_k=3):
    query_embedding = np.array(EMBEDDING_MODEL.encode([query]))
    results = []

    for knowledge_base, index in zip(KNOWLEDGE_BASES, INDICES):
        distances, indices = index.search(query_embedding, top_k)
        results.extend([
            knowledge_base[idx]['text'] for idx in indices[0] if idx < len(knowledge_base)
        ])

    # 去重并按距离排序
    return list(set(results))[:top_k]

# 生成回答
def generate_answer(query, context):
    prompt = f"Here is some background knowledge:\n{context}\n\nPlease answer the following question:\n{query}"
    inputs = TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = GENERATE_MODEL.generate(inputs.input_ids, max_length=200)
    return TOKENIZER.decode(outputs[0], skip_special_tokens=True)

# FastAPI 配置
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/qa/")
async def qa_endpoint(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 检索知识
    results = search_knowledge(query)
    context = " ".join(results)

    # 生成回答
    answer = generate_answer(query, context)
    return {"query": query, "context": context, "answer": answer}

@app.post("/add_knowledge/")
async def add_knowledge(text: str):
    global KNOWLEDGE_BASES, INDICES

    # 分配到最新分块
    new_entry = {"id": len(KNOWLEDGE_BASES[-1]), "text": text}
    KNOWLEDGE_BASES[-1].append(new_entry)

    # 更新索引
    new_embedding = np.array(EMBEDDING_MODEL.encode([text]))
    INDICES[-1].add(new_embedding)

    # 保存更新
    with open(os.path.join(KNOWLEDGE_BASE_DIR, f"kb_part_{len(KNOWLEDGE_BASES)}.json"), "w", encoding="utf-8") as file:
        json.dump(KNOWLEDGE_BASES[-1], file, ensure_ascii=False, indent=4)
    faiss.write_index(INDICES[-1], os.path.join(INDEX_DIR, f"index_part_{len(KNOWLEDGE_BASES)}.bin"))

    return {"detail": "Knowledge added successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
