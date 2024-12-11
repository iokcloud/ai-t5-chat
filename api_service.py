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
# 加载 mT5 多语言生成模型
generate_model_name = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(generate_model_name, legacy=False)
GENERATE_MODEL = AutoModelForSeq2SeqLM.from_pretrained(generate_model_name)

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

# 更稳定的语言检测函数
def detect_language(text):
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        return 'chinese'
    elif any('a' <= char.lower() <= 'z' for char in text):
        return 'english'
    else:
        return 'unknown'

# 生成回答增强
def generate_answer(query, context, language):
    try:
        if language == 'chinese':
            prompt = f"以下是一些背景知识：\n{context}\n\n请基于上述背景知识回答下列问题：\n问题：{query}\n回答："
        else:
            prompt = f"Here is some background knowledge:\n{context}\n\nBased on the above context, please answer the following question:\nQuestion: {query}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = GENERATE_MODEL.generate(inputs.input_ids, max_length=200, num_beams=3, early_stopping=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 确保回答与语言一致
        if language == 'chinese' and not any('\u4e00' <= char <= '\u9fff' for char in answer):
            print(f"Warning: Generated answer not in Chinese: {answer}")
            answer = "生成回答未正确匹配中文，请重新尝试。"
        elif language == 'english' and not any('a' <= char.lower() <= 'z' for char in answer):
            print(f"Warning: Generated answer not in English: {answer}")
            answer = "The generated answer did not match English. Please try again."

        return answer
    except Exception as e:
        return f"Error generating answer: {e}"

# FastAPI 配置
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/qa/")
async def qa_endpoint(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 检测语言
    language = detect_language(query)
    if language == 'unknown':
        raise HTTPException(status_code=400, detail="Unsupported language")

    # 检索知识库
    results = search_knowledge(query)
    if language == 'chinese':
        context = " ".join([res for res in results if any('\u4e00' <= char <= '\u9fff' for char in res)])
    else:
        context = " ".join([res for res in results if any('a' <= char.lower() <= 'z' for char in res)])

    # 生成回答
    answer = generate_answer(query, context, language)
    return {"query": query, "language": language, "context": context, "answer": answer}

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
