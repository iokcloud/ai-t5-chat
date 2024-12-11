import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import jieba
import nltk
from fastapi import FastAPI, HTTPException
import re

# 禁用联网请求
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 初始化 FastAPI 应用
app = FastAPI()

# 加载多语言嵌入模型 M3E
embedding_model = SentenceTransformer('moka-ai/m3e-base')
# 加载 mT5 多语言生成模型
generate_model_name = "google/mt5-small"

from transformers import MT5Tokenizer

# 强制使用慢速分词器
tokenizer = MT5Tokenizer.from_pretrained(generate_model_name, legacy=False)
generate_model = AutoModelForSeq2SeqLM.from_pretrained(generate_model_name)

# 知识库（示例）
knowledge_base = [
    "人工智能是研究如何让计算机完成需要人类智能才能完成的任务的一门学科。",
    "Artificial intelligence (AI) is the simulation of human intelligence processes by machines.",
    "机器学习是人工智能的一个分支，专注于通过数据训练模型。",
    "Machine learning is a subset of AI that focuses on training models using data.",
    "深度学习是一种基于神经网络的机器学习方法。",
    "Deep learning is a machine learning technique based on neural networks."
]

# 生成嵌入向量并构建 FAISS 索引
print("Building FAISS index...")
kb_embeddings = np.array(embedding_model.encode(knowledge_base))
dimension = kb_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(kb_embeddings)
print("FAISS index built successfully!")

# 分词函数支持中英文混合
def segment_text(text):
    zh_pattern = re.compile(r'[\u4e00-\u9fa5]+')
    en_pattern = re.compile(r'[a-zA-Z]+')
    zh_tokens = []
    en_tokens = []

    for match in zh_pattern.finditer(text):
        zh_tokens.extend(jieba.cut(match.group()))
    for match in en_pattern.finditer(text):
        en_tokens.extend(nltk.word_tokenize(match.group()))

    return zh_tokens + en_tokens

# 更稳定的语言检测函数
def detect_language(text):
    try:
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'chinese'
        elif re.search(r'[a-zA-Z]', text):
            return 'english'
        else:
            return 'unknown'
    except Exception:
        return 'unknown'

# 检索知识库
def search_knowledge(query, top_k=3):
    query_embedding = np.array(embedding_model.encode([query]))
    distances, indices = index.search(query_embedding, top_k)
    results = [knowledge_base[idx] for idx in indices[0]]
    return results

# 生成回答增强
def generate_answer(query, context, language):
    try:
        if language == 'chinese':
            prompt = f"已知：{context}\n问题：{query}\n回答："
        else:
            prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = generate_model.generate(inputs.input_ids, max_length=200)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating answer: {e}"

# API 路由
@app.post("/qa/")
async def qa_endpoint(query: str):
    # 检测语言
    language = detect_language(query)
    if language == 'unknown':
        raise HTTPException(status_code=400, detail="Unsupported language")

    # 检索知识库
    results = search_knowledge(query)
    context = " ".join(results)

    # 生成回答
    answer = generate_answer(query, context, language)
    return {"query": query, "language": language, "context": context, "answer": answer}

# 示例：启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)