import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import jieba
import nltk
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
    tokens = []
    for match in re.finditer(r'[一-龥]+|[a-zA-Z]+', text):
        chunk = match.group()
        if re.match(r'[一-龥]', chunk):
            tokens.extend(jieba.cut(chunk))
        else:
            tokens.extend(nltk.word_tokenize(chunk))
    return tokens

# 更稳定的语言检测函数
def detect_language(text):
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        return 'chinese'
    elif any('a' <= char.lower() <= 'z' for char in text):
        return 'english'
    else:
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
            prompt = f"以下是一些背景知识：\n{context}\n\n请基于上述背景知识回答下列问题：\n问题：{query}\n回答："
        else:
            prompt = f"Here is some background knowledge:\n{context}\n\nBased on the above context, please answer the following question:\nQuestion: {query}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = generate_model.generate(inputs.input_ids, max_length=200, num_beams=3, early_stopping=True)
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

# 请求数据模型
class QueryRequest(BaseModel):
    query: str

# API 路由
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

# 示例：启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
