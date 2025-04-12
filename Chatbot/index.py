# index.py

import os
import pandas as pd
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from get_embedding_function import get_embedding_function

CSV_PATH = "C:/Users/Ngo Thanh Nam/Test_CCAT/SCIC-2025-Integrating-LLM/algebra.csv"
INDEX_PATH = "faiss_index"

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Không tìm thấy file {CSV_PATH}")

    # 1️⃣ Đọc CSV và tạo Document
    df = pd.read_csv(CSV_PATH)
    docs = [
        Document(
            page_content=row["problem"],
            metadata={
                "solution": row["solution"],
                "level": row["level"],
                "type": row["type"],
            },
        )
        for _, row in df.iterrows()
    ]

    # 2️⃣ Split thành chunks ~1000 tokens
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = splitter.split_documents(docs)

    # 3️⃣ Build & save FAISS index
    emb = get_embedding_function()
    db = FAISS.from_documents(docs, emb)
    db.save_local(INDEX_PATH)
    print(f"✅ Đã tạo FAISS index tại '{INDEX_PATH}/'")

if __name__ == "__main__":
    main()
