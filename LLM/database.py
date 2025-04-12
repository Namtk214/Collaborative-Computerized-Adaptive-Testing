import argparse
import os
import shutil
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS  # Sử dụng FAISS thay vì Chroma
from langchain_openai import OpenAIEmbeddings
from get_embedding_function import get_embedding_function

from dotenv import load_dotenv
load_dotenv()

# Flow: Xoá thư mục DB -> load file CSV -> chia nhỏ nội dung (nếu cần) -> thêm các đoạn văn bản mới vào FAISS

FAISS_PATH = "faiss_index"
DATA_PATH = "C:/Users/Ngo Thanh Nam/Test_CCAT/SCIC-2025-Integrating-LLM/algebra (1).csv"  # CSV với các cột: problem, level, type, solution

embedding = get_embedding_function()

# Nếu chỉ mục FAISS đã tồn tại, load nó; nếu chưa có thì khởi tạo sau khi thêm tài liệu.
if os.path.exists(FAISS_PATH):
    FAISS_DB = FAISS.load_local(FAISS_PATH, embedding, allow_dangerous_deserialization=True)
else:
    FAISS_DB = None

def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)

def add_to_faiss(chunks: list[Document]):
    global FAISS_DB
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # Nếu chỉ mục đã tồn tại, lấy các id đã có từ docstore.
    if FAISS_DB is not None:
        existing_ids = set(FAISS_DB.docstore._dict.keys())
    else:
        existing_ids = set()
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        batch_size = 1000  # Xử lý theo lô để tránh vượt quá kích thước batch tối đa.
        # Nếu FAISS_DB chưa được khởi tạo, tạo mới từ batch đầu tiên
        if FAISS_DB is None:
            batch_chunks = new_chunks[0:batch_size]
            FAISS_DB = FAISS.from_documents(batch_chunks, embedding)
            start_index = batch_size
        else:
            start_index = 0
        # Nếu FAISS_DB đã tồn tại, thêm các document mới.
        for i in range(start_index, len(new_chunks), batch_size):
            batch_chunks = new_chunks[i:i+batch_size]
            FAISS_DB.add_documents(batch_chunks)
        # Lưu chỉ mục FAISS về đĩa.
        FAISS_DB.save_local(FAISS_PATH)
    else:
        print("✅ No new documents to add")

def retrieve_docs(query_text: str, limit: int = 5):
    if FAISS_DB is None:
        print("FAISS index is empty. No documents to retrieve.")
        return []
    results = FAISS_DB.similarity_search_with_score(query_text, k=limit)
    return results

def calculate_chunk_ids(chunks):
    last_doc_id = None
    current_chunk_index = 0

    for chunk in chunks:
        doc_type = chunk.metadata.get("type", "Unknown")
        level = chunk.metadata.get("level", "Unknown")
        orig_id = chunk.metadata.get("orig_id", "0")
        current_doc_id = f"{doc_type}:{level}:{orig_id}"
        
        # Nếu đây là cùng một tài liệu với đoạn trước, tăng chỉ số đoạn.
        if current_doc_id == last_doc_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        new_chunk_id = f"{current_doc_id}:{current_chunk_index}"
        last_doc_id = current_doc_id
        chunk.metadata["id"] = new_chunk_id

    return chunks

def clear_database():
    global FAISS_DB
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)
        FAISS_DB = None

def load_csv_documents(filepath: str) -> list[Document]:
    df = pd.read_csv(filepath)
    documents = []
    for idx, row in df.iterrows():
        content = f"Problem: {row['problem']}\nSolution: {row['solution']}"
        metadata = {
            "type": row["type"],
            "level": row["level"],
            "orig_id": str(idx)
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Clear the database before populating.")
    args = parser.parse_args()
    
    if args.reset:
        print("Clearing database…")
        clear_database()

    documents = load_csv_documents(DATA_PATH)
    chunks = split_documents(documents)
    add_to_faiss(chunks)

if __name__ == "__main__":
    main()
