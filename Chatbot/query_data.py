# query_data.py
import argparse, os
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-2.0-flash"
INDEX_PATH = "faiss_index"

PROMPT_TEMPLATE = """..."""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)
    args = parser.parse_args()
    query_rag(args.query_text)

def query_rag(query_text: str):
    emb_fn = get_embedding_function()
    # 1️⃣ Kiểm tra index tồn tại chưa
    index_file = os.path.join(INDEX_PATH, "index.faiss")
    if not os.path.exists(index_file):
        print(f"❌ Không tìm thấy FAISS index tại '{index_file}'.")
        print("Hãy chạy script tạo index trước, ví dụ: python build_index.py")
        return

    # 2️⃣ Load index
    db = FAISS.load_local(
        INDEX_PATH,
        embedding_function=emb_fn,
        allow_dangerous_deserialization=True,
    )

    results = db.similarity_search_with_score(query_text, k=5)
    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context, question=query_text
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(prompt)])]
    cfg = types.GenerateContentConfig(
        temperature=1, top_p=0.95, top_k=40, max_output_tokens=8192
    )

    print("Response:", end=" ")
    for chunk in genai_client.models.generate_content_stream(
        model=MODEL_NAME, contents=contents, config=cfg
    ):
        if chunk.text:
            print(chunk.text, end="", flush=True)

    sources = [doc.metadata.get("id", "Unknown") for doc, _ in results]
    print(f"\nSources: {sources}")

if __name__ == "__main__":
    main()
