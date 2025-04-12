from get_embedding_function import get_embedding_function
from langchain.vectorstores import FAISS

INDEX_PATH = "faiss_index"
query = """
How many vertical asymptotes does the graph of y=2/(x^2+x-6) have?
"""

# Load FAISS index đã build (persistent)
db = FAISS.load_local(INDEX_PATH, embeddings=get_embedding_function(), allow_dangerous_deserialization=True)

sample_docs = db.similarity_search(query, k=5)

for i, doc in enumerate(sample_docs):
    print(f"Document {i+1}:")
    print("ID:", doc.metadata.get("id"))
    print("Section:", doc.metadata.get("type"))
    print("Question and answer:", doc.page_content[:500])
    print("\n" + "="*50 + "\n")
