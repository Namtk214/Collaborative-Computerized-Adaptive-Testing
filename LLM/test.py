import argparse
import os
import base64
import tempfile
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from google import genai
from google.genai import types
from get_embedding_function import get_embedding_function
from database import retrieve_docs
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21"
FAISS_PATH = "C:/Users/Ngo Thanh Nam/my-app/LLM/faiss_index"

PROMPT_TEMPLATE = """
You are a helpful AI assistant. 
Context:
{context}

Question:
{question}

Chain of Thought: Let's solve this step by step:
"""

# Initialize embedding and FAISS
embedding = get_embedding_function()
FAISS_DB = FAISS.load_local(FAISS_PATH, embedding, allow_dangerous_deserialization=True)

def base64_to_image(base64_string):
    """
    Convert base64 image string to a temporary file.
    
    Args:
        base64_string (str): Base64 encoded image string
    
    Returns:
        str: Path to the temporary image file, or None if conversion fails
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            header, base64_string = base64_string.split(',', 1)
        
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(image_data)
            temp_file_path = temp_file.name
        
        return temp_file_path
    
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None

def get_response(question: str, image_path: str = None):
    """
    Generate a response based on the input query using AI model and retrieved documents.
    """
    results = retrieve_docs(question, limit=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)

    client = genai.Client(api_key="AIzaSyD3fiwgTYtzO4ImG8DjGAgEiRpdqyyB89M")

    parts = [types.Part.from_text(text=prompt)]

    if image_path:
        try:
            file_ref = client.files.upload(file=image_path)
            parts.append(
                types.Part.from_uri(file_uri=file_ref.uri, mime_type=file_ref.mime_type,)
            )
        except Exception as e:
            print(f"Error uploading image: {e}")
            # If image upload fails, continue with text-only prompt
    
    contents = [
        types.Content(
            role="user",
            parts=parts,
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    response_text = ""
    
    try:
        for chunk in genai.Client(api_key="AIzaSyD3fiwgTYtzO4ImG8DjGAgEiRpdqyyB89M").models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                response_text += chunk.text

        sources = [doc.metadata.get("id", "Unknown") for doc, _ in results]
        
        return {
            "message": response_text,
            "sources": sources
        }
    
    except Exception as e:
        return {
            "message": f"Error generating response: {str(e)}",
            "sources": []
        }
    finally:
        # Clean up temporary image file if it exists
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)

@app.route('/ai', methods=['POST'])
def ai_endpoint():
    """
    API endpoint for handling AI queries with optional image upload
    """
    data = request.json
    prompt = data.get('prompt', '')
    image = data.get('image', None)

    try:
        # Convert base64 image to temporary file if present
        image_path = None
        if image:
            image_path = base64_to_image(image)

        response = get_response(prompt, image_path)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)