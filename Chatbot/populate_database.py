import os
import base64
import tempfile
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from get_embedding_function import get_embedding_function
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
# load_dotenv()

# # Configuration
# OPENAI_API_KEY = "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"
# embedding = OpenAIEmbeddings(
#     model="text-embedding-3-small",  # or "text-embedding-ada-002"
#     openai_api_key=OPENAI_API_KEY
# )

embedding = get_embedding_function()

# Paths and model configuration
INDEX_PATH = "C:/Users/Ngo Thanh Nam/my-app/Chatbot/faiss_index"
MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21"

# Prompt template
PROMPT_TEMPLATE = """
Context from related documents:
{context}

Question:
{question}

Please provide a comprehensive and helpful response.
"""

# Initialize Gemini client

try:
    genai.configure(api_key="AIzaSyD3fiwgTYtzO4ImG8DjGAgEiRpdqyyB89M")
    # Use the specific model for both text and vision
    text_model = genai.GenerativeModel(MODEL_NAME)
    vision_model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")
    raise

# Create Flask app with CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["POST"]}})

def validate_input(prompt, image):
    """
    Validate input data
    Returns:
    - Boolean: Is input valid
    - String: Error message (if any)
    """
    if not prompt and not image:
        return False, "No input provided. Please enter a prompt or upload an image."
    
    if image:
        # Basic image validation
        try:
            # Check if image is a valid base64 string
            if ',' not in image:
                return False, "Invalid image format"
            
            # Try to decode the image
            header, base64_data = image.split(',', 1)
            base64.b64decode(base64_data)
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False, "Invalid image data"
    
    return True, ""

def save_base64_image(base64_str):
    """Save base64 encoded image to a temporary file."""
    try:
        # Validate base64 string
        if not base64_str or ',' not in base64_str:
            logger.warning("Invalid base64 image string")
            return None

        # Split the base64 string
        header, base64_data = base64_str.split(',', 1)
        
        # Determine mime type
        mime_type = header.split(':')[1].split(';')[0]
        file_extension = mime_type.split('/')[-1]
        
        # Decode base64 image
        image_data = base64.b64decode(base64_data)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}', mode='wb') as temp_file:
            temp_file.write(image_data)
            return temp_file.name
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return None

def query_rag(query_text: str, image_path: str = None):
    try:
        # Load FAISS index
        db = FAISS.load_local(INDEX_PATH, embedding, allow_dangerous_deserialization=True)

        # Perform similarity search
        results = db.similarity_search_with_score(query_text, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

        # Prepare prompt
        full_prompt = f"Context from related documents:\n{context_text}\n\nQuestion:\n{query_text}\n\nPlease provide a comprehensive and helpful response."

        # Determine model and generation parameters
        generation_config = {
            'temperature': 0.7,
            'top_p': 0.9,
            'max_output_tokens': 4096,
        }

        # Generate response with or without image
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as img_file:
                image_parts = img_file.read()
                response = vision_model.generate_content(
                    [full_prompt, image_parts],
                    generation_config=generation_config
                )
                os.unlink(image_path)  # Clean up temporary file
        else:
            response = text_model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

        # Extract text response
        response_text = response.text if hasattr(response, 'text') else str(response)

        # Extract sources
        sources = [doc.metadata.get("source", "Unknown") for doc, _ in results]
        
        return response_text, sources

    except Exception as e:
        logger.error(f"Error in query_rag: {e}")
        return f"Processing error: {str(e)}", []

@app.route('/ai', methods=['POST'])
def ai_endpoint():
    try:
        # Parse incoming JSON data
        data = request.json
        logger.info(f"Received request: {data}")

        # Extract prompt and image
        query_text = data.get('prompt', '').strip()
        image_base64 = data.get('image')

        # Validate input
        is_valid, error_msg = validate_input(query_text, image_base64)
        if not is_valid:
            logger.warning(f"Invalid input: {error_msg}")
            return jsonify({"message": error_msg}), 400
        
        # Save image if provided
        image_path = None
        if image_base64:
            image_path = save_base64_image(image_base64)
            if not image_path:
                logger.warning("Failed to save image")
                return jsonify({"message": "Invalid image format"}), 400
        
        # Process query
        response, sources = query_rag(query_text, image_path)
        
        # Return response
        return jsonify({
            "message": response,
            "sources": sources
        })
    
    except Exception as e:
        logger.error(f"Unexpected error processing request: {e}")
        return jsonify({"message": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)