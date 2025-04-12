# import argparse
# import os
# import base64
# import tempfile
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# from langchain.vectorstores import FAISS
# from langchain.prompts import ChatPromptTemplate
# from langchain_openai import OpenAIEmbeddings
# from google import genai
# from google.genai import types
# from LLM.get_embedding_function import get_embedding_function
# from LLM.database import retrieve_docs
# from dotenv import load_dotenv

# load_dotenv()

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Configuration
# MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21"
# FAISS_PATH = "C:/Users/Ngo Thanh Nam/my-app/LLM/faiss_index"

# PROMPT_TEMPLATE = """
# You are a helpful AI assistant. 
# Context:
# {context}

# Question:
# {question}

# Chain of Thought: Let's solve this step by step:
# """

# # Initialize embedding and FAISS
# embedding = get_embedding_function()
# FAISS_DB = FAISS.load_local(FAISS_PATH, embedding, allow_dangerous_deserialization=True)

# def base64_to_image(base64_string):
#     """
#     Convert base64 image string to a temporary file.
    
#     Args:
#         base64_string (str): Base64 encoded image string
    
#     Returns:
#         str: Path to the temporary image file, or None if conversion fails
#     """
#     try:
#         # Remove data URL prefix if present
#         if ',' in base64_string:
#             header, base64_string = base64_string.split(',', 1)
        
#         # Decode base64 string
#         image_data = base64.b64decode(base64_string)
        
#         # Create a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
#             temp_file.write(image_data)
#             temp_file_path = temp_file.name
        
#         return temp_file_path
    
#     except Exception as e:
#         print(f"Error converting base64 to image: {e}")
#         return None

# def get_response(question: str, image_path: str = None):
#     """
#     Generate a response based on the input query using AI model and retrieved documents.
#     """
#     results = retrieve_docs(question, limit=5)
#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=question)

#     client = genai.Client(api_key="AIzaSyD3fiwgTYtzO4ImG8DjGAgEiRpdqyyB89M")

#     parts = [types.Part.from_text(text=prompt)]

#     if image_path:
#         try:
#             file_ref = client.files.upload(file=image_path)
#             parts.append(
#                 types.Part.from_uri(file_uri=file_ref.uri, mime_type=file_ref.mime_type,)
#             )
#         except Exception as e:
#             print(f"Error uploading image: {e}")
#             # If image upload fails, continue with text-only prompt
    
#     contents = [
#         types.Content(
#             role="user",
#             parts=parts,
#         ),
#     ]

#     generate_content_config = types.GenerateContentConfig(
#         temperature=1,
#         top_p=0.95,
#         top_k=40,
#         max_output_tokens=8192,
#         response_mime_type="text/plain",
#     )

#     response_text = ""
    
#     try:
#         for chunk in genai.Client(api_key="AIzaSyD3fiwgTYtzO4ImG8DjGAgEiRpdqyyB89M").models.generate_content_stream(
#             model=MODEL_NAME,
#             contents=contents,
#             config=generate_content_config,
#         ):
#             if chunk.text:
#                 response_text += chunk.text

#         sources = [doc.metadata.get("id", "Unknown") for doc, _ in results]
        
#         return {
#             "message": response_text,
#             "sources": sources
#         }
    
#     except Exception as e:
#         return {
#             "message": f"Error generating response: {str(e)}",
#             "sources": []
#         }
#     finally:
#         # Clean up temporary image file if it exists
#         if image_path and os.path.exists(image_path):
#             os.unlink(image_path)

# @app.route('/ai', methods=['POST'])
# def ai_endpoint():
#     """
#     API endpoint for handling AI queries with optional image upload
#     """
#     data = request.json
#     prompt = data.get('prompt', '')
#     image = data.get('image', None)

#     try:
#         # Convert base64 image to temporary file if present
#         image_path = None
#         if image:
#             image_path = base64_to_image(image)

#         response = get_response(prompt, image_path)
#         return jsonify(response)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500




# import io
# import os
# import json
# import numpy as np
# import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt

# from flask import Flask, request, session, render_template, redirect, url_for, send_file, jsonify
# from flask_cors import CORS
# from flask_session import Session

# # Custom imports (assuming these exist in your project structure)
# from CCCAT.dataset import AdapTestDataset
# from CCCAT.setting import params
# from CCCAT.selection_strategy import MCMC_Selection
# from CCCAT.config.session_config import configure_session
# from CCCAT.services.question_service import select_next_question, init_test_session
# from CCCAT.utils.theta_updater import update_theta_ccat

# # Matplotlib configuration
# matplotlib.use('Agg')

# app = Flask(__name__)
# CORS(app, supports_credentials=True)
# configure_session(app)

# def load_data():
#     """
#     Load dataset metadata and prepare data for adaptive testing
#     """
#     # Determine base directory for data
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     data_dir = os.path.join(base_dir, "data", params.data_name)

#     # Load metadata
#     metadata_path = os.path.join(data_dir, "metadata.json")
#     concept_map_path = os.path.join(data_dir, "concept_map.json")
#     train_path = os.path.join(data_dir, "train_triples.csv")
#     test_path = os.path.join(data_dir, "test_triples.csv")

#     # Read JSON files
#     with open(metadata_path, 'r') as f:
#         metadata = json.load(f)
#     with open(concept_map_path, 'r') as f:
#         concept_map = json.load(f)

#     # Read CSV files
#     train_triplets = pd.read_csv(train_path, encoding='utf-8').to_records(index=False)
#     test_triplets = pd.read_csv(test_path, encoding='utf-8').to_records(index=False)

#     # Create datasets
#     train_data = AdapTestDataset(
#         train_triplets, 
#         metadata['num_train_students'], 
#         metadata['num_questions']
#     )
#     test_data = AdapTestDataset(
#         test_triplets, 
#         metadata['num_test_students'], 
#         metadata['num_questions']
#     )

#     return train_data, test_data, concept_map, metadata

# def load_irt_params():
#     """
#     Load Item Response Theory parameters
#     """
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     data_dir = os.path.join(base_dir, "data", params.data_name)
    
#     gamma = np.load(os.path.join(data_dir, "alpha.npy"))
#     beta = np.load(os.path.join(data_dir, "beta.npy"))
    
#     return gamma, beta

# def load_preprocessed_question_metadata():
#     """
#     Load preprocessed question metadata
#     """
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     data_dir = os.path.join(base_dir, "data", params.data_name)
#     file_path = os.path.join(data_dir, "preprocessed_questions.csv")
    
#     df = pd.read_csv(file_path, encoding='utf-8')
    
#     qmeta = {}
#     answer_mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
    
#     for _, row in df.iterrows():
#         try:
#             new_qid = int(row['NewQuestionId'])
#         except KeyError:
#             new_qid = int(row['new_question_id'])
        
#         try:
#             old_qid = int(row['OldQuestionId'])
#         except KeyError:
#             old_qid = new_qid
        
#         raw_ans = str(row["CorrectAnswer"]).strip()
#         correct_answer = answer_mapping.get(raw_ans, raw_ans)
        
#         subjects_raw = str(row.get("SubjectsNames", "")).strip()
#         subjects = [s.strip() for s in subjects_raw.split(',')] if subjects_raw else []
        
#         qmeta[new_qid] = {
#             "old_question_id": old_qid,
#             "correct_answer": correct_answer,
#             "subjects": subjects
#         }
    
#     valid_q_ids = sorted(qmeta.keys())
#     return qmeta, valid_q_ids

# # Global data loading
# train_data, test_data, concept_map, metadata = load_data()
# gamma, beta = load_irt_params()
# question_meta, valid_q_ids = load_preprocessed_question_metadata()

# # Prepare anchor thetas
# np.random.seed(42)
# anchor_thetas = np.random.normal(0, 1, 100)

# # Flask App Configuration
# app.config['GAMMA'] = gamma
# app.config['BETA'] = beta
# app.config['QUESTION_META'] = question_meta
# app.config['VALID_Q_IDS'] = valid_q_ids
# app.config['ANCHOR_THETAS'] = anchor_thetas

# @app.route('/')
# def index():
#     return jsonify({"message": "Adaptive Testing API", "status": "active"})

# @app.route('/start', methods=['GET'])
# def start():
#     session.clear()
#     init_test_session(app)
#     return jsonify({"status": "Test started"})

# @app.route('/question', methods=['GET'])
# def question():
#     current_index = session.get('current_index', 0)
#     total_questions = 20

#     if current_index >= total_questions or len(session.get('unanswered', [])) == 0:
#         return jsonify({
#             "result": True,
#             "score": session.get("score", 0),
#             "total": total_questions,
#             "final_theta": session.get("current_theta", 0),
#             "current_rank": session.get("current_rank", 1),
#             "total_anchor": session.get("total_anchor", 1)
#         })

#     current_theta = session.get('current_theta', 0)
#     unanswered = session.get('unanswered')
#     question_meta = app.config.get('QUESTION_META', {})

#     next_qid = select_next_question(
#         current_theta, gamma, beta, unanswered, question_meta)
#     session['current_question'] = next_qid

#     q_info = question_meta.get(next_qid, {})
#     old_qid = q_info.get("old_question_id", next_qid)
    
#     image_url = f'/static/images/{params.data_name}/{old_qid}.jpg'

#     # Calculate anchor mean and delta
#     anchor_thetas = app.config.get("ANCHOR_THETAS", np.array([]))
#     anchor_mean = np.mean(anchor_thetas) if anchor_thetas.size else 0.0
#     delta = current_theta - anchor_mean

#     return jsonify({
#         "qid": next_qid,
#         "image_url": image_url,
#         "current_index": current_index,
#         "total": total_questions,
#         "current_theta": current_theta,
#         "current_delta": delta,
#         "current_rank": session.get("current_rank", 1),
#         "total_anchor": session.get("total_anchor", 1),
#         "subjects": q_info.get('subjects', [])
#     })

# @app.route('/submit', methods=['POST'])
# def submit():
#     user_answer = request.form.get("answer", "").strip().upper()
#     current_qid = session.get("current_question")
#     question_meta = app.config.get("QUESTION_META", {})
#     correct_answer = question_meta.get(current_qid, {}).get("correct_answer", "")

#     is_correct = 1 if user_answer == correct_answer else 0
#     session["score"] = session.get("score", 0) + is_correct

#     answered = session.setdefault("answered_questions", [])
#     answered.append(current_qid)
#     responses = session.setdefault("responses", [])
#     responses.append(is_correct)

#     a_list = session.setdefault("a_list", [])
#     b_list = session.setdefault("b_list", [])
#     a_list.append(gamma[current_qid])
#     b_list.append(beta[current_qid])

#     unanswered = session.setdefault("unanswered", [])
#     if current_qid in unanswered:
#         unanswered.remove(current_qid)

#     current_theta = session.get("current_theta", 0.0)
#     anchor_thetas = app.config.get("ANCHOR_THETAS", np.array([]))
    
#     new_theta = update_theta_ccat(
#         current_theta, responses, a_list, b_list, anchor_thetas,
#         lambda_reg=0.005, lambda_ranking=0.6, damping=0.7
#     )
#     session["current_theta"] = new_theta

#     # Calculate delta (deviation from anchor mean)
#     anchor_mean = np.mean(anchor_thetas) if anchor_thetas.size else 0.0
#     delta = new_theta - anchor_mean

#     # Track ranking and delta history
#     if anchor_thetas.size:
#         rank = sum(1 for x in anchor_thetas if x > new_theta) + 1
#         session["current_rank"] = rank
#         session["total_anchor"] = len(anchor_thetas)
#     else:
#         session["current_rank"] = 1
#         session["total_anchor"] = 1

#     delta_history = session.setdefault("delta_history", [])
#     delta_history.append(delta)

#     session["current_index"] = session.get("current_index", 0) + 1
    
#     return jsonify({
#         "status": "Answer submitted",
#         "is_correct": is_correct,
#         "current_theta": new_theta,
#         "current_delta": delta
#     })

# @app.route('/delta_plot.png')
# def delta_plot():
#     delta_history = session.get("delta_history", [])

#     if not delta_history:
#         delta_history = [0]

#     fig, ax = plt.subplots(figsize=(3, 3))
#     x_vals = list(range(1, len(delta_history) + 1))
#     ax.plot(x_vals, delta_history, marker='o', linewidth=2, color="#007BFF")

#     ax.set_title("Performance Deviation", fontsize=8)
#     ax.set_xlabel("Question", fontsize=7)
#     ax.set_ylabel("Δθ", fontsize=7)
#     ax.tick_params(axis='both', labelsize=6)

#     ax.set_ylim(-3, 3)
#     ax.set_xlim(1, max(5, len(delta_history)))

#     plt.tight_layout(pad=1.0)

#     buf = io.BytesIO()
#     fig.savefig(buf, format='png', dpi=100)
#     buf.seek(0)
#     plt.close(fig)

#     response = send_file(buf, mimetype='image/png')
#     response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
#     response.headers["Pragma"] = "no-cache"
#     response.headers["Expires"] = "0"
#     return send_file(buf, mimetype='image/png')

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)