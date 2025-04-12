import io
import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from flask import Flask, request, session, render_template, redirect, url_for, send_file, jsonify
from flask_cors import CORS
from flask_session import Session

# Custom imports (assuming these exist in your project structure)
from dataset import AdapTestDataset
from setting import params
from selection_strategy import MCMC_Selection
from config.session_config import configure_session
from services.question_service import select_next_question, init_test_session
from utils.theta_updater import update_theta_ccat

# Matplotlib configuration
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app, supports_credentials=True)
configure_session(app)

def load_data():
    """
    Load dataset metadata and prepare data for adaptive testing
    """
    # Determine base directory for data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", params.data_name)

    # Load metadata
    metadata_path = os.path.join(data_dir, "metadata.json")
    concept_map_path = os.path.join(data_dir, "concept_map.json")
    train_path = os.path.join(data_dir, "train_triples.csv")
    test_path = os.path.join(data_dir, "test_triples.csv")

    # Read JSON files
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    with open(concept_map_path, 'r') as f:
        concept_map = json.load(f)

    # Read CSV files
    train_triplets = pd.read_csv(train_path, encoding='utf-8').to_records(index=False)
    test_triplets = pd.read_csv(test_path, encoding='utf-8').to_records(index=False)

    # Create datasets
    train_data = AdapTestDataset(
        train_triplets, 
        metadata['num_train_students'], 
        metadata['num_questions']
    )
    test_data = AdapTestDataset(
        test_triplets, 
        metadata['num_test_students'], 
        metadata['num_questions']
    )

    return train_data, test_data, concept_map, metadata

def load_irt_params():
    """
    Load Item Response Theory parameters
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", params.data_name)
    
    gamma = np.load(os.path.join(data_dir, "alpha.npy"))
    beta = np.load(os.path.join(data_dir, "beta.npy"))
    
    return gamma, beta

def load_preprocessed_question_metadata():
    """
    Load preprocessed question metadata
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", params.data_name)
    file_path = os.path.join(data_dir, "preprocessed_questions.csv")
    
    df = pd.read_csv(file_path, encoding='utf-8')
    
    qmeta = {}
    answer_mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
    
    for _, row in df.iterrows():
        try:
            new_qid = int(row['NewQuestionId'])
        except KeyError:
            new_qid = int(row['new_question_id'])
        
        try:
            old_qid = int(row['OldQuestionId'])
        except KeyError:
            old_qid = new_qid
        
        raw_ans = str(row["CorrectAnswer"]).strip()
        correct_answer = answer_mapping.get(raw_ans, raw_ans)
        
        subjects_raw = str(row.get("SubjectsNames", "")).strip()
        subjects = [s.strip() for s in subjects_raw.split(',')] if subjects_raw else []
        
        qmeta[new_qid] = {
            "old_question_id": old_qid,
            "correct_answer": correct_answer,
            "subjects": subjects
        }
    
    valid_q_ids = sorted(qmeta.keys())
    return qmeta, valid_q_ids

# Global data loading
train_data, test_data, concept_map, metadata = load_data()
gamma, beta = load_irt_params()
question_meta, valid_q_ids = load_preprocessed_question_metadata()

# Prepare anchor thetas
np.random.seed(42)
anchor_thetas = np.random.normal(0, 1, 100)

# Flask App Configuration
app.config['GAMMA'] = gamma
app.config['BETA'] = beta
app.config['QUESTION_META'] = question_meta
app.config['VALID_Q_IDS'] = valid_q_ids
app.config['ANCHOR_THETAS'] = anchor_thetas

@app.route('/')
def index():
    return jsonify({"message": "Adaptive Testing API", "status": "active"})

@app.route('/start', methods=['GET'])
def start():
    session.clear()
    init_test_session(app)
    return jsonify({"status": "Test started"})

@app.route('/question', methods=['GET'])
def question():
    current_index = session.get('current_index', 0)
    total_questions = 12

    if current_index >= total_questions or len(session.get('unanswered', [])) == 0:
        return jsonify({
            "result": True,
            "score": session.get("score", 0),
            "total": total_questions,
            "final_theta": session.get("current_theta", 0),
            "current_rank": session.get("current_rank", 1),
            "total_anchor": session.get("total_anchor", 1)
        })

    current_theta = session.get('current_theta', 0)
    unanswered = session.get('unanswered')
    question_meta = app.config.get('QUESTION_META', {})

    next_qid = select_next_question(
        current_theta, gamma, beta, unanswered, question_meta)
    session['current_question'] = next_qid

    q_info = question_meta.get(next_qid, {})
    old_qid = q_info.get("old_question_id", next_qid)
    
    image_url = f'/static/images/{params.data_name}/{old_qid}.jpg'

    # Calculate anchor mean and delta
    anchor_thetas = app.config.get("ANCHOR_THETAS", np.array([]))
    anchor_mean = np.mean(anchor_thetas) if anchor_thetas.size else 0.0
    delta = current_theta - anchor_mean

    return jsonify({
        "qid": next_qid,
        "image_url": image_url,
        "current_index": current_index,
        "total": total_questions,
        "current_theta": current_theta,
        "current_delta": delta,
        "current_rank": session.get("current_rank", 1),
        "total_anchor": session.get("total_anchor", 1),
        "subjects": q_info.get('subjects', [])
    })

@app.route('/submit', methods=['POST'])
def submit():
    user_answer = request.form.get("answer", "").strip().upper()
    current_qid = session.get("current_question")
    question_meta = app.config.get("QUESTION_META", {})
    correct_answer = question_meta.get(current_qid, {}).get("correct_answer", "")

    is_correct = 1 if user_answer == correct_answer else 0
    session["score"] = session.get("score", 0) + is_correct

    answered = session.setdefault("answered_questions", [])
    answered.append(current_qid)
    responses = session.setdefault("responses", [])
    responses.append(is_correct)

    a_list = session.setdefault("a_list", [])
    b_list = session.setdefault("b_list", [])
    a_list.append(gamma[current_qid])
    b_list.append(beta[current_qid])

    unanswered = session.setdefault("unanswered", [])
    if current_qid in unanswered:
        unanswered.remove(current_qid)

    current_theta = session.get("current_theta", 0.0)
    anchor_thetas = app.config.get("ANCHOR_THETAS", np.array([]))
    
    new_theta = update_theta_ccat(
        current_theta, responses, a_list, b_list, anchor_thetas,
        lambda_reg=0.005, lambda_ranking=0.6, damping=0.7
    )
    session["current_theta"] = new_theta

    # Calculate delta (deviation from anchor mean)
    anchor_mean = np.mean(anchor_thetas) if anchor_thetas.size else 0.0
    delta = new_theta - anchor_mean

    # Track ranking and delta history
    if anchor_thetas.size:
        rank = sum(1 for x in anchor_thetas if x > new_theta) + 1
        session["current_rank"] = rank
        session["total_anchor"] = len(anchor_thetas)
    else:
        session["current_rank"] = 1
        session["total_anchor"] = 1

    delta_history = session.setdefault("delta_history", [])
    delta_history.append(delta)

    session["current_index"] = session.get("current_index", 0) + 1
    
    return jsonify({
        "status": "Answer submitted",
        "is_correct": is_correct,
        "current_theta": new_theta,
        "current_delta": delta
    })

@app.route('/delta_plot.png')
def delta_plot():
    delta_history = session.get("delta_history", [])

    if not delta_history:
        delta_history = [0]

    fig, ax = plt.subplots(figsize=(3, 3))
    x_vals = list(range(1, len(delta_history) + 1))
    ax.plot(x_vals, delta_history, marker='o', linewidth=2, color="#007BFF")

    ax.set_title("Performance Deviation", fontsize=8)
    ax.set_xlabel("Question", fontsize=7)
    ax.set_ylabel("Δθ", fontsize=7)
    ax.tick_params(axis='both', labelsize=6)

    ax.set_ylim(-3, 3)
    ax.set_xlim(1, max(5, len(delta_history)))

    plt.tight_layout(pad=1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)

    response = send_file(buf, mimetype='image/png')
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, port=5000)