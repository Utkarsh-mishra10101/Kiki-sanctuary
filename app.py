import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# 1. IMPORT GOOGLE GENAI MODULES
from google import genai
from google.genai import types

# 2. FORCE LOAD .ENV FROM APP DIRECTORY
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GEMINI_API_KEY")

# 3. INITIALIZE FLASK APP
app = Flask(__name__)

# 4. INITIALIZE GEMINI CLIENT ONCE
if api_key:
    client = genai.Client(api_key=api_key)
    print("DEBUG: Found GEMINI_API_KEY: Yes")
    print("✅ Gemini API Client initialized.")
else:
    client = None
    print("DEBUG: Found GEMINI_API_KEY: No")
    print("⚠️ Warning: GEMINI_API_KEY not found in .env file.")
if api_key:
    client = genai.Client(api_key=api_key)
    print("✅ Gemini API Client initialized.")
else:
    client = None
    print("⚠️ Warning: GEMINI_API_KEY not found in .env file.")

# ---------------------------------------------------------
# 1. LOAD MODEL 1: Kaggle Multi-Disorder ML Model
# ---------------------------------------------------------
try:
    kaggle_model = joblib.load('disorder_model.pkl')
    kaggle_label_encoder = joblib.load('label_encoder.pkl')
    kaggle_model_loaded = True
    print("✅ Model 1 (ML Classifier) loaded successfully.")
except Exception as e:
    kaggle_model_loaded = False
    print("⚠️ Warning: Model 1 files not found. Run train_model.py first.")

# ---------------------------------------------------------
# 2. LOAD MODEL 2: Clinical Assessment Engine from data.csv
# ---------------------------------------------------------
CLINICAL_DATA = {}

def load_clinical_csv():
    global CLINICAL_DATA
    if not os.path.exists('data.csv'):
        print("⚠️ Warning: data.csv not found.")
        return

    df = pd.read_csv('data.csv')
    
    # Group by Condition and Questionnaire
    grouped = df.groupby(['Condition', 'Questionnaire'])
    
    for (condition, questionnaire), group in grouped:
        key = f"{condition} ({questionnaire})"
        
        # Extract unique questions in order
        unique_q_ids = group['Question_ID'].unique()
        questions_list = []
        
        for q_id in unique_q_ids:
            q_rows = group[group['Question_ID'] == q_id]
            q_text = q_rows['Question_Text'].iloc[0]
            
            options = []
            for _, r in q_rows.iterrows():
                options.append({
                    'label': str(r['Option_Label']),
                    'score': int(r['Option_Score_Value'])
                })
            
            questions_list.append({
                'id': q_id,
                'text': q_text,
                'options': options
            })
            
        CLINICAL_DATA[key] = {
            'condition': condition,
            'questionnaire': questionnaire,
            'questions': questions_list
        }
    print("✅ Model 2 (data.csv Psychometric Scales) loaded successfully.")

load_clinical_csv()

@app.route('/')
def index():
    return render_template('index.html', clinical_keys=list(CLINICAL_DATA.keys()))

# Endpoint: Get Clinical Scale Questions from data.csv
@app.route('/get_clinical_questions', methods=['POST'])
def get_clinical_questions():
    data = request.get_json() or {}
    key = data.get('key')
    assessment = CLINICAL_DATA.get(key, {})
    return jsonify({
        'questions': assessment.get('questions', []),
        'condition': assessment.get('condition', ''),
        'questionnaire': assessment.get('questionnaire', '')
    })

# Endpoint: Model 1 Prediction
@app.route('/predict_disorder', methods=['POST'])
def predict_disorder():
    if not kaggle_model_loaded:
        return jsonify({'error': 'ML model not loaded on server. Run train_model.py first.'}), 500

    data = request.get_json() or {}
    responses = data.get('responses', [])

    if len(responses) != 27:
        return jsonify({'error': f'Expected 27 features, got {len(responses)}.'}), 400

    input_array = np.array(responses).reshape(1, -1)
    probabilities = kaggle_model.predict_proba(input_array)[0]

    results = []
    for idx, prob in enumerate(probabilities):
        results.append({
            'disorder': str(kaggle_label_encoder.classes_[idx]),
            'probability': round(float(prob) * 100, 1)
        })

    results = sorted(results, key=lambda x: x['probability'], reverse=True)
    return jsonify({'predictions': results[:4]})

# Endpoint: Static Insights
# Endpoint: AI Insights generated via Gemini
@app.route('/disorder_insights', methods=['POST'])
def disorder_insights():
    data = request.get_json() or {}
    top_disorder = data.get('top_disorder', 'General Assessment')
    score = data.get('score', 0)
    max_score = data.get('max_score', 100)
    percentage = data.get('percentage', 0)

    if not client:
        return jsonify({'insight': "Gemini API key is not configured in .env."}), 500

    prompt = (
        f"The user completed a mental health screening for '{top_disorder}'. "
        f"Their score is {score}/{max_score} ({percentage}% match).\n\n"
        "Provide a compassionate, supportive, and concise diagnostic insight. "
        "Explain what this score framework generally reflects, summarize key clinical context, "
        "and offer constructive next steps. "
        "Important: Include a disclaimer that this is a screening tool, not a formal medical diagnosis."
    )

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.5,
            )
        )
        return jsonify({'insight': response.text})
    except Exception as e:
        print(f"Insight Exception: {e}")
        return jsonify({'insight': f"Error generating insight: {str(e)}"}), 500

# =========================================================
# ISOLATED CHATBOT ENDPOINT
# =========================================================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    user_message = data.get('message', '')
    history = data.get('history', [])

    if not user_message:
        return jsonify({'error': 'Message cannot be empty.'}), 400

    if not client:
        return jsonify({'reply': "Gemini API key is not configured in .env."}), 500

    system_instruction = (
        "You are Kiki, a compassionate mental health supportive chatbot. "
        "Provide helpful, empathetic, and safe responses. Never diagnose conditions or prescribe medications."
    )

    try:
        contents = []
        for msg in history:
            role = msg.get('role', 'user')
            text = msg.get('text', '')
            if text:
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=text)]
                ))

        contents.append(user_message)

        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
            )
        )
        return jsonify({'reply': response.text})
    except Exception as e:
        print(f"Chatbot Exception: {e}")
        return jsonify({'reply': f"Chatbot error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)