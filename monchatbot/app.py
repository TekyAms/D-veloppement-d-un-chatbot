import random
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import json
import pickle
from nltk.stem import WordNetLemmatizer
from langdetect import detect
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

app = Flask(__name__)

# Charger le modèle et les données
model = load_model('model.h5')
lemmatizer = WordNetLemmatizer()
# Charger les données de prétraitement
preprocessing_data = "preprocessing_data.pkl"
chatbot_tourisme_fr_en = "chatbot_tourisme_fr_en.json"
q_table_file = "q_table.json"

def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())  
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Charger les données de prétraitement
with open('preprocessing_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Charger le fichier JSON des intentions
with open('chatbot_tourisme_fr_en.json', encoding='utf-8') as file:
    intentions = json.load(file)['intentions']


# Charger ou initialiser la Q-table
def load_q_table():
    global q_table
    if os.path.exists(q_table_file):
        with open(q_table_file, "r", encoding="utf-8") as file:
            q_table = json.load(file)
    else:
        print("[INFO] Fichier q_table.json non trouvé. Création d'une nouvelle table...")
        q_table = {}
        save_q_table()  # Sauvegarde immédiate pour créer le fichier

def save_q_table():
    try:
        with open(q_table_file, "w", encoding="utf-8") as file:
            json.dump(q_table, file, indent=4)
        print("[INFO] Fichier q_table.json mis à jour.")
    except Exception as e:
        print(f"[ERREUR] Impossible d'écrire dans q_table.json : {e}")

# Vérification au démarrage
load_q_table()

words = data['words']
classes = data['classes']

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Mise à jour de la Q-table
def update_q_table(intent, response, reward):
    if intent not in q_table:
        q_table[intent] = {}
    if response not in q_table[intent]:
        q_table[intent][response] = 0
    q_table[intent][response] += reward
    save_q_table()
# Fonction pour trouver la meilleure correspondance avec TF-IDF
def find_best_match_tfidf(user_question, questions_list):
    if not questions_list:  # Vérifier si la liste est vide
        return None

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions_list + [user_question])

    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    best_match_idx = similarity_scores.argsort()[0][-1]  # Meilleur score
    best_match_score = similarity_scores[0, best_match_idx]

    print(f"Meilleure correspondance : {questions_list[best_match_idx]} (Score : {best_match_score})")

    if best_match_score > 0.4:  # Ajuste ce seuil si nécessaire
        return questions_list[best_match_idx]
    return None

# Fonction de réponse
def get_response(ints, intents_json, message):
    try:
        tag = ints[0]['intent'] if ints else None
        if not tag:
            return "Je ne comprends pas. Pouvez-vous reformuler?" if detect(message) == 'fr' else "I don't understand. Can you rephrase that?"
        
        try:
            language = detect(message)
        except:
            language = "fr"  # Défaut en français si détection échoue

        lang_key = "fr" if language == "fr" else "en"

        for intent in intents_json:
            if intent['tag'] == tag:
                questions_key = f"questions_{lang_key}"
                responses_key = f"réponses_{lang_key}"

                if questions_key in intent and responses_key in intent:
                    best_match = find_best_match_tfidf(message, intent[questions_key])

                    if best_match:
                        index = intent[questions_key].index(best_match)
                        return intent[responses_key][index]
                    else:
                        print(f"Aucune correspondance exacte trouvée pour '{message}', réponse aléatoire choisie.")
                        return random.choice(intent[responses_key])

        return "Je ne comprends pas. Pouvez-vous reformuler?" if language == "fr" else "I don't understand. Can you rephrase that?"
    
    except Exception as e:
        print(f"Erreur dans get_response : {e}")
        return "Une erreur s'est produite. Veuillez réessayer plus tard."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        message = request.json['message']
        ints = predict_class(message)
        if not ints:
            return jsonify({'response': 'Je ne comprends pas. Pouvez-vous reformuler?'})
        response = get_response(ints, intentions, message)
        return jsonify({'response': response})
    except Exception as e:
        print(f"Erreur: {str(e)}")  # Pour le debugging
        return jsonify({'response': 'Une erreur est survenue'}), 500

# Route pour collecter le feedback
feedback_data = []
import json

feedback_file = "feedbacks.json"

# Route pour collecter le feedback
@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        user_message = data.get('message')
        bot_response = data.get('response')
        correct = data.get('correct')
        expected_response = None  # Initialisation de la réponse attendue comme None

        # Prédire l'intention du message utilisateur
        ints = predict_class(user_message)
        if not ints:
            return jsonify({"message": "Intention non trouvée."})

        intent = ints[0]['intent']
        reward = 1 if correct else -1

        # Récupérer la réponse attendue à partir du dataset (basé sur l'intention)
        for intention in intentions:
            if intention['tag'] == intent:
                # Sélectionner la réponse attendue selon la langue (français ou anglais)
                if detect(user_message) == 'fr':
                    expected_response = random.choice(intention['réponses_fr'])
                else:
                    expected_response = random.choice(intention['réponses_en'])
                break

        # Mettre à jour la Q-table si la réponse n'est pas correcte
        if not correct and expected_response:
            if intent not in q_table:
                q_table[intent] = {}
            q_table[intent][expected_response] = 1

        update_q_table(intent, bot_response, reward)

        # Sauvegarder le feedback dans un fichier JSON
        feedback_entry = {
            'message': user_message,
            'bot_response': bot_response,
            'correct': correct,
            'expected_response': expected_response  # Enregistrer la réponse attendue ici
        }

        # Charger les feedbacks existants
        try:
            with open(feedback_file, "r", encoding="utf-8") as f:
                feedback_data = json.load(f)
        except FileNotFoundError:
            feedback_data = []

        # Ajouter le nouveau feedback
        feedback_data.append(feedback_entry)

        # Sauvegarder les feedbacks dans le fichier
        with open(feedback_file, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=4)

        return jsonify({"message": "Feedback reçu ! Merci."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Interface pour voir les feedbacks
@app.route('/admin')
def admin():
    return render_template('admin.html', feedbacks=feedback_data)

if __name__ == '__main__':
    app.run(debug=True)
