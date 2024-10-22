from flask import Flask, render_template, request, jsonify, abort
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker
import spacy
import os
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
load_dotenv()  # Load the .env file

app = Flask(__name__)

# API Key from environment variable
API_KEY = os.getenv("API_KEY")  # Get the API key

# Load FAQ data from faqs.json
with open('faqs.json', 'r') as f:
    faq_data = json.load(f)

# Convert JSON data into a pandas DataFrame
faq_df = pd.DataFrame(faq_data)

# Load spaCy model for lemmatization
nlp = spacy.load('en_core_web_sm')

# Initialize spell checker
spell = SpellChecker()

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(faq_df['question'])

def preprocess_query(query):
    # Correct spelling
    corrected_query = []
    for word in query.split():
        corrected_word = spell.correction(word)
        if corrected_word is None:
            corrected_word = word  # If no correction found, keep the original word
        corrected_query.append(corrected_word)

    corrected_query = " ".join(corrected_query)

    # Lemmatize the query
    doc = nlp(corrected_query.lower())
    lemmatized_query = " ".join([token.lemma_ for token in doc])

    return lemmatized_query

@app.route('/search', methods=['POST'])
def search():
    print("GOT REQUEST!!!")
    # Check for API key
    api_key = request.headers.get('x-api-key')
    if api_key != API_KEY:
        abort(401, description="Unauthorized: Invalid API Key")

    query = request.form['query']

    # Preprocess the query
    processed_query = preprocess_query(query)

    # Transform the processed query into a vector
    query_vector = vectorizer.transform([processed_query])

    # Compute cosine similarity
    cosine_similarities = cosine_similarity(query_vector, faq_vectors).flatten()

    # Get results above a similarity threshold
    results = []
    for i, score in enumerate(cosine_similarities):
        if score > 0.1:  # Adjust threshold to filter out weak matches
            category = faq_df.iloc[i]['category']
            faq = {
                'question': faq_df.iloc[i]['question'],
                'answer': faq_df.iloc[i]['answer']
            }
            results.append({'category': category, 'faqs': [faq]})

    # Fallback to exact or partial string match if no results found
    if not results:
        for i, row in faq_df.iterrows():
            if query.lower() in row['question'].lower():
                category = row['category']
                faq = {
                    'question': row['question'],
                    'answer': row['answer']
                }
                results.append({'category': category, 'faqs': [faq]})

    return jsonify({'query': query, 'results': results})

if __name__ == '__main__':
    app.run(debug=True)
