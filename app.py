from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import nltk
"""nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')"""
import re
from collections import Counter
from textstat import textstat
from language_tool_python import LanguageTool
from nltk.tokenize import word_tokenize
import spacy
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertForMaskedLM, BertTokenizer
import torch
from sklearn.preprocessing import StandardScaler



# Load spaCy language model
nlp = spacy.load("en_core_web_sm")

# Load the trained model
model = pickle.load(open('model4.pkl', 'rb'))

#loading scaler 
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Initialize the grammar checker tool
tool =LanguageTool('en-US')

# Feature extraction functions with normalization
def normalize_text(text):
    return ' '.join(text.split())


# Feature extraction functions
def punctuation_count(text):
    punctuation = re.findall(r'[^\w\s]',text)
    return len(punctuation)
    
def word_count_regex(text):
    words = re.findall(r'\b\w+\b', text)
    return len(words)
    
def noun_count(text):
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    noun_count = sum(1 for _, tag in tagged_tokens if tag.startswith('NN'))
    return noun_count
    
def verb_count(text):
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    verb_count = sum(1 for _, tag in tagged_tokens if tag.startswith('VB'))
    return verb_count
    
def adj_count(text):
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    adj_count = sum(1 for _, tag in tagged_tokens if tag.startswith('JJ'))
    return adj_count

def  average_sentence(text):
    # Split the text into sentences
    sentences = nltk.sent_tokenize(text)
    # Split each sentence into words and calculate the length of each sentence
    sentence_lengths = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
    # Calculate the average sentence length
    if sentence_lengths:
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
    else:
        avg_length = 0
    return avg_length

def count_ner_tokens(text):
    doc = nlp(text)
    ner_count = len(doc.ents)
    return ner_count

def count_grammatical_errors(text):
    matches = tool.check(text)
    return len(matches)

def hapax_legomena_count(text):
    # Tokenize the text into words
    words = text.split()
    # Count the frequency of each word
    word_counts = Counter(words)
    # Hapax legomena are words that appear exactly once
    hapax_legomena = [word for word, count in word_counts.items() if count == 1]
    hapax_ratio = len(hapax_legomena)/len(words)
    return hapax_ratio


def calculate_readability_scores(text):
    return pd.Series({
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'gunning_fog': textstat.gunning_fog(text),
        'coleman_liau_index': textstat.coleman_liau_index(text),
        'dale_chall_readability_score': textstat.dale_chall_readability_score(text),
        'ari': textstat.automated_readability_index(text),
        'linsear_write_formula': textstat.linsear_write_formula(text),
        'spache_readability': textstat.spache_readability(text)
    })
    
def split_text(text, tokenizer, max_length):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
    return chunks

def calculate_perplexity(model, tokenizer, tokens):
    tokens_tensor = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(tokens_tensor, labels=tokens_tensor)
        loss = outputs.loss.item()
    return np.exp(loss)

def calculate_perplexities(text, gpt2_model_name='gpt2', bert_model_name='bert-base-uncased', max_length=512, random_seed=42):
    # Set random seed for consistency
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Load GPT-2 model and tokenizer
    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
    gpt2_model.eval()

    # Load BERT model and tokenizer
    bert_model = BertForMaskedLM.from_pretrained(bert_model_name)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    bert_model.eval()

    # Split the input text into chunks for GPT-2
    gpt2_chunks = split_text(text, gpt2_tokenizer, max_length=1024)
    
    # Calculate perplexity for each chunk and average them for GPT-2
    gpt2_perplexities = [calculate_perplexity(gpt2_model, gpt2_tokenizer, chunk) for chunk in gpt2_chunks]
    avg_gpt2_perplexity = np.mean(gpt2_perplexities)

    # Split the input text into chunks for BERT
    bert_chunks = split_text(text, bert_tokenizer, max_length=512)
    
    # Calculate perplexity for each chunk and average them for BERT
    bert_perplexities = []
    mask_token = bert_tokenizer.mask_token_id

    for chunk in bert_chunks:
        if len(chunk) > 2:  # Ensure there are enough tokens to mask
            masked_chunk = chunk.copy()
            mask_position = np.random.randint(1, len(masked_chunk) - 1)
            masked_chunk[mask_position] = mask_token
            bert_perplexity = calculate_perplexity(bert_model, bert_tokenizer, masked_chunk)
            bert_perplexities.append(bert_perplexity)
    
    avg_bert_perplexity = np.mean(bert_perplexities) if bert_perplexities else float('inf')

    return avg_gpt2_perplexity, avg_bert_perplexity

def extract_features(text):
    
    text = normalize_text(text)
    
    gpt2_perplexity, bert_perplexity = calculate_perplexities(text)
    # Call each feature extraction function and collect the results
    features = {
        "word_count_regex": word_count_regex(text),
        "punctuation_count": punctuation_count(text),
        "verb_count": verb_count(text),
        "adj_count": adj_count(text),
        "noun_count": noun_count(text),
        " average_sentence": average_sentence(text),
        "count_ner_tokens": count_ner_tokens(text),
        "count_grammatical_errors": count_grammatical_errors(text),
        "hapax_legomena_count": hapax_legomena_count(text),
        **calculate_readability_scores(text), # Merge readability scores
        "gpt2_perplexity": gpt2_perplexity,
        "bert_perplexity": bert_perplexity
        }
    return features

# Create a Flask app
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text data from the request
    data = request.form['text']
    
    if not data:
        return jsonify({"error": "Input text is empty"}), 400

    # Extract features from the text
    features = extract_features(data)
    features_df = pd.DataFrame(features, index=[0])
    st_scaler = scaler.transform(features_df)
    # Predict the outcome
    pred = model.predict(st_scaler)[0]
    probabilities = model.predict_proba(st_scaler)[0]
    
    # Get the confidence percentage for the human or AI generated class
    probability = probabilities[pred]
    AI = probabilities[1]* 100
    Human = probabilities[0]*100
    percent_generated = probability * 100
    
    # Create a response dictionary
    response = {
        'text': data,
        'prediction': 'AI' if pred == 1 else 'Human',
        'confidence': f'{percent_generated:.1f}%',
        'AI': F'{AI:.1F}%',
        'HUMAN': F'{Human:.1F}%'
    }
    
    return jsonify(response)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)