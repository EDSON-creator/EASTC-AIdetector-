#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd 
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import re
import textstat
from collections import Counter
import language_tool_python
from sklearn.pipeline import Pipeline
import pickle 
import spacy
# Load the English NER model
nlp = spacy.load("en_core_web_sm")
from sklearn.metrics import accuracy_score
tool = language_tool_python.LanguageTool('en-US')
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[2]:


data22 = pd.read_csv("sample22.csv", encoding ="latin1")


# In[21]:


#Punctuation count
def punctuation_count(text):
    punctuation = re.findall(r'[^\w\s]',text)
    return len(punctuation)

#word count
def word_count_regex(text):
    words = re.findall(r'\b\w+\b', text)
    return len(words)

#noun count
def noun_count(text):
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    noun_count = sum(1 for _, tag in tagged_tokens if tag.startswith('NN'))
    return noun_count

#verb count
def verb_count(text):
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    verb_count = sum(1 for _, tag in tagged_tokens if tag.startswith('VB'))
    return verb_count

#adjective count
def adj_count(text):
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    adj_count = sum(1 for _, tag in tagged_tokens if tag.startswith('JJ'))
    return adj_count

#Average sentence length
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

#number of named entity
def count_ner_tokens(text):
    doc = nlp(text)
    ner_count = len(doc.ents)
    return ner_count


# function to calculate readability scores
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


#number of grammatical errors 
def count_grammatical_errors(text):
    matches = tool.check(text)
    return len(matches)


#number of unique words in a text 
def hapax_legomena_count(text):
    # Tokenize the text into words
    words = text.split()
    # Count the frequency of each word
    word_counts = Counter(words)
    # Hapax legomena are words that appear exactly once
    hapax_legomena = [word for word, count in word_counts.items() if count == 1]
    hapax_ratio = len(hapax_legomena)/len(words)
    return hapax_ratio


def extract_features(text):
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
        **calculate_readability_scores(text)  # Merge readability scores
        }
    return features


# In[7]:





# In[8]:


chunk_size = 1000  

# Initialize an empty list to hold the processed chunks
processed_chunks = []
for chunk in pd.read_csv("sample22.csv", encoding='latin1', chunksize=chunk_size):
  # Apply the functions to the 'text' column
   chunk['word_count_regex'] = chunk['text'].apply(word_count_regex)
   chunk['punctuation_count'] = chunk['text'].apply(punctuation_count)
   chunk['verb_count'] = chunk['text'].apply(verb_count)
   chunk['adj_count'] = chunk['text'].apply(adj_count)
   chunk['noun_count'] = chunk['text'].apply(noun_count)
   chunk[' average_sentence'] = chunk['text'].apply(average_sentence)
   chunk['count_ner_tokens'] = chunk['text'].apply(count_ner_tokens)
   chunk['count_grammatical_errors'] = chunk['text'].apply(count_grammatical_errors)
   chunk['hapax_legomena_count'] = chunk['text'].apply(hapax_legomena_count)
   #append the processed chunk to the list
   processed_chunks.append(chunk)

# Concatenate all processed chunks into a single DataFrame
processed_data = pd.concat(processed_chunks, ignore_index=True)
# Combine the readability scores with the original data


# In[10]:


readability_scores_df = processed_data['text'].apply(calculate_readability_scores)
combined_df = pd.concat([processed_data, readability_scores_df], axis=1)


# In[12]:


X = combined_df.iloc[:,list(range(2, combined_df.shape[1]))]
Y = combined_df.iloc[:,1]


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[15]:


svm = SVC(kernel='linear',probability = True) 
svm.fit(X_train, y_train)


# In[16]:


y_pred = svm.predict(X_test)


# In[17]:


accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[24]:


filename = "model.pkl"
pickle.dump(svm,open(filename,'wb'))


# In[23]:


fname2 = "features.pkl"
pickle.dump(extract_features,open(fname2,'wb'))


# In[ ]:




