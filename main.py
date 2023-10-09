import numpy as np
from numpy import dot, linalg
from bert_embedding import BertEmbedding

# Initialize BERT embedding
bert_embedding = BertEmbedding()

# Define word categories
categories = {
    'social_spiritual': ['religious', 'spiritual'],
    'social_emotional': ['symbolic', 'emotion', 'moral'],
    'economic': ['economy', 'financial', 'commercial'],
    'political': ['political', 'government'],
    'age': ['old', 'ancient'],
    'scientific_workmanship': ['intelligent', 'knowledge', 'technical'],
    'scientific_technological': ['academic', 'technology', 'engineering'],
    'ecological_spiritual': ['ecological', 'environmental', 'natural'],
    'aesthetical': ['beautiful', 'beauty', 'art', 'artistic'],
    'historic': ['historic', 'history', 'historical']
}

# Create word-to-category mapping
word_to_category = {word: category for category, words in categories.items() for word in words}
print("Word to Category Mapping:", word_to_category)

# Get embeddings for words in categories
embeddings_index = {}
word_embeddings = bert_embedding(list(word_to_category.keys()))
for word, embedding in word_embeddings:
    embeddings_index[word] = embedding[0]

# Process a query and calculate scores for categories
def process_query(query):
    query_words = query.split()
    print("Query Words:", query_words)
    query_embedding = bert_embedding(query_words)
    query_embedding = query_embedding[0][1][0]
    scores = {}

    for word, embed in embeddings_index.items():
        category = word_to_category[word]
        dist = dot(query_embedding, embed) / (linalg.norm(query_embedding) * linalg.norm(embed))
        dist /= len(category)
        scores[category] = scores.get(category, 0) + dist

    print("Query:", query_words)
    print("Scores:", scores)
    return scores

# Example usage
process_query("old")

# Function to return the category with the highest score
def return_category(scores):
    max_category = max(scores, key=scores.get)
    if scores[max_category] > 0.72:
        return max_category
    else:
        return "undefined"

# Function to return the category for a word
def return_category_from_word(word):
    try:
        scores = process_query(word)
        return return_category(scores)
    except:
        return "undefined"

print("Model is defined")
