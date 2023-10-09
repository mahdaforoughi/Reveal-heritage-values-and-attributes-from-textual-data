# Import necessary libraries
from numpy import dot
from numpy.linalg import norm
import numpy as np
from bert_embedding import BertEmbedding

# Initialize BERT embedding
bert_embedding = BertEmbedding()

# Define the categories and their representative words
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

# Words -> category
categories = {word: key for key, words in categories.items() for word in words}
print(categories)
# Embeddings for available words
embeddings_index = {}
tmp = bert_embedding(list(categories.keys()))
for item in tmp:
    embeddings_index[item[0][0]] = item[1][0]

# Processing the query
def process(query):
    query2 = query.strip('][').split(" ")
    print(query2)
    query_embed = bert_embedding(query2)
    query_embed = query_embed[0][1][0]
    scores = {}
    for word, embed in embeddings_index.items():
        print(categories)
        category = categories[word]
        print(category)
        dist = dot(query_embed, embed) / (norm(query_embed) * norm(embed))
        dist /= len(category)
        scores[category] = scores.get(category, 0) + dist
    print(query2, scores)
    return scores

# Example usage
process("old")

# Dictionaries to return the value class of each word:
def ReturnClass2(dic1):
    min_n = None
    min_word = None
    for x in dic1:
        if min_n is None or min_n < dic1[x]:
            min_word = x
            min_n = dic1[x]
    if min_n > 0.72:
        return min_word
    else:
        return "undefined"

def ReturnClass3(word):
    try:
        dic = process(word)
        return ReturnClass2(dic)
    except:
        return "undefined"

print("Model is defined")

