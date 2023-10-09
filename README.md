# Reveal-heritage-values-and-attributes-from-textual-data
# Code Overview
Importing Libraries:
We start by importing necessary libraries, including numpy and bert-embedding.

Initializing BERT Embeddings:
We initialize BERT embeddings using BertEmbedding() from the bert_embedding library.

Defining Categories:
We define categories and their representative words in a dictionary. Each category is associated with a list of words.

Words to Categories:
We create a dictionary that maps each word to its corresponding category.

Obtaining Word Embeddings:
We calculate BERT embeddings for the words in the categories and store them in embeddings_index.

Processing Queries:
The process function takes a query as input, tokenizes it, calculates its BERT embedding, and computes cosine similarities with category embeddings. It returns a dictionary of category scores fo
