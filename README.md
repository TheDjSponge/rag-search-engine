# Learn-retrieval-augmented-generation

This project contains code from the lesson learn-retrieval-augmented-generation 
from boot.dev. 

https://www.boot.dev/courses/learn-retrieval-augmented-generation

the state of the code at the end of each chapter will be available as a tagged commit.
This project has a purpose of learning and is not meant to be used as a production tool


# Run the project

since the movies.json dataset is provided as a downloaded file, we can first get it by running 

```bash
## If it's not an executable
# chmod +x get_data.sh 
./get_data.sh
```

Then we can simply run the project with.

```bash
## Simple word matching
uv run cli/keyword_search_cli.py search "furious"
```

# CI
CI is heavily inspired by the modern python code quality setup described by Simone carolini
in the following [forum post](https://simone-carolini.medium.com/modern-python-code-quality-setup-uv-ruff-and-mypy-8038c6549dcc).



# Project chapters

## Chapter 1: Preprocessing.
In this chapter we implemented a pretty naive word-matching between the user query and movie titles.
To avoid false positives regarding semantics, we filtered some meaningless stop-words and reduced
words to their root form (stem). The general idea of this word matching is to project the user query
into a space that collects more semantics. Critically, it's obvious that this method works well 
for words that are domain specific (searching vampire will very likely result in vampire movies)
but words that have different meanings depending on context will cause a problem.

Run this part of the project with 
```bash
uv run cli/keyword_search_cli.py search "furious"
```

## Chapter 2: TF-IDF
In this chapter we implemented the TF-IDF counting method. 

TF in a document represents term frequency, the number

of occurences of a word in a document. If a word appears a lot we might assume it is of strong relevance in a document.
IDF represents inverse document frequency. It tracks the number of documents a word appears in and computes the inverse of it. The idea 
behind IDF is that a word that appears on every document is less likely to be a good discriminator of semantics. Whereas a rare word might.

TF-IDF is simply the product of the two terms. It creates a score where a word that appears a lot in a document but less in others is very relevant,
but a word that appears a lot in a document but is also in all the others is way less relevant.

We can run the cli commands of this chapter with 
```bash
uv run cli/keyword_search_cli.py build ## Mandatory to build the index

uv run cli/keyword_search_cli.py tf doc_id term # Get TF score for a term in a doc

uv run cli/keyword_search_cli.py idf term # Get IDF score over dataset for a term

uv run cli/keyword_search_cli.py tfidf doc_id term # Get TF-IDF score for a term in a doc over all documents
```

## Chapter 3: Keyword search

In this chapter, we implement Okapi BM25, which is a TF-IDF calculation that provides more stability.
formulas are:
$$BM25_{idf} = \frac{num\_docs - doc\_freq + 0.5}{doc\_freq + 0.5}$$
$$BM25_{tf} = \frac{tf * (k1 + 1)}{tf + k1 * length\_norm}$$
with 
$$length\_norm = 1 - b + b * \frac{doc\_length}{avg\_doc\_length}$$

In our implementation of TF-IDF, the TF weight scales lineraly with the appearances of a term, regardless of document length. It means that an excessively
long document containing a unique word a lot of times would have a massive score.
BM25 introduces saturation. More occurences of the same word in a document results in a progressively lower increase on score, 
as well as document length normalization to mitigate that effect. Both term effects can be regulated through the k1 and b parameters
that generally hold values of $k_1 \in [1.2,2.0]$ and $b=0.75$ (source https://fr.wikipedia.org/wiki/Okapi_BM25)

Practically, a document with a million words that uses the word "car" a thousand times would give a better TF-IDF score
than a document of 100 words where the word car appears 50, which is a surprising conclusion since document one would have 0.1% of car occurences 
when document 2 would have 50%.

we can run the cli commands of this chapter with
```bash
uv run cli/keyword_search_cli.py build ## Mandatory to build the index

uv run cli/keyword_search_cli.py bm25idf grizzly

uv run cli/keyword_search_cli.py bm25tf 1 anbuselvan

uv run cli/keyword_search_cli.py bm25search "love story"
```

## Chapter 4: Semantic search

In this chapter, the goal is to produce text embeddings (represent documents in a fixed dimension vector space) and 
use the distance in that space to compute a similarity metric.

To compute similarity between two embeddings (vectors), we generally use cosine similarity:
$$s = \frac{A \cdot B}{\lVert A \rVert \lVert B \rVert}$$

The reason behind cosine similarity usage as a distance metric is also because it's the metric used to commonly train 
embedding models. As we can see on the [model's page](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#fine-tuning).
Note that it's also important to always compare embeddings from similar models since different ones wouldn't rely on the 
same embedding spaces.

A relevant model choice can be made by inspecting the following leaderboard on huggingface
https://huggingface.co/spaces/mteb/leaderboard

Advice from coursework:
```
General Purpose Models

    Use case: Broad semantic understanding across domains
    Examples: all-MiniLM-L6-v2, all-mpnet-base-v2
    Best for: Movie search, general document retrieval

Domain-Specific Models

    Use case: Specialized knowledge (medical, legal, scientific)
    Examples: allenai-specter, microsoft/BiomedNLP-PubMedBERT
    Best for: Technical documentation, research papers

Multilingual Models

    Use case: Data in multiple languages in the same search system
    Examples: paraphrase-multilingual-MiniLM-L12-v2
    Best for: International movie databases
```

The core idea that we implement in this chapter is that we can leverage such embedding models to generate
embeddings for each document (movie data) and store them, effectively producing a vector database.
Then, whenever we we want to perform a search on that database, we can simple compute cosine similarity
(or the model's similarity metrics, in general) to find the maximum scores.

Recommended production vector databases options mentionned in this course are the following:
- PGVector: Open-source vector similarity search for PostgreSQL
- sqlite-vec: Open-source vector similarity search for SQLite
- LanceDB: Local-first, simple setup, small–medium scale
- Weaviate: Full-featured, GraphQL API, complex schema

