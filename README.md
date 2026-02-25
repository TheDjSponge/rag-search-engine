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

# Project chapters

## Chapter 1.
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

## Chapter 2. 
In this chapter we implemented the TF-IDF counting method. 

TF in a document represents term frequency, the number

of occurences of a word in a document. If a word appears a lot we might assume it is of strong relevance in a document.
IDF represents inverse document frequency. It tracks the number of documents a word appears in and computes the inverse of it. The idea 
behind IDF is that a word that appears on every document is less likely to be a good discriminator of semantics. Whereas a rare word might.

TF-IDF is simply the product of the two terms. It creates a score where a word that appears a lot in a document but less in others is very relevant,
but a word that appears a lot in a document but is also in all the others is way less relevant.

We can run the cli commands of this chapter with 
```bash
uv run cli/keyword_search_cli.py build

uv run cli/keyword_search_cli.py tf doc_id term # Get TF score for a term in a doc

uv run cli/keyword_search_cli.py idf term # Get IDF score over dataset for a term

uv run cli/keyword_search_cli.py tfidf doc_id term # Get TF-IDF score for a term in a doc over all documents
```
