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