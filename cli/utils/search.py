import string
from typing import Dict, List
from files import load_stopwords
from nltk.stem import PorterStemmer

def find_matching_movies(movies: Dict[str,List[Dict]],query: str) -> List[str]:
    matched = []
    movies_list = movies["movies"]
    stopwords_list = load_stopwords('./data/stopwords.txt')
    query = remove_stopwords(query, stopwords_list)
    tokenized_query = tokenize(format_string(query))
    for movie in movies_list:
        if any([token in format_string(remove_stopwords(movie["title"], stopwords_list)) for token in tokenized_query]):
            matched.append(movie["title"])
    
    return matched

def print_matched_movies(matched_list: List[str]) -> None:
    print("\n--- Words found through direct matching---\n")
    for movie_title in matched_list:
        print(f"- {movie_title}")



def format_string(to_format: str) -> str: 
    translation_table = str.maketrans("","",string.punctuation)
    lower_string = to_format.lower()
    translated_string = lower_string.translate(translation_table)  
    return translated_string


def tokenize(to_tokenize: str) -> List[str]:
    return to_tokenize.split()

def has_token_intersection(tokens_1: List[str], tokens_2: List[str]):
    return len(set(tokens_1).intersection(set(tokens_2))) > 0


def remove_stopwords(text :str, stopwords: List[str]):
    tokenized_text = tokenize(text)
    for stopword in stopwords:
        if stopword in tokenized_text:
            tokenized_text.remove(stopword)
    return " ".join(tokenized_text)


if __name__ == "__main__":
    #example = "je mange du pain, qu'il est bon!!"
    #translation_table =str.maketrans("","",string.punctuation)
    #translated = example.translate(translation_table)
    #print(translated)
    #set1 = ["yes", "no"]
    #set2 = ["no", "maybe"]
    #set3 = ["lol"]
    #print(f"Intersection between set 1 and 2? {has_token_intersection(set1,set2)}")
    #print(f"Intersection between set 1 and 1? {has_token_intersection(set1,set3)}")
    #stopwords = ["a", "the"]
    #query = "the dog ate a banana"
    #print(remove_stopwords(query, stopwords))
    token = "running"
    stemmer =  PorterStemmer()
    print(stemmer.stem(token))

