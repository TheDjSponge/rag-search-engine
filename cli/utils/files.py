import json
import os
from typing import Dict, List


def load_movies(file_path: str) -> Dict:
    if not os.path.exists(file_path):
        raise ValueError("provided file path does not exists, couldn't load file")

    with open(file_path, "r") as file:
        movies = json.load(file)

    return movies


def load_stopwords(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise ValueError(
            "provided file path for stopwords.txt does not exist. Couldn't load file"
        )

    with open(file_path, "r") as file:
        stopwords = file.read()
        stopwords_list = stopwords.splitlines()

    return list(stopwords_list)


if __name__ == "__main__":
    # movies = load_movies("./data/movies.json")
    # print(movies["movies"][0]["title"])
    print(load_stopwords("./data/stopwords.txt"))
