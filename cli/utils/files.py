import json
import os

from cli.utils.models import MovieEntry


def load_movies(file_path: str) -> dict[str, list[MovieEntry]]:
    if not os.path.exists(file_path):
        raise ValueError(
            "provided file path does not exists, couldn't load file"
        )

    with open(file_path) as file:
        movies = json.load(file)
    if not isinstance(movies, dict):
        raise ValueError("invalid movies.json file structure")

    return movies


def load_stopwords(file_path: str) -> list[str]:
    if not os.path.exists(file_path):
        raise ValueError(
            "provided file path for stopwords.txt does not exist. Couldn't load file"
        )

    with open(file_path) as file:
        stopwords = file.read()
        stopwords_list = stopwords.splitlines()

    return list(stopwords_list)


if __name__ == "__main__":
    # movies = load_movies("./data/movies.json")
    # print(movies["movies"][0]["title"])
    print(load_stopwords("./data/stopwords.txt"))
