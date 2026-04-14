import json
import time

from sentence_transformers import CrossEncoder

from cli.llm_shenanigans.llm_utils import make_llm_query
from cli.utils.models import RRFMatchItem


def get_reranking_prompt_text(
    query: str, doc_title: str, doc_corpus: str
) -> str:
    return f"""Rate how well this movie matches the search query.
                Query: "{query}"
                Movie: {doc_title} - {doc_corpus}

                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Output ONLY the number in your response, no other text or explanation.

                Score:"""


def get_batch_ranking_prompt(query: str, doc_list_str: str) -> str:
    return f"""Rank the movies listed below by relevance to the following search query.

            Query: "{query}"

            Movies:
            {doc_list_str}

            Return ONLY the movie IDs in order of relevance (best match first, index starts at 0). Return a valid JSON list, nothing else.

            For example:
            [75, 12, 34, 2, 1]

            Ranking:"""


def get_evaluation_prompt(query: str, formatted_results: str) -> str:
    return f"""Rate how relevant each result is to this query on a 0-3 scale:

                Query: "{query}"

                Results:
                {formatted_results}

                Scale:
                - 3: Highly relevant
                - 2: Relevant
                - 1: Marginally relevant
                - 0: Not relevant

                Do NOT give any numbers other than 0, 1, 2, or 3.

                Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                [2, 0, 3, 2, 0, 1]"""


def rerank_movies(
    query: str, candidate_matches: list[RRFMatchItem], limit: int = 5
) -> list[RRFMatchItem]:
    print(
        f"Re-ranking top {len(candidate_matches)} results using individual method..."
    )
    for candidate in candidate_matches:
        prompt = get_reranking_prompt_text(
            query, candidate.get("title", ""), candidate.get("description", "")
        )
        answer = make_llm_query(prompt)
        candidate["rerank_score"] = float(answer)
        time.sleep(3)
    sorted_hybrid_matches = sorted(
        candidate_matches,
        key=lambda x: x.get("rerank_score", 0.0),
        reverse=True,
    )

    return sorted_hybrid_matches[:limit]


def rerank_batch(
    query: str, candidate_matches: list[RRFMatchItem], limit: int = 5
) -> list[RRFMatchItem]:
    print(
        f"Re-ranking top {len(candidate_matches)} results using batch method..."
    )
    doc_list = ""
    for match in candidate_matches:
        doc_list += json.dumps(match)

    prompt = get_batch_ranking_prompt(query=query, doc_list_str=doc_list)
    answer = make_llm_query(prompt)
    id_list = json.loads(answer)

    sorted_matches = []
    for id in id_list:
        sorted_matches.append(candidate_matches[id])
    return sorted_matches[:limit]


def rerank_with_cross_encoder(
    query: str,
    candidate_matches: list[RRFMatchItem],
    limit: int = 5,
    debug: str | None = None,
) -> list[RRFMatchItem]:
    pairs = []
    for match in candidate_matches:
        pairs.append(
            [query, f"{match.get('title', '')} - {match.get('document', '')}"]
        )
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)
    for id, match in enumerate(candidate_matches):
        match["cross_encoder_score"] = float(scores[id])
    sorted_matches = sorted(
        candidate_matches,
        key=lambda x: x.get("cross_encoder_score", 0.0),
        reverse=True,
    )
    if debug:
        found_idx = -1
        for idx, elem in enumerate(sorted_matches):
            if debug in elem["title"]:
                found_idx = idx
                break
        print(
            f"DEBUG[CROSS_ENCODER_RERANKING]: The searched movie {debug} is at position {found_idx}"
        )
        print(
            f"DEBUG[CROSS_ENCODER_RERANKING]: Cross encoder score {sorted_matches[found_idx].get('cross_encoder_score', 'None')}"
        )
    return sorted_matches[:limit]


def evaluate(
    query: str,
    candidate_matches: list[RRFMatchItem],
) -> list[int]:
    doc_list = ""
    for match in candidate_matches:
        doc_list += json.dumps(match)
    prompt = get_evaluation_prompt(query, doc_list)
    answer = make_llm_query(prompt)
    scores_list = json.loads(answer)
    if type(scores_list) is not list:
        raise ValueError(
            "Failed the json parsing of the LLM response in reranking:evaluate"
        )
    return scores_list
