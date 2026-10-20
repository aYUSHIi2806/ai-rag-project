
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = FastAPI()


class SearchRequest(BaseModel):
    document: str
    question: str


def find_matching_sentence(document, question):
    # Split document into sentences
    sentences = [
        s.strip()
        for s in re.split(r"[.!?]+", document)
        if len(s.strip()) > 10
    ]

    if not sentences or not question.strip():
        return {"answer": "Please enter a question.", "score": 0}

    # Keyword matching
    question_words = set(re.findall(r"\w+", question.lower()))
    keyword_scores = []

    for sentence in sentences:
        sentence_words = set(re.findall(r"\w+", sentence.lower()))
        matches = question_words & sentence_words
        keyword_scores.append(
            len(matches) / max(len(question_words), 1)
        )

    # TF-IDF similarity
    vectorizer = TfidfVectorizer()

    try:
        vectors = vectorizer.fit_transform(sentences + [question])
        tfidf_scores = cosine_similarity(
            vectors[-1], vectors[:-1]
        ).flatten()
    except ValueError:
        tfidf_scores = [0.0] * len(sentences)

    # Combine both scores
    final_scores = [
        0.5 * keyword_scores[i] + 0.5 * float(tfidf_scores[i])
        for i in range(len(sentences))
    ]

    best_idx = max(range(len(final_scores)), key=final_scores.__getitem__)
    best_score = final_scores[best_idx]

    # Basic low-match threshold; validate with test examples
    if best_score < 0.10:
        return {
            "answer": "No relevant match found in the document.",
            "score": round(best_score, 3),
        }

    return {
        "answer": sentences[best_idx],
        "score": round(best_score, 3),
    }


@app.post("/search")
async def search_document(request: SearchRequest):
    # Async FastAPI endpoint
    return find_matching_sentence(
        request.document, request.question
    )


@app.get("/")
async def home():
    return {"message": "Document QA API is running"}
