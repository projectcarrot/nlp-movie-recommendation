import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data_path = "TMDB_movie_dataset_v11.csv"

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

movie_data = load_dataset(data_path)
print(movie_data.shape)
print(movie_data.columns)
# 

cols = [
    "id", "title", "release_date", "original_language", "overview",
    "popularity", "tagline", "genres", "keywords", "homepage"
    ]

use_cols = movie_data[cols].copy()
use_cols.isnull().sum()
#

str_cols = ["title", "overview", "tagline", "genres", "keywords"]
for c in str_cols:
    use_cols[c] = use_cols[c].fillna("")

use_cols["release_date"] = pd.to_datetime(use_cols["release_date"], errors="coerce")
use_cols["release_year"] = use_cols["release_date"].dt.year
use_cols["release_year"] = use_cols["release_year"].fillna(0).astype(int)

use_cols["homepage"] = use_cols["homepage"].fillna("")

use_cols.isnull().sum()
#

use_cols.dtypes
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

use_cols["title_clean"] = use_cols["title"].apply(clean_text)
use_cols["overview_clean"] = use_cols["overview"].apply(clean_text)
use_cols["tagline_clean"] = use_cols["tagline"].apply(clean_text)
# use_cols["genres_clean"] = use_cols["genres"].apply(clean_text)
# use_cols["keywords_clean"] = use_cols["keywords"].apply(clean_text)

use_cols.head(3)
#

def split_genres(genre_str):
    if not genre_str:
        return []
    return [g.strip().lower() for g in genre_str.split(",")]

def split_keywords(keyword_str):
    if not keyword_str:
        return []
    return [k.strip().lower() for k in keyword_str.split(",")]

use_cols["genre_list"] = use_cols["genres"].apply(split_genres)
use_cols["keyword_list"] = use_cols["keywords"].apply(split_keywords)

use_cols[["genres", "genre_list", "keywords", "keyword_list"]].head(3)
#

use_cols["genre_text"] = use_cols["genre_list"].apply(lambda xs: " ".join(xs))
use_cols["keyword_text"] = use_cols["keyword_list"].apply(lambda xs: " ".join(xs))
use_cols[["genre_text", "keyword_text"]].head(3)
#

# for vectorization
def build_combined_text(row):
    
    return (
        row["title_clean"] + " " +
        row["title_clean"] + " " +
        row["overview_clean"] + " " +
        row["tagline_clean"] + " " +
        clean_text(row["genre_text"]) + " " +
        clean_text(row["keyword_text"])
    ).strip()

use_cols["combined_text"] = use_cols.apply(build_combined_text, axis=1)

use_cols[["title", "combined_text"]].head(2)
#

# fitted vectorizer to transform user input
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2)
)
# for all movies
movie_vectors = tfidf.fit_transform(use_cols["combined_text"])

movie_vectors.shape
#

# user input to TF_IDF vector
def vectorize_user_text(user_text):
    user_text = clean_text(user_text)
    return tfidf.transform([user_text])

# top-k candidates retrieval
def topk_candidates(user_text, K=200):
    user_text = clean_text(user_text)

    if user_text == "":
        return [], np.array([])
    
    user_vector = tfidf.transform([user_text])
    similarities = cosine_similarity(user_vector, movie_vectors).ravel()

    # top-K indices
    K = min(K, len(similarities))
    top_idx = np.argpartition(-similarities, range(K))[:K]
    top_idx = top_idx[np.argsort(-similarities[top_idx])]

    return top_idx.tolist(), similarities[top_idx]

user_text = "mafia comedy"
top_idx, top_scores = topk_candidates(user_text, K=10)

use_cols.loc[top_idx, ["title", "release_year", "genres", "popularity", "original_language"]].assign(tfidf_score=top_scores)
#

from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from sentence_transformers import SentenceTransformer

sbert = SentenceTransformer("all-MiniLM-L6-v2")
#

def normalize01(x):
    x = np.array(x, dtype=float)
    if len(x) == 0:
        return x
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-9)

def genre_match_score(movie_genres_list, selected_genres):
    if not selected_genres:
        return 0.0
    mg = set([g.strip().lower() for g in movie_genres_list])
    sg = [g.strip().lower() for g in selected_genres if str(g).strip()]
    if not sg:
        return 0.0
    hits = sum(1 for g in sg if g in mg)
    return hits / len(sg)

def recommend_movies(user_text, selected_genres=None, top_n=10, K=200,
                    use_genre_filter=False,
                    w_sbert=0.65, w_tfidf=0.25, w_genre=0.10, w_pop=0.03):
    selected_genres = selected_genres or []
    user_text_clean = clean_text(user_text or "")

    # 1) Candidate retrieval using Step 5 function
    cand_idx, cand_tfidf_scores = topk_candidates(user_text_clean, K=K)

    # If user_text empty, build a query from genres for retrieval
    if (not user_text_clean) and selected_genres:
        genre_query = " ".join([g.strip().lower() for g in selected_genres])
        cand_idx, cand_tfidf_scores = topk_candidates(genre_query, K=K)

    # If still no signal -> fallback (popularity)
    if len(cand_idx) == 0:
        top = use_cols.sort_values("popularity", ascending=False).head(top_n)
        return top[["title", "release_year", "overview", "genres", "popularity", "original_language"]]

    # 2) Optional hard genre filter on candidates
    if use_genre_filter and selected_genres:
        mask = []
        sg = set([g.strip().lower() for g in selected_genres])
        for i in cand_idx:
            gl = use_cols.iloc[i]["genre_list"]
            mask.append(any(g in sg for g in gl))
        mask = np.array(mask, dtype=bool)

        cand_idx = list(np.array(cand_idx)[mask])
        cand_tfidf_scores = np.array(cand_tfidf_scores)[mask]

        # if filter becomes too strict, fallback to no filter
        if len(cand_idx) == 0:
            cand_idx, cand_tfidf_scores = topk_candidates(user_text_clean, K=K)

    # 3) SBERT reranking on candidates ONLY
    # If user_text is empty, rerank using genre query
    sbert_query = user_text_clean if user_text_clean else " ".join([g.strip().lower() for g in selected_genres])

    query_emb = sbert.encode([sbert_query], normalize_embeddings=True)

    cand_texts = use_cols.iloc[cand_idx]["combined_text"].tolist()
    cand_embs = sbert.encode(cand_texts, normalize_embeddings=True)

    sbert_scores = cos_sim(query_emb, cand_embs).ravel()  # ~0..1

    # 4) Genre score + popularity score (for candidates)
    genre_scores = np.array([
        genre_match_score(use_cols.iloc[i]["genre_list"], selected_genres)
        for i in cand_idx
    ], dtype=float)

    pop_scores = use_cols.iloc[cand_idx]["popularity"].to_numpy(dtype=float)
    pop_scores = normalize01(pop_scores)

    # 5) Normalize TF-IDF candidate scores (they're already 0..1-ish, but normalize for stable mixing)
    tfidf_scores = normalize01(cand_tfidf_scores)

    # 6) Final combined score
    final_scores = (
        w_sbert * sbert_scores +
        w_tfidf * tfidf_scores +
        w_genre * genre_scores +
        w_pop * pop_scores
    )

    # 7) Rank and return Top-N
    order = np.argsort(-final_scores)
    pick_idx = np.array(cand_idx)[order]

    out = use_cols.iloc[pick_idx][["title", "release_year", "overview", "genres", "popularity", "original_language", "homepage"]].copy()
    out["similarity_percent"] = (final_scores[order] * 100).round(2)

    out = out[out["overview"].str.strip() != ""]
    out = out.drop_duplicates(subset=["title", "release_year", "original_language"])
    
    out = out.head(top_n)
    return out.reset_index(drop=True)
#

# ---- quick test ----
user_text = "sad ending"
selected_genres = ["Romance"]
print(recommend_movies(user_text, selected_genres, top_n=10, K=200, use_genre_filter=False))
