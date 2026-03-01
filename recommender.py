import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data_path = "TMDB_all_movies.csv"
EMBEDDING_CACHE_MAX_SIZE = 50000

cols = [
    "id", "title", "release_date", "original_language", "overview",
    "genres", "popularity", "tagline", "cast", "imdb_rating",
    "poster_path", "vote_count", "homepage", "keywords"
]

def load_dataset(path: str, use_columns: list[str]) -> pd.DataFrame:
    # Read only needed columns to reduce memory and startup time.
    header = pd.read_csv(path, nrows=0)
    available = [c for c in use_columns if c in header.columns]
    df = pd.read_csv(path, usecols=available)
    for col in use_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df

use_cols = load_dataset(data_path, cols).copy()

str_cols = ["title", "overview", "tagline", "genres", "cast", "keywords", "poster_path", "homepage"]
for c in str_cols:
    use_cols[c] = use_cols[c].fillna("")

use_cols["release_date"] = pd.to_datetime(use_cols["release_date"], errors="coerce")
use_cols["release_year"] = use_cols["release_date"].dt.year
use_cols["release_year"] = use_cols["release_year"].fillna(0).astype(int)

use_cols["homepage"] = use_cols["homepage"].fillna("")
use_cols["popularity"] = pd.to_numeric(use_cols["popularity"], errors="coerce").fillna(0.0)
use_cols["vote_count"] = pd.to_numeric(use_cols["vote_count"], errors="coerce").fillna(0.0)
use_cols["imdb_rating"] = pd.to_numeric(use_cols["imdb_rating"], errors="coerce").fillna(0.0)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

use_cols["title_clean"] = use_cols["title"].apply(clean_text)
use_cols["overview_clean"] = use_cols["overview"].apply(clean_text)
use_cols["tagline_clean"] = use_cols["tagline"].apply(clean_text)

def split_genres(genre_str):
    if not genre_str:
        return []
    return [g.strip().lower() for g in genre_str.split(",")]

def split_keywords(keyword_str):
    if not keyword_str:
        return []
    return [k.strip().lower() for k in keyword_str.split(",")]

def split_cast(cast_str):
    if not cast_str:
        return []
    return [name.strip().lower() for name in cast_str.split(",")]

use_cols["genre_list"] = use_cols["genres"].apply(split_genres)
use_cols["keyword_list"] = use_cols["keywords"].apply(split_keywords)
use_cols["cast_list"] = use_cols["cast"].apply(split_cast)

use_cols["genre_text"] = use_cols["genre_list"].apply(lambda xs: " ".join(xs))
use_cols["keyword_text"] = use_cols["keyword_list"].apply(lambda xs: " ".join(xs))
use_cols["cast_text"] = use_cols["cast_list"].apply(lambda xs: " ".join(xs))

LOW_QUALITY_OVERVIEWS = {
    "plot tba",
    "tba",
    "no overview",
    "no overview found",
    "overview not available",
}

QUALITY_THRESHOLDS = {
    "base_vote_count": 50,
    "base_popularity": 3.0,
    "base_imdb": 6.0,
    "mainstream_vote_count": 300,
    "mainstream_popularity": 10.0,
    "mainstream_imdb": 6.8,
}

TEXT_WEIGHTS = {
    "title": 1,
    "overview": 2,
    "tagline": 1,
    "genres": 2,
    "cast": 1,
    "keywords": 1,
}

def is_good_overview(text):
    t = (text or "").strip().lower()
    if not t:
        return False
    if t in LOW_QUALITY_OVERVIEWS:
        return False
    return len(t) >= 20

# for vectorization
def repeat_text(text, n):
    t = (text or "").strip()
    if not t or n <= 0:
        return ""
    return " ".join([t] * n)

def build_combined_text(row):
    # Balance text influence so title does not dominate query matching.
    parts = [
        repeat_text(row["title_clean"], TEXT_WEIGHTS["title"]),
        repeat_text(row["overview_clean"], TEXT_WEIGHTS["overview"]),
        repeat_text(row["tagline_clean"], TEXT_WEIGHTS["tagline"]),
        repeat_text(clean_text(row["genre_text"]), TEXT_WEIGHTS["genres"]),
        repeat_text(clean_text(row["cast_text"]), TEXT_WEIGHTS["cast"]),
        repeat_text(clean_text(row["keyword_text"]), TEXT_WEIGHTS["keywords"]),
    ]
    return " ".join(p for p in parts if p).strip()

use_cols["combined_text"] = use_cols.apply(build_combined_text, axis=1)
use_cols["has_good_overview"] = use_cols["overview"].apply(is_good_overview)
use_cols["quality_ok"] = (
    use_cols["has_good_overview"] &
    (
        (use_cols["vote_count"] >= QUALITY_THRESHOLDS["base_vote_count"]) |
        (use_cols["popularity"] >= QUALITY_THRESHOLDS["base_popularity"]) |
        (
            (use_cols["imdb_rating"] >= QUALITY_THRESHOLDS["base_imdb"]) &
            (use_cols["vote_count"] >= 20)
        )
    )
)
use_cols["mainstream_ok"] = (
    (use_cols["vote_count"] >= QUALITY_THRESHOLDS["mainstream_vote_count"]) |
    (use_cols["popularity"] >= QUALITY_THRESHOLDS["mainstream_popularity"]) |
    (
        (use_cols["imdb_rating"] >= QUALITY_THRESHOLDS["mainstream_imdb"]) &
        (use_cols["vote_count"] >= 120)
    )
)

# fitted vectorizer to transform user input
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2)
)
# for all movies
movie_vectors = tfidf.fit_transform(use_cols["combined_text"])

# top-k candidates retrieval
def topk_candidates(user_text, K=200):
    user_text = clean_text(user_text)

    if user_text == "":
        return [], np.array([])

    if K <= 0:
        return [], np.array([])

    user_vector = tfidf.transform([user_text])
    similarities = cosine_similarity(user_vector, movie_vectors).ravel()

    # top-K indices
    K = min(K, len(similarities))
    if K == 0:
        return [], np.array([])
    top_idx = np.argpartition(-similarities, K - 1)[:K]
    top_idx = top_idx[np.argsort(-similarities[top_idx])]

    return top_idx.tolist(), similarities[top_idx]

from sklearn.metrics.pairwise import cosine_similarity as cos_sim
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

sbert = SentenceTransformer("all-MiniLM-L6-v2") if SentenceTransformer else None
embedding_cache = {}

def _get_cached_candidate_embeddings(cand_idx):
    if sbert is None:
        return np.empty((0, 0), dtype=float)

    missing = [i for i in cand_idx if i not in embedding_cache]
    if missing:
        missing_texts = use_cols.iloc[missing]["combined_text"].tolist()
        missing_embs = sbert.encode(missing_texts, normalize_embeddings=True)
        for i, emb in zip(missing, missing_embs):
            embedding_cache[i] = emb

        # Keep cache size bounded with a simple FIFO eviction.
        while len(embedding_cache) > EMBEDDING_CACHE_MAX_SIZE:
            embedding_cache.pop(next(iter(embedding_cache)))

    return np.array([embedding_cache[i] for i in cand_idx], dtype=float)

def normalize01(x):
    x = np.array(x, dtype=float)
    if len(x) == 0:
        return x
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-9)

def query_looks_english(text):
    letters = re.findall(r"[a-zA-Z]", text or "")
    if not letters:
        return False
    ascii_letters = sum(1 for ch in letters if ord(ch) < 128)
    return (ascii_letters / len(letters)) >= 0.8

def is_name_like_query(text):
    tokens = [t for t in (text or "").split() if t]
    if len(tokens) < 2 or len(tokens) > 4:
        return False
    # Person-name queries are usually short alphabetic tokens.
    return all(re.fullmatch(r"[a-z]+", t) for t in tokens)

def cast_query_match_score(cast_list, query_text):
    q = (query_text or "").strip().lower()
    if not q:
        return 0.0
    cast_names = [str(n).strip().lower() for n in (cast_list or []) if str(n).strip()]
    if not cast_names:
        return 0.0

    # Strong signal: full query appears in at least one cast name.
    if any(q in name for name in cast_names):
        return 1.0

    q_tokens = set(q.split())
    if not q_tokens:
        return 0.0

    # Fallback: token overlap with best matching cast name.
    best = 0.0
    for name in cast_names:
        n_tokens = set(name.split())
        if not n_tokens:
            continue
        overlap = len(q_tokens & n_tokens) / len(q_tokens)
        if overlap > best:
            best = overlap
    return best

def genre_match_score(movie_genres_list, selected_genres):
    if not selected_genres:
        return 0.0
    mg = set([g.strip().lower() for g in movie_genres_list])
    sg = [g.strip().lower() for g in selected_genres if str(g).strip()]
    if not sg:
        return 0.0
    hits = sum(1 for g in sg if g in mg)
    return hits / len(sg)

def sort_results(out, sort_by):
    if out.empty:
        return out

    if sort_by == "popularity":
        return out.sort_values(["popularity", "similarity_percent"], ascending=[False, False])

    if sort_by == "imdb_rating":
        return out.sort_values(["imdb_rating", "vote_count", "similarity_percent"], ascending=[False, False, False])

    if sort_by == "release_date":
        return out.sort_values(["release_year", "similarity_percent"], ascending=[False, False])

    if sort_by == "balanced":
        sim_n = normalize01(out["similarity_percent"].to_numpy(dtype=float))
        pop_n = normalize01(out["popularity"].to_numpy(dtype=float))
        imdb_n = normalize01(out["imdb_rating"].to_numpy(dtype=float))
        vote_n = normalize01(out["vote_count"].to_numpy(dtype=float))
        out = out.copy()
        out["balanced_score"] = (
            0.55 * sim_n +
            0.20 * imdb_n +
            0.15 * pop_n +
            0.10 * vote_n
        )
        out = out.sort_values(["balanced_score", "similarity_percent"], ascending=[False, False])
        return out.drop(columns=["balanced_score"])

    return out.sort_values(["similarity_percent", "popularity"], ascending=[False, False])

def recommend_movies(user_text, selected_genres=None, top_n=10, K=200,
                    use_genre_filter=False, mode="hybrid",
                    sort_by="similarity",
                    prefer_english=True,
                    w_sbert=0.42, w_tfidf=0.20, w_genre=0.10, w_pop=0.08, w_vote=0.08, w_imdb=0.06, w_lang=0.06, w_mainstream=0.10):
    selected_genres = selected_genres or []
    user_text_clean = clean_text(user_text or "")
    name_query = is_name_like_query(user_text_clean)

    # 1) Candidate retrieval using Step 5 function
    cand_idx, cand_tfidf_scores = topk_candidates(user_text_clean, K=K)

    # If user_text empty, build a query from genres for retrieval
    if (not user_text_clean) and selected_genres:
        genre_query = " ".join([g.strip().lower() for g in selected_genres])
        cand_idx, cand_tfidf_scores = topk_candidates(genre_query, K=K)

    # If still no signal -> fallback (popularity)
    if len(cand_idx) == 0:
        top = use_cols.sort_values("popularity", ascending=False).head(top_n)
        out = top[["title", "release_year", "overview", "genres", "popularity", "original_language", "poster_path", "imdb_rating", "vote_count", "homepage"]].copy()
        out["similarity_percent"] = 0.0
        out = sort_results(out, sort_by)
        out = out.head(top_n)
        return out.reset_index(drop=True)

    # 1.5) Quality filter: remove low-information movies first.
    quality_mask = np.array([bool(use_cols.iloc[i]["quality_ok"]) for i in cand_idx], dtype=bool)
    if quality_mask.any():
        cand_idx = list(np.array(cand_idx)[quality_mask])
        cand_tfidf_scores = np.array(cand_tfidf_scores)[quality_mask]
    else:
        # Softer fallback: require only decent overview text.
        overview_mask = np.array([bool(use_cols.iloc[i]["has_good_overview"]) for i in cand_idx], dtype=bool)
        if overview_mask.any():
            cand_idx = list(np.array(cand_idx)[overview_mask])
            cand_tfidf_scores = np.array(cand_tfidf_scores)[overview_mask]

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

    # 2.3) For general text queries, prefer mainstream candidates if enough exist.
    if (not name_query) and user_text_clean:
        mainstream_mask = np.array([bool(use_cols.iloc[i]["mainstream_ok"]) for i in cand_idx], dtype=bool)
        if mainstream_mask.sum() >= max(top_n * 3, 30):
            cand_idx = list(np.array(cand_idx)[mainstream_mask])
            cand_tfidf_scores = np.array(cand_tfidf_scores)[mainstream_mask]

    # 2.5) If the query looks like a person name, prefer real cast matches.
    cast_scores = np.array([
        cast_query_match_score(use_cols.iloc[i]["cast_list"], user_text_clean)
        for i in cand_idx
    ], dtype=float)
    if name_query:
        cast_mask = cast_scores >= 0.5
        if cast_mask.any():
            cand_idx = list(np.array(cand_idx)[cast_mask])
            cand_tfidf_scores = np.array(cand_tfidf_scores)[cast_mask]
            cast_scores = cast_scores[cast_mask]

    if len(cand_idx) == 0:
        top = use_cols.sort_values("popularity", ascending=False).head(top_n)
        out = top[["title", "release_year", "overview", "genres", "popularity", "original_language", "poster_path", "imdb_rating", "vote_count", "homepage"]].copy()
        out["similarity_percent"] = 0.0
        out = sort_results(out, sort_by)
        out = out.head(top_n)
        return out.reset_index(drop=True)

    # 3) SBERT reranking on candidates ONLY
    # If user_text is empty, rerank using genre query
    sbert_query = user_text_clean if user_text_clean else " ".join([g.strip().lower() for g in selected_genres])
    if sbert is None:
        sbert_scores = np.zeros(len(cand_idx), dtype=float)
    else:
        query_emb = sbert.encode([sbert_query], normalize_embeddings=True)
        cand_embs = _get_cached_candidate_embeddings(cand_idx)
        sbert_scores = cos_sim(query_emb, cand_embs).ravel()  # ~0..1

    # 4) Genre score + popularity score (for candidates)
    genre_scores = np.array([
        genre_match_score(use_cols.iloc[i]["genre_list"], selected_genres)
        for i in cand_idx
    ], dtype=float)

    pop_scores = use_cols.iloc[cand_idx]["popularity"].to_numpy(dtype=float)
    pop_scores = normalize01(pop_scores)
    vote_scores = normalize01(use_cols.iloc[cand_idx]["vote_count"].to_numpy(dtype=float))
    imdb_scores = normalize01(use_cols.iloc[cand_idx]["imdb_rating"].to_numpy(dtype=float))
    mainstream_scores = np.array([
        1.0 if bool(use_cols.iloc[i]["mainstream_ok"]) else 0.0
        for i in cand_idx
    ], dtype=float)
    lang_scores = np.zeros(len(cand_idx), dtype=float)
    if prefer_english and query_looks_english(user_text_clean):
        lang_scores = np.array([
            1.0 if str(use_cols.iloc[i]["original_language"]).lower() == "en" else 0.0
            for i in cand_idx
        ], dtype=float)

    # 5) Normalize TF-IDF candidate scores (they're already 0..1-ish, but normalize for stable mixing)
    tfidf_scores = normalize01(cand_tfidf_scores)

    # 6) Final combined score
    if mode == "tfidf":
        if name_query:
            final_scores = (
                0.55 * tfidf_scores +
                0.35 * cast_scores +
                0.05 * pop_scores +
                0.05 * vote_scores
            )
        else:
            final_scores = (
                0.70 * tfidf_scores +
                0.15 * genre_scores +
                0.10 * pop_scores +
                0.05 * vote_scores
            )
    else:
        if name_query:
            final_scores = (
                0.40 * sbert_scores +
                0.20 * tfidf_scores +
                0.30 * cast_scores +
                0.04 * pop_scores +
                0.03 * vote_scores +
                0.03 * imdb_scores
            )
        else:
            final_scores = (
                w_sbert * sbert_scores +
                w_tfidf * tfidf_scores +
                w_genre * genre_scores +
                w_pop * pop_scores +
                w_vote * vote_scores +
                w_imdb * imdb_scores +
                w_lang * lang_scores +
                w_mainstream * mainstream_scores
            )

    # 7) Rank and return Top-N
    order = np.argsort(-final_scores)
    pick_idx = np.array(cand_idx)[order]

    out = use_cols.iloc[pick_idx][["title", "release_year", "overview", "genres", "popularity", "original_language", "poster_path", "imdb_rating", "vote_count", "homepage"]].copy()
    out["similarity_percent"] = np.clip(final_scores[order] * 100, 0, 100).round(2)

    out = out[out["overview"].str.strip() != ""]
    out = out.drop_duplicates(subset=["title", "release_year", "original_language"])

    if (not name_query) and user_text_clean:
        known_mask = (
            (out["vote_count"] >= QUALITY_THRESHOLDS["base_vote_count"]) |
            (out["popularity"] >= QUALITY_THRESHOLDS["base_popularity"]) |
            ((out["imdb_rating"] >= QUALITY_THRESHOLDS["base_imdb"]) & (out["vote_count"] >= 20))
        )
        known_out = out[known_mask]
        if len(known_out) >= top_n:
            out = known_out
        elif len(known_out) > 0:
            remainder = out[~known_mask]
            out = pd.concat([known_out, remainder], axis=0)

    out = sort_results(out, sort_by)
    out = out.head(top_n)
    return out.reset_index(drop=True)
