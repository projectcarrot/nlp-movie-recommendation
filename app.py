import os
from flask import Flask, render_template, request, jsonify
from recommender import recommend_movies

app = Flask(__name__)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True) or {}

    user_text = data.get("user_text", "")
    selected_genres = data.get("selected_genres", [])
    mode = data.get("mode", "hybrid")
    sort_by = data.get("sort_by", "balanced")

    if not isinstance(user_text, str):
        user_text = str(user_text)

    if not isinstance(selected_genres, list):
        selected_genres = []
    else:
        selected_genres = [g for g in selected_genres if isinstance(g, str) and g.strip()]

    if mode not in {"hybrid", "tfidf"}:
        mode = "hybrid"
    if sort_by not in {"similarity", "popularity", "imdb_rating", "release_date", "balanced"}:
        sort_by = "balanced"

    results_df = recommend_movies(
        user_text=user_text,
        selected_genres=selected_genres,
        top_n=10,
        K=200,
        use_genre_filter=True,
        mode=mode,
        sort_by=sort_by
    )

    results = results_df.to_dict(orient="records")

    return jsonify(results)


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode)
