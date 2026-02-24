from flask import Flask, render_template, request, jsonify
from recommender import recommend_movies

app = Flask(__name__)

# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Recommendation endpoint
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    user_text = data.get("user_text", "")
    selected_genres = data.get("selected_genres", [])

    results_df = recommend_movies(
        user_text=user_text,
        selected_genres=selected_genres,
        top_n=10,
        K=200,
        use_genre_filter=False
    )

    # Convert DataFrame to JSON
    results = results_df.to_dict(orient="records")

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)