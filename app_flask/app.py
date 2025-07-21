from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# === Chargement des embeddings ===
embeddings = pickle.load(open("data/articles_embeddings.pickle", "rb"))

# Charger les métadonnées pour récupérer les IDs des articles
df_metadata = pd.read_csv("data/articles_metadata.csv")
article_ids = df_metadata["article_id"].values

# Associer embeddings et article_id dans un DataFrame
df_embeddings = pd.DataFrame({
    "article_id": article_ids,
    "embedding": list(embeddings)
})

# === Simulation des interactions utilisateur (on suppose que vous avez ce fichier) ===
df_clicks = pd.read_csv("data/clicks_sample.csv")
df_user_clicks = df_clicks[["user_id", "click_article_id"]].drop_duplicates()

# === Fonction de recommandation Content-Based ===
def get_content_based_recommendations(user_id, df_user_clicks, df_embeddings, top_n=5):
    clicked_articles = df_user_clicks[df_user_clicks["user_id"] == user_id]["click_article_id"].unique()

    if len(clicked_articles) == 0:
        return []

    embeddings_clicked = df_embeddings[df_embeddings["article_id"].isin(clicked_articles)]["embedding"].tolist()
    user_profile = np.mean(embeddings_clicked, axis=0).reshape(1, -1)
    all_embeddings = np.stack(df_embeddings["embedding"].values)
    similarities = cosine_similarity(user_profile, all_embeddings).flatten()

    df_scores = pd.DataFrame({
        "article_id": df_embeddings["article_id"],
        "similarity": similarities
    })
    df_scores = df_scores[~df_scores["article_id"].isin(clicked_articles)]
    top_recommendations = df_scores.sort_values(by="similarity", ascending=False).head(top_n)
    return top_recommendations.to_dict(orient="records")

# === Route principale ===
@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user_id", type=int)
    if user_id is None:
        return jsonify({"error": "Paramètre user_id manquant"}), 400

    recos = get_content_based_recommendations(user_id, df_user_clicks, df_embeddings, top_n=5)
    return jsonify({"user_id": user_id, "recommendations": recos})

# === Lancement de l'API ===
if __name__ == "__main__":
    app.run(debug=True)