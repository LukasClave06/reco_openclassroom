import azure.functions as func
import logging
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = func.FunctionApp()

# Chargement des données
with open("data/articles_embeddings.pickle", "rb") as f:
    articles_embeddings = pickle.load(f)  # → ndarray

df_articles = pd.read_csv("data/articles_metadata.csv")
clicks_df = pd.read_csv("data/clicks_sample.csv")

# Création du DataFrame article_id + embedding
embeddings_df = pd.DataFrame({
    "article_id": df_articles["article_id"].astype(int),
    "embedding": list(articles_embeddings)
})

@app.route(route="recommendation", auth_level=func.AuthLevel.FUNCTION)
def recommendation(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('⏳ Requête reçue pour la fonction content-based.')

    user_id = req.params.get('user_id')
    if not user_id:
        return func.HttpResponse("❌ user_id manquant", status_code=400)

    try:
        user_id = int(user_id)
    except ValueError:
        return func.HttpResponse("❌ user_id invalide", status_code=400)

    # Récupération des articles cliqués par cet utilisateur
    clicked_articles = clicks_df[clicks_df["user_id"] == user_id]["click_article_id"].unique()

    if len(clicked_articles) == 0:
        return func.HttpResponse("⚠️ Aucun clic enregistré pour cet utilisateur.", status_code=200)

    # Embeddings des articles cliqués
    embeddings_clicked = embeddings_df[embeddings_df["article_id"].isin(clicked_articles)]["embedding"].tolist()

    if len(embeddings_clicked) == 0:
        return func.HttpResponse("⚠️ Aucun embedding trouvé pour les articles cliqués.", status_code=200)

    # Calcul du profil utilisateur
    user_profile = np.mean(embeddings_clicked, axis=0).reshape(1, -1)

    # Embeddings de tous les articles
    all_embeddings = np.stack(embeddings_df["embedding"].values)

    # Similarité cosine
    similarities = cosine_similarity(user_profile, all_embeddings).flatten()

    # Score + article_id
    scores_df = pd.DataFrame({
        "article_id": embeddings_df["article_id"],
        "similarity": similarities
    })

    # Supprimer articles déjà vus
    scores_df = scores_df[~scores_df["article_id"].isin(clicked_articles)]

    # Top 5 recommandations
    top_reco = scores_df.sort_values(by="similarity", ascending=False).head(5)

    # Format JSON
    response_data = top_reco.to_dict(orient="records")

    return func.HttpResponse(json.dumps(response_data), mimetype="application/json")
