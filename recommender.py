
# IMPORTS
import pandas as pd
import numpy as np
from pathlib import Path
import re
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.neighbors import NearestNeighbors


# PATHS
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# LOAD MODELS & DATA
df = pd.read_parquet(MODEL_DIR / "products.parquet")
X_final = joblib.load(MODEL_DIR / "embeddings.joblib")
nn_model = joblib.load(MODEL_DIR / "nn_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
ordinal_encoder = joblib.load(MODEL_DIR / "ordinal_encoder.joblib")

TEXT_COLUMNS = ["product_name_clean", "brands_clean", "main_category_clean",
                "labels_tags_clean", "ingredients_tags_clean", "nutriscore_clean", "origins_clean"]

for col in TEXT_COLUMNS:
    if col not in df.columns:
        df[col] = ""

# Nutriscore mapping
NUTRI_MAP = {"a":5, "b":4, "c":3, "d":2, "e":1}


# LABEL KEYWORDS
def labels_matches(text: str, label_keywords=LABEL_KEYWORDS):
    """
    Returns a list of detected labels from a text (e.g., labels_tags or ingredients_tags)
    """
    # Check if input is valid string
    if not isinstance(text, str) or text.strip() == "":
        return []

    # Convert text to lowercase for case-insensitive matching
    text = text.lower()

    # Remove short prefixes like 'en:' or 'fr:' (language codes)
    text = re.sub(r"\b[a-z]{2,3}:", "", text)

    # Replace semicolons with commas
    text = text.replace(";", ",")

    # Remove extra spaces around commas
    text = re.sub(r"\s*,\s*", ",", text)

    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)

    # Split text into tokens using commas
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    detected = set()

    # Loop through each label and its keywords
    for label, keywords in label_keywords.items():
        for kw in keywords:
            for token in tokens:
                # If a keyword is found in a token, add the label
                if kw in token:
                    detected.add(label)

    # Return the list of detected labels
    return list(detected)

# Column with detected labels
df["detected_labels"] = df["labels_tags_clean"].apply(labels_matches)

# RECOMMENDATION FUNCTION
def recommend_products(product_name, brand=None, nutriscore=None, label=None, origin=None,
                       substitute_other_brand=True, similarity_level=5,
                       label_weight=1.0, nutri_weight=1.0, env_weight=1.0,
                       top_n=5):
    
    product_name = product_name.lower().strip()
    brand = brand.lower().strip() if brand else None
    label = label.lower().strip() if label else None
    nutriscore = nutriscore.lower().strip() if nutriscore else None
    origin = origin.lower().strip() if origin else None

    # Filter candidates by label/brand/origin before NN
    candidates = df.copy()
    if label:
        candidates = candidates[
        candidates["detected_labels"].apply(lambda labels: label in labels)
    ]
    if not substitute_other_brand and brand:
        candidates = candidates[candidates["brands_clean"].apply(lambda x: brand in x.split(","))]
    if origin:
        candidates = candidates[candidates["origins_clean"].apply(lambda x: origin in x.split(","))]

    if candidates.empty:
        print(f"Aucun produit avec label '{label}', marque '{brand}' ou origine '{origin}'")
        return pd.DataFrame()

    # Find product reference
    mask = df["product_name_clean"].str.contains(product_name)
    if brand:
        mask &= df["brands_clean"].str.contains(brand)
    if mask.sum() == 0:
        print("Produit de référence introuvable.")
        return pd.DataFrame()

    idx_ref = df[mask].index[0]
    ref_vector = X_final[idx_ref].reshape(1, -1)

    # NN on filtered candidates
    candidate_indices = candidates.index.tolist()
    distances, indices = nn_model.kneighbors(X_final[candidate_indices], n_neighbors=len(candidate_indices))
    similarity_scores = 1 - distances[0]  # cosine similarity
    sim_df = pd.DataFrame({"idx": candidate_indices, "similarity": similarity_scores}).set_index("idx")
    candidates = candidates.join(sim_df, how="inner")

    # Apply similarity threshold
    similarity_thresholds = {1:0.10, 2:0.20, 3:0.30, 4:0.40, 5:0.50}
    threshold = similarity_thresholds.get(similarity_level, 0.3)
    candidates = candidates[candidates["similarity"] >= threshold]
    if candidates.empty:
        print("Aucun produit assez similaire.")
        return pd.DataFrame()

    # Final score calculation
    score = candidates["similarity"].copy()
    if label:
        score += candidates["detected_labels"].apply(lambda labels: int(label in labels)) * label_weight

    if nutriscore:
        orig_val = NUTRI_MAP.get(nutriscore,3)
        cand_val = candidates["nutriscore_clean"].map(NUTRI_MAP).fillna(3)
        score += ((cand_val > orig_val).astype(int)) * nutri_weight
    if origin:
        score += candidates["origins_clean"].apply(lambda x: int(origin in x.split(","))) * 1.0  # weight 1 pour l'origine
    score += candidates["environmental_score_score"].fillna(0)/candidates["environmental_score_score"].max()*env_weight
    candidates["final_score"] = score

    # Better nutriscore flag
    candidates["better_nutriscore"] = False
    if nutriscore:
        candidates["better_nutriscore"] = (
            candidates["nutriscore_clean"].map(NUTRI_MAP).fillna(3) >
            NUTRI_MAP.get(nutriscore,3)
        )

    # Select top N
    candidates = candidates.sort_values("final_score", ascending=False).head(top_n)

    candidates["category_display"] = (
    candidates["main_category_clean"]
    .replace("", np.nan)
    .fillna(candidates.get("main_category_fr"))
    .fillna("Non renseignée")
)
    return candidates

# CLI TESTING
if __name__ == "__main__":
    print("\nBienvenue dans le système de recommandation !\n")
    product = input("Nom du produit à remplacer : ")
    brand = input("Marque (optionnel) : ") or None
    nutriscore = input("Nutriscore du produit actuel (a-e, optionnel) : ") or None
    label = input("Label préféré (bio, vegan, etc., optionnel) : ") or None
    origin = input("Origine préférée (optionnel) : ") or None
    similarity_level = int(input("Critère de ressemblance (1=lointain, 5=identique) : "))
    label_weight = float(input("Poids du label : ") or 1)
    nutri_weight = float(input("Poids Nutriscore : ") or 1)
    env_weight = float(input("Poids environnement : ") or 1)

    recs = recommend_products(
        product_name=product,
        brand=brand,
        nutriscore=nutriscore,
        label=label,
        origin=origin,
        similarity_level=similarity_level,
        label_weight=label_weight,
        nutri_weight=nutri_weight,
        env_weight=env_weight,
        top_n=5
    )

    if recs.empty:
        print("\nAucune recommandation.")
    else:
        print(f"\nTop recommandations pour '{product}':\n")
        for _, r in recs.iterrows():
            print(f"- {r['product_name']} | Catégorie: {r['main_category_clean']} | "
                  f"Similarité: {round(r['similarity'],2)} | Nutriscore: {r['nutriscore_grade']} | "
                  f"Meilleur Nutriscore: {r['better_nutriscore']} | Labels: {r['labels_tags_clean']} | "
                  f"Origine: {r['origins_clean']}")