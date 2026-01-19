# recommender.py

# Libraries imports
import pandas as pd
import numpy as np
import re
import joblib
import gdown
import zipfile
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# SETUP PATHS AND DOWNLOAD/EXTRACT MODELS
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)  # Create if not exists

ZIP_ID = "1z5RsjM7pxHJ_0FjNiLcdF7WJYYYbH7H0"
ZIP_PATH = BASE_DIR / "models.zip"

# Download ZIP if not exists
if not ZIP_PATH.exists():
    print("Téléchargement du ZIP models via gdown…")
    gdown.download(
        id=ZIP_ID,
        output=str(ZIP_PATH),
        quiet=False
    )

# Dezip if model directory is empty
if not any(MODEL_DIR.iterdir()):
    print("Extraction des modèles...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        # Check each member
        for member in zip_ref.namelist():
            # Extract only files, ignore directories
            filename = Path(member).name
            if filename:  # Non-empty means it's a file
                target_path = MODEL_DIR / filename
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())
    print("✓ Modèles extraits avec succès")

# Load data and models
df = pd.read_parquet(MODEL_DIR / "products.parquet")
X_final = joblib.load(MODEL_DIR / "embeddings.joblib")
nn_model = joblib.load(MODEL_DIR / "nn_model.joblib")


# Text cleaning 
TEXT_COLUMNS = ["product_name_clean", "brands_clean", "main_category_clean",
                "labels_tags_clean", "ingredients_tags_clean", "nutriscore_clean", "origins_clean"]
for col in TEXT_COLUMNS:
    if col not in df.columns:
        df[col] = ""

# Attributes to numeric mappings and thresholds
NUTRI_MAP = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}
SIMILARITY_THRESHOLDS = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.7, 5: 0.8}

# Label keywords to facilitate detection. List based on common labels found in the dataset.
LABEL_KEYWORDS = {
    "bio": ["bio", "biologique", "ab", "naturel", "organic", "natural", "nature", "naturland"],
    
    "vegan": ["vegan", "végétal", "végane", "plant-based", "plant based", "v-", "plant-free", "plant free", "100-vegetable", "vegecert-certified-vegan", 
              "the-vegan-society", "certified-vegan", "vegan-friendly", "vegan-friendly-certified", "no-milk", "no-egg", "dairy-free", "egg-free"],
  
    "végétarien": ["végétarien", "vegetarian", "ovo-vegetarien", "lacto-vegetarien", "ovo-lacto-vegetarien", 
        "european-vegetarian-union", "vegecert-certified-vegan",
        "v-label", "label-vegetarien-ue", "ovo-vegetarian",
        "lacto-vegetarian", "100-vegetable"],

    "bon nutriscore": ["nutriscore a", "nutriscore b", "nutriscore c", "good nutriscore"],
    
    "sans additifs": ["sans-additif", "no-additives", ],
    
    "sans conservateurs": ["sans conservateur", "no preservative"],

    "allégé gras_sel_sucre": ["light", "allégé", "low-fat", "low-sugar", "low-salt", "réduit-en-sel", "réduit-en-sucre", "réduit-en-gras",
                               "low-or-no-fat", 
        "no-fat", "low-or-no-sugar", "no-added-sugar", 
        "en:no-preservatives", "no-colorings", "no-additives",
        "no-gmos", "no-gluten", "no-lactose", "source-of-fibre",
        "high-proteins", "omega-3", "without-sodium-nitrite"],    
    
    "durable environnement": ["durable", "éco", "responsable", "planet", "sustainable", "eco-friendly", "environment-friendly", "carbon-footprint", "carbon-compensated-product", 
        "sustainable-farming", "fair-trade", "max-havelaar",
        "rainforest-alliance", "utz-certified", "fsc", "pefc",
        "sustainable-seafood-msc", "responsible-aquaculture-asc",
        "haute-valeur-environnementale", "eco-score", "green-dot"],
    
    "label qualité": ["label", "certifié", "quality", "aop", "igp", "label-rouge", "red-label", "aoc", "label-rouge", "pdo", "pgi", "tsg",  
        "superior-quality", "fait-maison", "artisanal-production",
        "made-in-france", "origine-france", "french-meat",
        "french-milk", "pure-pork", "pure-beef"],
    
    "sans gluten": ["sans gluten", "gluten free", "gluten-free", "glutenfree", "glutenfrei", "glutenfrei-certified", "nogluten"],}


# Label detection
def labels_matches(text: str, label_keywords=LABEL_KEYWORDS):
    """Détecte labels catégorisés avec matching regex intelligent"""
    if not isinstance(text, str) or text.strip() == "":
        return []
    text = text.lower() # Normalize
    # Remove language prefixes
    text = re.sub(r"\b[a-z]{2,3}:", "", text)
    text = text.replace(";", ",")
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"\s+", " ", text)
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    detected = set()
    # Check for each label
    for label, keywords in label_keywords.items():
        for kw in keywords:
            for token in tokens:
                # Exact or partial match (regex for word boundaries)
                if kw == token or kw in token:
                    detected.add(label)
                elif re.search(r'\b' + re.escape(kw) + r'[\w-]*', token):
                    detected.add(label)
    return list(detected)

df["detected_labels"] = df["labels_tags_clean"].apply(labels_matches)

# Score computation
def compute_score(candidates, similarity_scores, label=None, label_weight=1.0,
                  ref_nutri=None, nutri_weight=1.0, origin=None, origin_weight=1.0,
                  env_weight=1.0):

    bonus = np.zeros(len(candidates))

    # Label bonus
    if label:
        bonus += candidates["detected_labels"].apply(lambda x: int(label in x)).values * label_weight

    # Nutriscore bonus
    if ref_nutri:
        cand_val = candidates["nutriscore_clean"].map(NUTRI_MAP).fillna(3).values
        bonus += (cand_val > ref_nutri).astype(int) * nutri_weight

    # Origin bonus
    if origin:
        bonus += candidates["origins_clean"].apply(lambda x: int(origin in x.split(","))).values * origin_weight

    # Environmental score bonus
    if "environmental_score_score" in candidates.columns:
        env = candidates["environmental_score_score"].fillna(0).values
        if env.max() > 0:
            bonus += (env / env.max()) * env_weight

    return similarity_scores * (1 + bonus)

# Vegan filter
def filter_vegan(candidates):
    animal_keywords = ["lait", "fromage", "oeuf", "beurre", "crème", "viande", "jambon", 
                       "bœuf", "porc", "poisson", "saumon"]
    # Function to check if ingredients contain animal products
    def is_vegan(ingredients):
        if not isinstance(ingredients, str):
            return True
        ingredients = ingredients.lower()
        return not any(kw in ingredients for kw in animal_keywords)
    return candidates[candidates["ingredients_tags_clean"].apply(is_vegan)]

# Main recommendation function
def recommend_products_web(df, nn_model, X_final, product_id, top_n=5,
                           label=None, origin=None, brand=None,
                           substitute_other_brand=True,
                           similarity_level=5,
                           category_weight=1.0, label_weight=1.0, nutri_weight=1.0, env_weight=1.0):

    if product_id not in df.index:
        return pd.DataFrame()
    # Reference product details
    ref_product = df.loc[product_id]
    ref_nutri = NUTRI_MAP.get(ref_product["nutriscore_clean"], 3)
    ref_categories = set(str(ref_product["main_category_clean"]).split(","))

    # Nearest neighbors
    n_neighbors = min(400, len(df)) # To have enough candidates
    distances, neighbor_indices = nn_model.kneighbors(
        X_final[product_id].reshape(1, -1), n_neighbors=n_neighbors
    ) 
    neighbor_indices = neighbor_indices[0]
    similarity_scores = 1 - distances[0] # Convert distance to similarity

    candidates = df.loc[neighbor_indices].copy()

    # Category similarity computation 
    candidates["category_similarity"] = candidates["main_category_clean"].apply(
        lambda cats: len(ref_categories & set(str(cats).split(","))) / max(1, len(ref_categories | set(str(cats).split(","))))
    )

    # Apply similarity threshold based on similarity_level
    threshold = SIMILARITY_THRESHOLDS[similarity_level]
    similarity_mask = similarity_scores >= threshold
    neighbor_indices = neighbor_indices[similarity_mask]
    similarity_scores = similarity_scores[similarity_mask]
    candidates = candidates.loc[neighbor_indices].copy()
    candidates["category_similarity"] = candidates["category_similarity"].loc[neighbor_indices]

    # Remove the reference product itself
    if label:
        label_mask = candidates["detected_labels"].apply(lambda x: label in x)
        candidates = candidates[label_mask].copy()
        similarity_scores = similarity_scores[label_mask.values]
        if candidates.empty:
            return pd.DataFrame()

    # Custom similarity threshold
    mask = pd.Series(True, index=candidates.index)
    if origin:
        mask &= candidates["origins_clean"].apply(lambda x: origin in x.split(","))
    if brand and not substitute_other_brand:
        mask &= candidates["brands_clean"].apply(lambda x: brand in x.split(","))
    # Same category filter if no label specified
    if not label:
        mask &= candidates["main_category_clean"].apply(
            lambda cats: len(set(str(cats).split(",")) & set(ref_categories)) > 0
        )

    candidates = candidates[mask].copy()
    similarity_scores = similarity_scores[mask.values]
    # If no candidates left
    if candidates.empty:
        return pd.DataFrame()


    # Score computation 
    # Score = (sim text) + (cat sim * weight) + bonuses
    score = (
        similarity_scores +
        category_weight * candidates["category_similarity"].values
        # Other weights applied in compute_score
    )
    candidates["final_score"] = compute_score(
        candidates, score,
        label=label, label_weight=label_weight,
        ref_nutri=ref_nutri, nutri_weight=nutri_weight,
        origin=origin, origin_weight=1.0,
        env_weight=env_weight
    )
    # Better nutriscore flag
    candidates["better_nutriscore"] = candidates["nutriscore_clean"].map(NUTRI_MAP).fillna(3) > ref_nutri

    return candidates.sort_values("final_score", ascending=False).head(top_n)

# Test block
if __name__ == "__main__":
    print("\n🔍 Test du moteur de recommandation\n")

    # Display sample products
    sample = df[["product_name_clean"]].dropna().head(15)
    print("Exemples de produits disponibles :")
    for idx, row in sample.iterrows():
        print(f"{idx} -> {row['product_name_clean']}")

    # User input
    try:
        product_id = int(input("\nEntrez l'ID du produit de référence : "))
        if product_id not in df.index:
            raise ValueError
    except ValueError:
        print("ID invalide ou non présent dans le dataset.")
        exit()

    label = input("Label souhaité (bio, vegan, etc. - optionnel) : ").lower().strip() or None
    origin = input("Origine souhaitée (optionnel) : ").lower().strip() or None
    brand = input("Marque identique ? (laisser vide = peu importe) : ").lower().strip() or None

    # Numerical parameters
    try:
        similarity_level = int(input("Niveau de ressemblance (1-5) : ") or 5)
        label_weight = float(input("Poids du label (défaut=1) : ") or 1)
        nutri_weight = float(input("Poids Nutriscore (défaut=1) : ") or 1)
        env_weight = float(input("Poids environnement (défaut=1) : ") or 1)
    except ValueError:
        print("Valeur incorrecte, paramètres par défaut appliqués.")
        similarity_level = 5
        label_weight = nutri_weight = env_weight = 1

    substitute_other_brand = brand is None

    print("\n Calcul des recommandations...\n")

    recs = recommend_products_web(
        df, nn_model, X_final, product_id,
        top_n=10,
        label=label,
        origin=origin,
        brand=brand,
        substitute_other_brand=substitute_other_brand,
        similarity_level=similarity_level,
        label_weight=label_weight,
        nutri_weight=nutri_weight,
        env_weight=env_weight
    )

    if recs.empty:
        print(" Aucune recommandation trouvée.")
    else:
        print(" Recommandations :\n")
        for _, r in recs.iterrows():
            print(
                f"- {r['product_name_clean']} | "
                f"Score: {round(r['final_score'], 2)} | "
                f"Nutri: {r['nutriscore_clean']} | "
                f"Better Nutri: {r['better_nutriscore']} | "
                f"Labels: {r['labels_tags_clean']} | "
                f"Origine: {r['origins_clean']}"
            )
