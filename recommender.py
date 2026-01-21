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
from sklearn.metrics.pairwise import cosine_distances

# SETUP PATHS AND DOWNLOAD/EXTRACT MODELS
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists to avoid FileNotFoundError

ZIP_ID = "1z5RsjM7pxHJ_0FjNiLcdF7WJYYYbH7H0"
ZIP_PATH = BASE_DIR / "models.zip"

# Download the ZIP file from Google Drive if it's not already on the server
if not ZIP_PATH.exists():
    print("Downloading models ZIP from Google Drive via gdown...")
    gdown.download(id=ZIP_ID, output=str(ZIP_PATH), quiet=False)

# Extract models if the main data file (parquet) is missing
if not (MODEL_DIR / "products.parquet").exists():
    print("Extracting models from ZIP...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        # Extract everything directly into MODEL_DIR
        zip_ref.extractall(MODEL_DIR)
    
    # Bug fix: If the zip created an extra 'models/' subfolder, move files to the parent directory
    internal_folder = MODEL_DIR / "models"
    if internal_folder.exists():
        import shutil
        for file_path in internal_folder.iterdir():
            shutil.move(str(file_path), str(MODEL_DIR / file_path.name))
        print("✓ Files moved from subfolder to root MODEL_DIR")
    
    print("✓ Models extracted successfully")

# Define paths for loading to ensure absolute reference
path_parquet = MODEL_DIR / "products.parquet"
path_embeddings = MODEL_DIR / "embeddings.joblib"
path_nn = MODEL_DIR / "nn_model.joblib"

# Final safety check before loading
if not path_parquet.exists():
    raise FileNotFoundError(f"CRITICAL ERROR: {path_parquet} was not found after extraction.")

# Load data and models into memory
df = pd.read_parquet(path_parquet)
X_final = joblib.load(path_embeddings)
nn_model = joblib.load(path_nn)


# Text cleaning 
TEXT_COLUMNS = ["product_name_clean", "brands_clean", "main_category_clean",
                "labels_tags_clean", "ingredients_tags_clean", "nutriscore_clean", "origins_clean"]
for col in TEXT_COLUMNS:
    if col not in df.columns:
        df[col] = ""

# Attributes to numeric mappings and thresholds
NUTRI_MAP = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}
SIMILARITY_THRESHOLDS = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.7, 5: 0.8}

# Product type keywords for smart detection
ANIMAL_PRODUCTS = {
    "meat": ["meat", "chicken", "beef", "pork", "lamb", "turkey", "veal", "viande", "poulet", "boeuf", "porc"],
    "fish": ["fish", "salmon", "tuna", "cod", "trout", "poisson", "saumon", "thon", "morue"],
    "dairy": ["milk", "cheese", "yogurt", "butter", "cream", "fromage", "lait", "yaourt", "crème", "beurre"],
    "egg": ["egg", "œuf", "eggs", "œufs"]
}

VEGAN_REPLACEMENTS = {
    "meat": ["tofu", "tempeh", "seitan", "lentil", "pea", "bean", "légumineuse", "protéine végétale"],
    "fish": ["algae", "seaweed", "tofu", "tempeh", "algue"],
    "dairy": ["almond milk", "soy milk", "oat milk", "coconut milk", "lait d'amande", "lait de soja", "lait d'avoine"],
    "egg": ["flax", "chia", "egg replacement", "lin", "chia"]
}

def detect_product_type(product_name: str, category: str):
    """DDetect the product type (meat, fish, dairy, egg)"""
    text = f"{product_name} {category}".lower()
    for ptype, keywords in ANIMAL_PRODUCTS.items():
        for kw in keywords:
            if kw in text:
                return ptype
    return None

def is_vegan_replacement(product_name: str, product_type: str):
    """ Check if product_name is a vegan replacement for the given product_type """
    if not product_type or product_type not in VEGAN_REPLACEMENTS:
        return False
    text = product_name.lower()
    for kw in VEGAN_REPLACEMENTS[product_type]:
        if kw in text:
            return True
    return False

# Vegan ingredient patterns for regex matching
VEGAN_INGREDIENTS_REGEX = r"(soja|soy|tofu|tempeh|seitan|lentil|lentille|pois|pea|beans|legumineuse|vegetal|cereal|riz|ble|avoine|oat|lupini|pois-chiche|chickpea|feve)"

def has_vegan_ingredients(ingredients_text):
    """Check if ingredients contain vegan protein sources using regex"""
    if not isinstance(ingredients_text, str) or ingredients_text.strip() == "":
        return False
    text = ingredients_text.lower()
    return bool(re.search(VEGAN_INGREDIENTS_REGEX, text))

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
    
    # Detect if ref product is animal-based + vegan requested
    ref_product_type = detect_product_type(ref_product["product_name_clean"], ref_product["main_category_clean"])
    smart_vegan_mode = (label == "vegan" and ref_product_type is not None)

    # If a label is requested, filter from the entire dataframe first
    if label:
        # Find products with the requested label
        label_candidates_mask = df["detected_labels"].apply(lambda x: label in x)
        
        # For "vegan" label, also include products with vegan ingredients
        if label == "vegan":
            vegan_ingredients_mask = df["ingredients_tags_clean"].apply(has_vegan_ingredients)
            label_candidates_mask = label_candidates_mask | vegan_ingredients_mask
        
        label_candidates_indices = df[label_candidates_mask].index.tolist()
        
        if not label_candidates_indices:
            # No products with this label found
            return pd.DataFrame()
        
        # Convert indices to positional indices for the model
        # Find nearest neighbors among LABEL candidates only
        similarities_all = []
        for idx in label_candidates_indices:
            # Compute cosine similarity (same as KNN uses)
            dist = cosine_distances([X_final[product_id]], [X_final[idx]])[0][0]
            sim = 1 - dist  # Convert distance to similarity (cosine returns 0-2, so 1-dist gives -1 to 1, clamped to 0-1)
            sim = max(0, min(1, sim))  # Clamp to [0, 1]
            similarities_all.append((idx, sim))
        
        # Sort by similarity
        similarities_all.sort(key=lambda x: x[1], reverse=True)
        
        # For label-based searches: ignore threshold, take top N by similarity
        # This ensures we always return results when user filters by label (vegan, bio, etc)
        # We'll use a softer minimum: only products with sim > 0.1 (very relaxed)
        min_similarity = 0.1  # Very relaxed threshold for label searches
        filtered = [(idx, sim) for idx, sim in similarities_all if sim >= min_similarity]
        
        if not filtered:
            # If even 0.1 threshold fails, just take top 20 anyway
            filtered = similarities_all[:20]
        
        neighbor_indices = np.array([idx for idx, _ in filtered])
        similarity_scores = np.array([sim for _, sim in filtered])
    else:
        # Original KNN logic (no label filter)
        n_neighbors = min(400, len(df))
        distances, neighbor_indices = nn_model.kneighbors(
            X_final[product_id].reshape(1, -1), n_neighbors=n_neighbors
        ) 
        neighbor_indices = neighbor_indices[0]
        similarity_scores = 1 - distances[0]

        # Apply similarity threshold based on similarity_level
        threshold = SIMILARITY_THRESHOLDS[similarity_level]
        similarity_mask = similarity_scores >= threshold
        neighbor_indices = neighbor_indices[similarity_mask]
        similarity_scores = similarity_scores[similarity_mask]

    candidates = df.loc[neighbor_indices].copy()

    # Category similarity computation 
    candidates["category_similarity"] = candidates["main_category_clean"].apply(
        lambda cats: len(ref_categories & set(str(cats).split(","))) / max(1, len(ref_categories | set(str(cats).split(","))))
    )

    # Remove the reference product itself
    candidates = candidates[candidates.index != product_id]
    if candidates.empty:
        return pd.DataFrame()

    # Custom similarity threshold
    mask = pd.Series(True, index=candidates.index)
    if origin:
        mask &= candidates["origins_clean"].apply(lambda x: origin in str(x).split(",") if pd.notna(x) else False)
    if brand and not substitute_other_brand:
        mask &= candidates["brands_clean"].apply(lambda x: brand in str(x).split(",") if pd.notna(x) else False)
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
    
    # Boost score for good vegan replacements
    if smart_vegan_mode:
        # **NEW APPROACH**: Boost based on ingredients (much more reliable!)
        # Check if ingredients contain vegan protein sources
        vegan_ingredient_bonus = candidates["ingredients_tags_clean"].apply(
            lambda ing: 2.0 if has_vegan_ingredients(ing) else 0
        )
        candidates["final_score"] = candidates["final_score"] + vegan_ingredient_bonus
    
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
