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
TEXT_COLUMNS = ["product_name_clean", "brands_clean", "main_category_fr_clean",
                "labels_tags_clean", "ingredients_tags_clean", "nutriscore_clean", "origins_clean",
                "food_groups_tags"]
for col in TEXT_COLUMNS:
    if col not in df.columns:
        df[col] = ""

# Attributes to numeric mappings and thresholds
NUTRI_MAP = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}
SIMILARITY_THRESHOLDS = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.7, 5: 0.8}

# FOOD GROUPS SUBSTITUTION FOR VEGAN MODE
VEGAN_FOOD_GROUP_SUBSTITUTES = {
    "en:fish-meat-eggs": ["en:cereals-and-potatoes", "en:legumes", "en:cereals"],  # Meat/fish → cereals/legumes
    "en:milk-and-dairy-products": ["en:plant-based-milk-substitutes"]  # Dairy → plant milk
}

def get_vegan_food_group_substitutes(food_groups_str):
    """Return vegan substitute food groups for a reference product"""
    if not isinstance(food_groups_str, str) or pd.isna(food_groups_str):
        return []
    
    food_groups_list = str(food_groups_str).split(",")
    substitutes = []
    
    for fg in food_groups_list:
        fg = fg.strip()
        if fg in VEGAN_FOOD_GROUP_SUBSTITUTES:
            substitutes.extend(VEGAN_FOOD_GROUP_SUBSTITUTES[fg])
    
    return substitutes

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
    """Detect labels in the cleaned labels_tags column"""
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
        def safe_origin_check(x):
            try:
                if pd.isna(x):
                    return 0
                return int(origin in str(x).lower().split(","))
            except:
                return 0
        bonus += candidates["origins_clean"].apply(safe_origin_check).values * origin_weight

    # Environmental score bonus
    if "environmental_score_score" in candidates.columns:
        env = candidates["environmental_score_score"].fillna(0).values
        if env.max() > 0:
            bonus += (env / env.max()) * env_weight

    return similarity_scores * (1 + bonus)

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
    ref_categories = set(str(ref_product["food_groups_tags"]).split(",")) if pd.notna(ref_product["food_groups_tags"]) else set()
    
    # Detect if ref product is animal-based + vegan requested
    # Check food_groups_tags directly for "en:fish-meat-eggs" or "en:milk-and-dairy-products"
    ref_food_groups = str(ref_product.get("food_groups_tags", "")).lower()
    is_animal_product = "en:fish-meat-eggs" in ref_food_groups or "en:milk-and-dairy-products" in ref_food_groups
    smart_vegan_mode = (label == "vegan" and is_animal_product)
    
    # For vegan mode: filter to substitute food groups first, then score
    if smart_vegan_mode:
        vegan_substitutes = get_vegan_food_group_substitutes(ref_product["food_groups_tags"])
        if vegan_substitutes:
            # Filter to products with substitute food groups
            substitute_mask = df["food_groups_tags"].apply(
                lambda fg: any(sub in str(fg) for sub in vegan_substitutes) if pd.notna(fg) else False
            )
            substitute_candidates = df[substitute_mask].copy()
            
            # Score by embedding similarity with these candidates
            if not substitute_candidates.empty:
                # Compute similarity scores for all substitute candidates
                similarity_scores_all = 1 - cosine_distances(X_final[product_id].reshape(1, -1), X_final)[0]
                
                # Extract scores for substitute candidates
                candidate_indices = substitute_candidates.index
                candidate_similarity = similarity_scores_all[candidate_indices]
                
                # Apply boost for tofu/tempeh/seitan (well-known vegan proteins)
                protein_boost = substitute_candidates["product_name_clean"].apply(
                    lambda name: 5.0 if any(p in str(name).lower() for p in ["tofu", "tempeh", "seitan"]) else 0
                )
                boosted_similarity = candidate_similarity + protein_boost.values
                
                # Sort by boosted similarity - get MORE candidates for filtering
                top_indices = np.argsort(-boosted_similarity)[:top_n * 10]  # Get many more for filtering
                candidates = substitute_candidates.iloc[top_indices].copy()
                similarity_scores = candidate_similarity[top_indices]
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    else:
        # KNN search logic (normal flow)
        n_neighbors = min(400, len(df))
        distances, neighbor_indices = nn_model.kneighbors(
            X_final[product_id].reshape(1, -1), n_neighbors=n_neighbors
        )
        neighbor_indices = neighbor_indices[0]
        similarity_scores = 1 - distances[0]

        # Apply similarity threshold based on similarity_level (relaxed for label searches)
        if label:
            # For label-based searches, use very relaxed threshold
            threshold = 0.1
        else:
            threshold = SIMILARITY_THRESHOLDS[similarity_level]
        
        similarity_mask = similarity_scores >= threshold
        neighbor_indices = neighbor_indices[similarity_mask]
        similarity_scores = similarity_scores[similarity_mask]

        candidates = df.loc[neighbor_indices].copy()

    # Category similarity computation using food_groups_tags (more standardized)
    candidates["category_similarity"] = candidates["food_groups_tags"].apply(
        lambda cats: len(ref_categories & set(str(cats).split(","))) / max(1, len(ref_categories | set(str(cats).split(",")))) if pd.notna(cats) else 0
    )

    # Remove the reference product itself and sync similarity_scores
    ref_mask = candidates.index != product_id
    candidates = candidates[ref_mask]
    similarity_scores = similarity_scores[ref_mask]
    
    if candidates.empty:
        return pd.DataFrame()

    # Custom filters
    mask = pd.Series(True, index=candidates.index)
    
    # Filter by label if specified
    if label:
        label_mask = candidates["detected_labels"].apply(lambda x: label in x if isinstance(x, list) else False)
        
        # For "vegan" label, also look for food_group substitutes
        if label == "vegan":
            # Use vegan_substitutes calculated earlier if in smart_vegan_mode
            if smart_vegan_mode and vegan_substitutes:
                # Include products with substitute food groups (e.g., legumes for meat)
                vegan_food_groups_mask = candidates["food_groups_tags"].apply(
                    lambda fg: any(sub in str(fg) for sub in vegan_substitutes) if pd.notna(fg) else False
                )
            else:
                vegan_food_groups_mask = False
            
            # Also include officially vegan products
            official_vegan_mask = candidates["detected_labels"].apply(lambda x: "vegan" in x if isinstance(x, list) else False)
            
            # Combine: official vegan OR substitute food groups
            label_mask = official_vegan_mask | vegan_food_groups_mask
        
        mask &= label_mask
    
    # Filter by origin if specified
    if origin:
        def safe_origin_filter(x):
            try:
                if pd.isna(x):
                    return False
                return origin.lower() in str(x).lower()
            except:
                return False
        mask &= candidates["origins_clean"].apply(safe_origin_filter)
    
    # Filter by brand if specified
    if brand and not substitute_other_brand:
        def safe_brand_filter(x):
            try:
                if pd.isna(x):
                    return False
                return brand.lower() in str(x).lower()
            except:
                return False
        mask &= candidates["brands_clean"].apply(safe_brand_filter)
    
    # Filter by same category to keep results relevant (using food_groups_tags for consistency)
    # In smart_vegan_mode: candidates already filtered by substitute food_groups, so skip
    if not smart_vegan_mode:
        def has_similar_food_group(food_groups):
            if pd.isna(food_groups) or not ref_categories:
                return True  # If no reference categories, accept all
            candidate_groups = set(str(food_groups).split(","))
            return bool(candidate_groups & ref_categories)
        
        mask &= candidates["food_groups_tags"].apply(has_similar_food_group)

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
    
    # Boost score for good vegan replacements based on food groups
    if smart_vegan_mode and vegan_substitutes:
        # Boost products that have substitute food groups (already filtered in)
        vegan_food_group_bonus = candidates["food_groups_tags"].apply(
            lambda fg: 2.0 if pd.notna(fg) and any(sub in str(fg) for sub in vegan_substitutes) else 0
        )
        candidates["final_score"] = candidates["final_score"] + vegan_food_group_bonus
        
        # Boost for tofu, tempeh, seitan (well-known vegan proteins)
        protein_boost = candidates["product_name_clean"].apply(
            lambda name: 5.0 if any(p in str(name).lower() for p in ["tofu", "tempeh", "seitan"]) else 0
        )
        candidates["final_score"] = candidates["final_score"] + protein_boost
        
        # Light boost for officially vegan products
        vegan_label_bonus = candidates["detected_labels"].apply(
            lambda x: 0.5 if isinstance(x, list) and "vegan" in x else 0
        )
        candidates["final_score"] = candidates["final_score"] + vegan_label_bonus
    
    # Better nutriscore flag
    candidates["better_nutriscore"] = candidates["nutriscore_clean"].map(NUTRI_MAP).fillna(3) > ref_nutri

    return candidates.sort_values("final_score", ascending=False).head(top_n)

# Test block
if __name__ == "__main__":
    print("\n🔍 Test recommandation\n")

    # Display sample products
    sample = df[["product_name_clean"]].dropna().head(15)
    print("Available products:\n")
    for idx, row in sample.iterrows():
        print(f"{idx} -> {row['product_name_clean']}")

    # User input
    try:
        product_id = int(input("\nEnter the reference product ID: "))
        if product_id not in df.index:
            raise ValueError
    except ValueError:
        print("IInvalid ID or not present in the dataset.")
        exit()

    label = input("Desired label (organic, vegan, etc. - optional): ").lower().strip() or None
    origin = input("Desired origin (optional): ").lower().strip() or None
    brand = input("Same brand? (leave empty = any): ").lower().strip() or None

    # Numerical parameters
    try:
        similarity_level = int(input("Similarity level (1-5): ") or 5)
        label_weight = float(input("Label weight (default=1): ") or 1)
        nutri_weight = float(input("Nutriscore weight (default=1): ") or 1)
        env_weight = float(input("Environment weight (default=1): ") or 1)
    except ValueError:
        print("Incorrect value, default parameters applied.")
        similarity_level = 5
        label_weight = nutri_weight = env_weight = 1

    substitute_other_brand = brand is None

    print("\n Calculating recommendations...\n")

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
        print(" No recommendations found.")
    else:
        print(" Recommendations:\n")
        for _, r in recs.iterrows():
            print(
                f"- {r['product_name_clean']} | "
                f"Score: {round(r['final_score'], 2)} | "
                f"Nutri: {r['nutriscore_clean']} | "
                f"Better Nutri: {r['better_nutriscore']} | "
                f"Labels: {r['labels_tags_clean']} | "
                f"Origine: {r['origins_clean']}"
            )
