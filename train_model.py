# train_model.py

# IMPORTS
import pandas as pd
import numpy as np
from pathlib import Path
import re
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# PATHS
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# LOAD CSV
try:
    DATA_PATH = next(DATA_DIR.glob("*.csv"))
    print("📄 Fichier utilisé :", DATA_PATH)
except StopIteration:
    print("❌ Aucun fichier CSV trouvé dans", DATA_DIR)
    exit()

print("📁 Dossier modèles :", MODEL_DIR)


# CLEAN TEXT
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove language prefixes (e.g. ‘en:’, ‘fr:’)
    text = re.sub(r"\b[a-z]{2,3}:", "", text)
    # Standardise separators and spaces
    text = text.replace(";", ",")
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Columns to be cleaned
TEXT_COLUMNS = [
    "product_name", "brands_tags", "main_category_fr",
    "labels_tags", "ingredients_tags", "nutriscore_grade", "origins"
]

# Cleaning
df = pd.read_csv(DATA_PATH, sep=";")
df = df[df["product_name"].notna()].reset_index(drop=True)

for col in TEXT_COLUMNS:
    df[col + "_clean"] = df[col].fillna("").map(clean_text)

# Clean Nutriscore column for mapping
df["nutriscore_clean"] = df["nutriscore_grade"].fillna("").map(clean_text)

# Nutriscore mapping
NUTRI_MAP = {"a":5, "b":4, "c":3, "d":2, "e":1}


# ENCODER DENSE
print("🔹 Encodage dense des textes avec SentenceTransformer...")
encoder_model = SentenceTransformer('all-MiniLM-L6-v2')

# Relevant columns for the dense vectors
TEXT_EMB_COLUMNS = ["product_name_clean", "ingredients_tags_clean",
                    "categories_clean" if "categories_clean" in df.columns else "main_category_fr_clean",
                    "food_groups_tags_clean" if "food_groups_tags_clean" in df.columns else "main_category_fr_clean"]

# Check if columns are present
for col in TEXT_EMB_COLUMNS:
    if col not in df.columns:
        df[col] = ""

# Encode each column separately
print("Encodage product_name...")
name_emb = encoder_model.encode(df["product_name_clean"].tolist(), show_progress_bar=True)
print("Encodage ingredients_tags...")
ingredients_emb = encoder_model.encode(df["ingredients_tags_clean"].tolist(), show_progress_bar=True)
print("Encodage categories...")
categories_emb = encoder_model.encode(df[TEXT_EMB_COLUMNS[2]].tolist(), show_progress_bar=True)
print("Encodage food_groups_tags...")
foodgroups_emb = encoder_model.encode(df[TEXT_EMB_COLUMNS[3]].tolist(), show_progress_bar=True)

# Apply weights for importance
X_text = np.hstack([
    name_emb * 2.0,          # important product name
    ingredients_emb * 3.0,   # even more important ingredients
    categories_emb * 1.0,
    foodgroups_emb * 1.0
])
print("Dimension vecteurs texte :", X_text.shape)


# DIGITAL FEATURES
NUMERIC_COLUMNS = ["carbon-footprint_100g", "environmental_score_score"]
X_num = df[NUMERIC_COLUMNS].fillna(0)
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# ORDINAL FEATURES
ORDINAL_COLUMNS = ["nutriscore_clean", "nova_group", "environmental_score_grade"]
X_ord = df[ORDINAL_COLUMNS].fillna("missing").astype(str)
ordinal_encoder = OrdinalEncoder()
X_ord_encoded = ordinal_encoder.fit_transform(X_ord)

# FINAL EMBEDDINGS
X_final = np.hstack([X_text, X_num_scaled, X_ord_encoded])
print("📐 Dimension finale :", X_final.shape)

# NEAREST NEIGHBORS
nn_model = NearestNeighbors(n_neighbors=20, metric="cosine", n_jobs=-1)
nn_model.fit(X_final)


# SAVE MODELS & DATA
joblib.dump(encoder_model, MODEL_DIR / "text_encoder.joblib")
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
joblib.dump(ordinal_encoder, MODEL_DIR / "ordinal_encoder.joblib")
joblib.dump(nn_model, MODEL_DIR / "nn_model.joblib")
joblib.dump(X_final, MODEL_DIR / "embeddings.joblib")
df.to_parquet(MODEL_DIR / "products.parquet", index=False)

print("🎉 Entraînement terminé avec succès !")
