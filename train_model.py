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
    print(" Fichier utilisé :", DATA_PATH)
except StopIteration:
    print(" Aucun fichier CSV trouvé dans", DATA_DIR)
    exit()

print(" Dossier modèles :", MODEL_DIR)


# CLEAN TEXT
def fix_encoding_errors(text):
    """Fix common encoding errors from UTF-8/Latin-1 mismatches"""
    if not isinstance(text, str):
        return ""
    
    # Common character replacements for encoding corruption
    replacements = {
        'Ё': 'a',  # Cyrillic E
        'ё': 'a',
        'ћ': 'e',  # Cyrillic c
        'Ћ': 'c',
        'ђ': 'd',  # Cyrillic d
        'Ђ': 'd',
        'Ў': 'u',  # Cyrillic u
        'џ': 'dz',
        'Џ': 'dz',
        '™': '',   # Trademark symbol
        '®': '',   # Registered symbol
        '€': 'e',  # Euro symbol (often corrupted)
        '‰': '',   # Per mille
        '‚': '',   # Various corrupted punctuation
        'ƒ': 'f',
        '„': '',
        '…': '',
        '†': '',
        '‡': '',
        'ˆ': '',
        '‰': '',
        'Š': 's',
        '‹': '',
        'Œ': 'oe',
        ''': "'",
        ''': "'",
        '"': '"',
        '"': '"',
        '•': '-',
        '–': '-',
        '—': '-',
        '˜': '',
        '™': '',
        'š': 's',
        '›': '',
        'œ': 'oe',
        'Ÿ': 'y',
        'ў': 'u',
        'ѓ': 'g',
        'Ѓ': 'g',
        '–': '-',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove any remaining non-ASCII control characters
    text = ''.join(c if ord(c) < 128 or ord(c) > 127 else '' for c in text if ord(c) < 32 or (32 <= ord(c) < 127) or ord(c) > 127)
    
    return text

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Fix encoding errors first
    text = fix_encoding_errors(text)
    
    text = text.lower()

    # Remove language prefixes everywhere (en:, fr:, etc.)
    text = re.sub(r"\b[a-z]{2,3}:", "", text)

    # Replace separators
    text = text.replace(";", ",")
    text = text.replace("-", " ")

    # Normalize commas and spaces
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"\s+", " ", text)

    # Split, remove duplicates, keep order
    items = [item.strip() for item in text.split(",") if item.strip()]
    items = list(dict.fromkeys(items))

    return ",".join(items)

# Columns to be cleaned
TEXT_COLUMNS = [
    "product_name", "brands_tags", "main_category_fr",
    "labels_tags", "ingredients_tags", "nutriscore_grade", "origins"
]

# Cleaning
df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8")
df = df[df["product_name"].notna()].reset_index(drop=True)

import unicodedata
# Normalize Unicode characters and fix encoding issues
def normalize_unicode(x):
    if not isinstance(x, str):
        return x
    
    # First fix encoding errors
    x = fix_encoding_errors(x)
    
    # Decompose Unicode (é → e + accent), then remove accents
    nfd = unicodedata.normalize('NFD', x)
    normalized = ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
    
    return normalized

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(normalize_unicode)

for col in TEXT_COLUMNS:
    df[col + "_clean"] = df[col].fillna("").map(clean_text)

# Clean Nutriscore column for mapping
df["nutriscore_clean"] = df["nutriscore_grade"].fillna("").map(clean_text)

# Nutriscore mapping
NUTRI_MAP = {"a":5, "b":4, "c":3, "d":2, "e":1}


# ENCODER DENSE
print("🔹 Encodage dense des textes avec SentenceTransformer...")
encoder_model = SentenceTransformer('all-MiniLM-L6-v2')

# Check available columns
available_cols = df.columns.tolist()
print(f"Colonnes disponibles : {available_cols}")

# TEXT EMBEDDINGS 
TEXT_EMB_COLUMNS = [
    "product_name_clean",
    "ingredients_tags_clean",
    "brands_tags_clean",
    "main_category_fr_clean"  # Fallback if main_category_clean not present
]

# Ensure all required columns exist
for col in TEXT_EMB_COLUMNS:
    if col not in df.columns:
        print(f"⚠️  Colonne manquante : {col}, création avec valeurs vides")
        df[col] = ""

# Encode each column separately with normalization
print(" Encodage product_name...")
name_emb = encoder_model.encode(
    df["product_name_clean"].fillna("").tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

print(" Encodage ingredients_tags...")
ingredients_emb = encoder_model.encode(
    df["ingredients_tags_clean"].fillna("").tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

print(" Encodage brands...")
brands_emb = encoder_model.encode(
    df["brands_tags_clean"].fillna("").tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Encodage main_category...")
category_emb = encoder_model.encode(
    df["main_category_fr_clean"].fillna("").tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)

# Apply BALANCED weights for importance
X_text = np.hstack([
    name_emb * 1.5,          # important product name
    ingredients_emb * 2.0,   # ingredients are key for recommendations
    brands_emb * 0.8,        # brand less important
    category_emb * 1.0       # category matters
])
print("✅ Dimension vecteurs texte :", X_text.shape)


# DIGITAL FEATURES 
print("\n💾 Traitement des features numériques...")
available_numeric = [col for col in ["carbon-footprint_100g", "environmental_score_score"] if col in df.columns]
print(f"Features numériques disponibles : {available_numeric}")

if available_numeric:
    X_num = df[available_numeric].fillna(df[available_numeric].median())  # Fill with median, not 0
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    print(f"Numeric features scaled: {X_num_scaled.shape}")
else:
    print("  No numeric features found, creating a dummy column")
    X_num_scaled = np.zeros((len(df), 1))

# ORDINAL FEATURES
print("\n Processing ordinal features...")
available_ordinal = [col for col in ["nutriscore_clean", "nova_group", "environmental_score_grade"] if col in df.columns]
print(f"Available ordinal features: {available_ordinal}")

if available_ordinal:
    X_ord = df[available_ordinal].fillna("missing").astype(str)
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_ord_encoded = ordinal_encoder.fit_transform(X_ord)
    print(f" Ordinal features encoded: {X_ord_encoded.shape}")
else:
    print("  No ordinal features found, creating a dummy column")
    X_ord_encoded = np.zeros((len(df), 1))

# FINAL EMBEDDINGS
print("\n🔗 Combining all embeddings...")
X_final = np.hstack([X_text, X_num_scaled, X_ord_encoded])
print(f"Final dimension: {X_final.shape}")

# STATISTICS
print(f"📈 Min embeddings: {X_final.min():.4f}")
print(f"📈 Max embeddings: {X_final.max():.4f}")
print(f"📈 Mean embeddings: {X_final.mean(axis=0)[:5]}")  # First 5 elements
print(f"📈 Contains NaN: {np.isnan(X_final).sum()}")

# Check for NaN values
if np.isnan(X_final).sum() > 0:
    print(" ERROR: Presence of NaN in X_final, filling with 0")
    X_final = np.nan_to_num(X_final, nan=0.0)

# NEAREST NEIGHBORS
print("\n Training Nearest Neighbors model...")
nn_model = NearestNeighbors(n_neighbors=20, metric="cosine", n_jobs=-1) 
nn_model.fit(X_final)
print(" Model trained successfully")

# TESTING THE MODEL
print("\n Test: searching for 5 neighbors for the first product")
distances, indices = nn_model.kneighbors(X_final[:1], n_neighbors=6)           
print("Found indices:", indices[0])
print("Names of similar products:")
for idx in indices[0][1:]:  # Skip the first one (itself)
    print(f"  - {df.iloc[idx]['product_name_clean']}")
print(" Test passed")


# SAVE MODELS & DATA
joblib.dump(encoder_model, MODEL_DIR / "text_encoder.joblib")
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
joblib.dump(ordinal_encoder, MODEL_DIR / "ordinal_encoder.joblib")
joblib.dump(nn_model, MODEL_DIR / "nn_model.joblib")
joblib.dump(X_final, MODEL_DIR / "embeddings.joblib")
df.to_parquet(MODEL_DIR / "products.parquet", index=False)

print("Training completed successfully!")
