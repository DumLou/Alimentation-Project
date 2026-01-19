from flask import Flask, render_template, request, jsonify, redirect, url_for
from recommender import recommend_products_web, df, nn_model, X_final

# Initialize Flask app
app = Flask(__name__)


# Home page
@app.route("/")
def home():
    return render_template("home.html")

# Research page 
@app.route("/research", methods=["GET", "POST"])
def research():
    results = []
    error = None
    # Handle form submission
    if request.method == "POST":
        product_name = request.form.get("product", "").strip() # Get product name from form
        if not product_name:
            error = "Please enter a product name."
            return render_template("research.html", results=[], error=error)
        
        # Find product ID by name (case-insensitive partial match)  
        product_match = df[df["product_name_clean"].str.contains(product_name, case=False, na=False)]
        if product_match.empty:
            error = f"Product '{product_name}' not found."
            return render_template("research.html", results=[], error=error)
        
        # Get the first matching product ID
        product_id = product_match.index[0]
        brand = request.form.get("brand") or None
        nutriscore = request.form.get("nutriscore") or None
        label = request.form.get("label") or None
        origin = request.form.get("origin") or None
        try:
            similarity_level = int(request.form.get("similarity_level", 5))
            category_weight = float(request.form.get("category_weight", 1.0))
            label_weight = float(request.form.get("label_weight", 1.0))
            nutri_weight = float(request.form.get("nutri_weight", 1.0))
            env_weight = float(request.form.get("env_weight", 1.0))
        except ValueError:
            similarity_level = 5
            category_weight = label_weight = nutri_weight = env_weight = 1.0
        substitute_other_brand = (
            request.form.get("substitute_other_brand", "oui") == "oui"
        )
        # Call recommendation engine
        df_results = recommend_products_web(
            df, nn_model, X_final,
            product_id=product_id,
            top_n=10,
            label=label,
            origin=origin,
            brand=brand,
            substitute_other_brand=substitute_other_brand,
            similarity_level=similarity_level,
            category_weight=category_weight,
            label_weight=label_weight,
            nutri_weight=nutri_weight,
            env_weight=env_weight
        )
        if df_results is not None and not df_results.empty:
            # Select columns to return - FICHE PRODUIT COMPLÈTE
            columns_to_return = [
                # Infos basiques
                'product_name_clean', 
                'brands_tags_clean', 
                'main_category_fr_clean',
                
                # Nutrition & santé
                'nutriscore_clean',
                'nova_group',
                'ingredients_tags_clean',
                
                # Environnement
                'environmental_score_score',
                'environmental_score_grade',
                'carbon-footprint_100g',
                
                # Labels & certifications
                'labels_tags_clean',
                'food_groups_tags',
                
                # Origine & lieu de production
                'origins_clean',
                'manufacturing_places',
                'countries_tags',
                
                # Qualité des données
                'completeness_score',
                
                # Scoring & recommandation
                'better_nutriscore',
                'final_score',
                'category_similarity',
                'detected_labels'
            ]
            
            # Garder seulement les colonnes qui existent
            available_cols = [col for col in columns_to_return if col in df_results.columns]
            results = df_results[available_cols].to_dict(orient="records")
        else:
            error = "No recommendations found for these criteria."
    return render_template("research.html", results=results, error=error)

# Page about
@app.route("/about")
def about():
    return render_template("about.html")

# Page work
@app.route("/work")
def work():
    return render_template("work.html")

# Page contact (GET/POST)
@app.route("/contact", methods=["GET", "POST"])
def contact():
    success = False
    if request.method == "POST":
        # Get request data
        success = True
    return render_template("contact.html", success=success)


# API endpoint for autocomplete
@app.route("/api/search", methods=["GET"])
def search_products():
    """Return products matching the search query"""
    query = request.args.get("q", "").strip()
    
    if len(query) < 2:
        return jsonify([])
    
    # Search products by name
    matches = df[df["product_name_clean"].str.contains(query, case=False, na=False)].head(20)
    
    results = [
        {
            "id": int(idx),
            "name": row["product_name_clean"],
            "brand": str(row.get("brands_clean", "")).strip() or "-",
            "category": str(row.get("main_category_clean", "")).strip() or "-"
        }
        for idx, row in matches.iterrows()
    ]
    
    return jsonify(results)


# Run the app
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
