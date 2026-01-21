from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import time

# Initialize Flask app
app = Flask(__name__)

# Safety delay to allow recommender.py to finish downloading/extracting models on Render
time.sleep(2) 

try:
    # Importing the recommender engine and data
    from recommender import recommend_products_web, df, nn_model, X_final
    print("✓ Recommender system and data loaded successfully")
except Exception as e:
    print(f"X Error loading recommender: {e}")
    # Define as None to prevent immediate crashes; handle in routes
    df = None
    nn_model = None
    X_final = None

# Home page
@app.route("/")
def home():
    return render_template("home.html")  

# Research page 
@app.route("/research", methods=["GET", "POST"])
def research():
    # Check if data is available before processing
    if df is None:
        return "The system is currently initializing data. Please try again in a moment.", 503

    results = []
    error = None
    if request.method == "POST":
        product_name = request.form.get("product", "").strip() 
        if not product_name:
            error = "Please enter a product name."
            return render_template("research.html", results=[], error=error)
        
        # Case-insensitive partial match to find product
        product_match = df[df["product_name_clean"].str.contains(product_name, case=False, na=False)]
        if product_match.empty:
            error = f"Product '{product_name}' not found."
            return render_template("research.html", results=[], error=error)
        
        product_id = product_match.index[0]
        brand = request.form.get("brand") or None
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
            
        substitute_other_brand = (request.form.get("substitute_other_brand", "oui") == "oui")
        
        # Call the recommendation engine
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
            columns_to_return = [
                'product_name_clean', 'brands_tags_clean', 'main_category_fr_clean',
                'nutriscore_clean', 'nova_group', 'ingredients_tags_clean',
                'environmental_score_score', 'environmental_score_grade',
                'carbon-footprint_100g', 'labels_tags_clean', 'food_groups_tags',
                'origins_clean', 'manufacturing_places', 'countries_tags',
                'completeness_score', 'better_nutriscore', 'final_score',
                'category_similarity', 'detected_labels'
            ]
            
            available_cols = [col for col in columns_to_return if col in df_results.columns]
            results = df_results[available_cols].to_dict(orient="records")
        else:
            error = "No recommendations found for these criteria."
            
    return render_template("research.html", results=results, error=error)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/work")
def work():
    return render_template("work.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    success = False
    if request.method == "POST":
        success = True
    return render_template("contact.html", success=success)

@app.route("/api/search", methods=["GET"])
def search_products():
    if df is None:
        return jsonify([])
        
    query = request.args.get("q", "").strip()
    if len(query) < 2:
        return jsonify([])
    
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

if __name__ == "__main__":
    # Ensure the app binds to 0.0.0.0 for external access on Render
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)