from flask import Flask, render_template, request
from recommender import recommend_products
import os

app = Flask(__name__)

# Home route 
@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        # Get form data
        product = request.form.get("product", "").strip()

        if not product:
            return render_template(
                "index.html",
                results=[],
                error="Veuillez entrer un nom de produit."
            )
        # Optional parameters
        brand = request.form.get("brand") or None
        nutriscore = request.form.get("nutriscore") or None
        label = request.form.get("label") or None
        origin = request.form.get("origin") or None

        # Numerical parameters
        try:
            similarity_level = int(request.form.get("similarity_level", 5)) # 1 to 5
            label_weight = float(request.form.get("label_weight", 1.0))
            nutri_weight = float(request.form.get("nutri_weight", 1.0))
            env_weight = float(request.form.get("env_weight", 1.0))
        except ValueError:
            similarity_level = 5
            label_weight = nutri_weight = env_weight = 1.0

        substitute_other_brand = (
            request.form.get("substitute_other_brand", "oui") == "oui"
        )

        # Recognition engine call
        df_results = recommend_products(
            product_name=product,
            brand=brand,
            nutriscore=nutriscore,
            label=label,
            origin=origin,
            substitute_other_brand=substitute_other_brand,
            similarity_level=similarity_level,
            label_weight=label_weight,
            nutri_weight=nutri_weight,
            env_weight=env_weight,
            top_n=10
        )
        # Process results
        if df_results is not None and not df_results.empty:
            results = df_results.to_dict(orient="records")

    return render_template("index.html", results=results)


# Run the app (Render compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render port
    app.run(host="0.0.0.0", port=port)
