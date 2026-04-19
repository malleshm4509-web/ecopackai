from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# ==============================
# LOAD MODELS
# ==============================
def load_model(path):
    return joblib.load(path)

model_material = load_model("material.pkl")
model_strength = load_model("strength.pkl")
model_cost = load_model("cost.pkl")
model_co2 = load_model("co2.pkl")
model_eco = load_model("eco.pkl")

model_bio = load_model("bio.pkl")
model_recycle = load_model("recycle.pkl")

# ==============================
# LOAD ENCODERS
# ==============================
le_material = load_model("material_encoder.pkl")
le_type = load_model("type_encoder.pkl")
le_fragility = load_model("fragility_encoder.pkl")

# ==============================
# SAFE ENCODER
# ==============================
def safe_encode(encoder, value):
    value = str(value).strip().lower()
    classes = [str(c).lower() for c in encoder.classes_]

    if value in classes:
        index = classes.index(value)
        return encoder.transform([encoder.classes_[index]])[0]
    return 0

# ==============================
# SHIPPING IMPACT
# ==============================
def apply_shipping_effect(shipping, cost, co2):
    shipping = shipping.lower()

    if shipping == "air":
        cost += 10
        co2 += 15
    elif shipping == "sea":
        cost -= 5
        co2 -= 10
    elif shipping == "road":
        cost += 2
        co2 += 3

    return cost, co2

# ==============================
# TOP 5 PREDICTION
# ==============================
def predict_top5(type_, weight, fragility, shipping):

    weight = float(weight)

    t_enc = safe_encode(le_type, type_)
    f_enc = safe_encode(le_fragility, fragility)

    base_input = np.array([[0, t_enc, weight, f_enc]])

    probs = model_material.predict_proba(base_input)[0]
    material_ids = model_material.classes_
    materials = le_material.inverse_transform(material_ids)

    results = []

    for mat, prob in zip(materials, probs):

        try:
            m_enc = le_material.transform([mat])[0]
            full_input = np.array([[m_enc, t_enc, weight, f_enc]])

            strength = float(model_strength.predict(full_input)[0])
            cost = float(model_cost.predict(full_input)[0])
            co2 = float(model_co2.predict(full_input)[0])
            eco = float(model_eco.predict(full_input)[0])

            bio = int(model_bio.predict(full_input)[0])
            recycle = int(model_recycle.predict(full_input)[0])

            cost, co2 = apply_shipping_effect(shipping, cost, co2)

        except Exception as e:
            print("Error:", e)
            strength, cost, co2, eco = 80, 50, 20, 60
            bio, recycle = 1, 1

        final_score = eco + (prob * 100)

        results.append({
            "Material": mat,
            "Strength": round(strength, 2),
            "Cost": round(cost, 2),
            "CO2": round(co2, 2),
            "Bio": bio,
            "Recycle": recycle,
            "EcoScore": round(final_score, 2),
            "Shipping": shipping
        })

    results = sorted(results, key=lambda x: x["EcoScore"], reverse=True)

    return {"top5": results[:5]}

# ==============================
# ROUTE
# ==============================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        try:
            weight = float(request.form.get("weight", 1))
            type_ = request.form.get("type", "general")
            fragility = request.form.get("fragility", "low")
            shipping = request.form.get("shipping", "road")

            result = predict_top5(type_, weight, fragility, shipping)

        except Exception as e:
            print("Error:", e)
            result = {"top5": []}

    return render_template("index.html", result=result)

# ==============================
# RUN APP (RENDER SAFE)
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)