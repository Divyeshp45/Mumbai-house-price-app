from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("house_model.pkl")
columns = joblib.load("columns.pkl")

@app.route("/")
def home():
    locations = [col.replace("locality_", "") for col in columns if "locality_" in col]
    return render_template("index.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    area = int(request.form["area"])
    bhk = int(request.form["bhk"])
    location = request.form["location"]

    input_data = pd.DataFrame([[0]*len(columns)], columns=columns)

    input_data["bhk"] = bhk
    input_data["area"] = area

    col_name = "locality_" + location
    if col_name in input_data.columns:
        input_data[col_name] = 1

    prediction = model.predict(input_data)[0]

    return render_template("index.html",
                           prediction=round(prediction, 2),
                           locations=[col.replace("locality_", "") for col in columns if "locality_" in col])

if __name__ == "__main__":
    app.run(debug=True)