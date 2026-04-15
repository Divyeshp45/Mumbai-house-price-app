from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# Load model & columns
model = joblib.load("house_model.pkl")
columns = joblib.load("columns.pkl")

# Get locations
def get_locations():
    return [col.replace("locality_", "") for col in columns if "locality_" in col]

# Home route
@app.route("/")
def home():
    return render_template("index.html",
                           locations=get_locations(),
                           selected_location=None,
                           bhk=None,
                           area=None,
                           prediction=None)

# Prediction route
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("home"))

    try:
        area = int(request.form["area"])
        bhk = int(request.form["bhk"])
        location = request.form["location"]

        # Validation
        if area <= 0 or bhk <= 0:
            return render_template("index.html",
                                   locations=get_locations(),
                                   selected_location=location,
                                   bhk=bhk,
                                   area=area,
                                   prediction="Invalid input")

        # Create input dataframe
        input_data = pd.DataFrame([[0]*len(columns)], columns=columns)

        input_data["bhk"] = bhk
        input_data["area"] = area

        col_name = "locality_" + location
        if col_name in input_data.columns:
            input_data[col_name] = 1

        # Predict
        prediction = model.predict(input_data)[0]

        return render_template("index.html",
                               locations=get_locations(),
                               selected_location=location,
                               bhk=bhk,
                               area=area,
                               prediction=round(prediction, 2))

    except Exception as e:
        return render_template("index.html",
                               locations=get_locations(),
                               selected_location=None,
                               bhk=None,
                               area=None,
                               prediction="Error occurred")

# Run app
if __name__ == "__main__":
    app.run(debug=True)