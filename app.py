import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("🏠 Mumbai House Price Prediction")

# Load dataset

df = pd.read_csv("mumbai_house.csv")
# Preprocessing
def convert_price(row):
    if row["price_unit"] == "Cr":
        return row["price"] * 100
    else:
        return row["price"]

df["price"] = df.apply(convert_price, axis=1)
df = df.drop(columns=["price_unit"])
df = df[df["price"] < 500]

top_localities = df["locality"].value_counts().head(20).index
df["locality"] = df["locality"].apply(lambda x: x if x in top_localities else "other")

df = df.drop(columns=["type", "status", "age"])
df = pd.get_dummies(df, drop_first=True)

# Train model
X = df.drop(columns=["price"])
y = df["price"]

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Inputs
area = st.number_input("Area (sqft)", 100, 10000, 500)
bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])

location = st.selectbox("Location", [
    "Andheri West", "Borivali West", "Mira Road East",
    "Panvel", "Thane", "other"
])

# Prepare input
input_data = pd.DataFrame(columns=X.columns)
input_data.loc[0] = 0

input_data["bhk"] = bhk
input_data["area"] = area

col_name = "locality_" + location
if col_name in input_data.columns:
    input_data[col_name] = 1

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ₹ {round(prediction[0], 2)} Lakhs")