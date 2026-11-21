import joblib
import pandas as pd

model = joblib.load("linear_regression_model.pkl")

features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"]

print("Enter details for a California district:")
values = [float(input(f"{name}: ")) for name in features]

X_new = pd.DataFrame([values], columns=features)
price = model.predict(X_new)[0] * 100000

print(f"\nPredicted median house value: ${price:,.0f}")