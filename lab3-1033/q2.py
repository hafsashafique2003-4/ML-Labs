# Hafsa-1033

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


# Load dataset
data = pd.read_csv("Car Price Prediction.csv")

print(data.head())
print(data.info())


# Select features from YOUR dataset
X = data[[
    "enginesize",
    "horsepower",
    "curbweight",
    "citympg"
]]

y = data["price"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------
# Linear Regression
# --------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_rmse = np.sqrt(
    mean_squared_error(y_test, linear_pred)
)

linear_r2 = r2_score(y_test, linear_pred)

print("\nLinear Regression")
print("-----------------------")
print("RMSE:", linear_rmse)
print("R2 Score:", linear_r2)


# --------------------------------
# Polynomial Regression
# --------------------------------

results = {}

for degree in [2, 3, 4]:

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False
    )

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()

    model.fit(X_train_poly, y_train)

    predictions = model.predict(X_test_poly)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results[degree] = {
        "RMSE": rmse,
        "R2": r2
    }

    print("\nPolynomial Regression Degree", degree)
    print("-----------------------")
    print("RMSE:", rmse)
    print("R2 Score:", r2)


# --------------------------------
# Compare Results
# --------------------------------

print("\n========== COMPARISON ==========")

print("\nLinear Regression")
print("RMSE:", linear_rmse)
print("R2 Score:", linear_r2)

for degree in [2, 3, 4]:

    print("\nPolynomial Degree", degree)
    print("RMSE:", results[degree]["RMSE"])
    print("R2 Score:", results[degree]["R2"])


# --------------------------------
# Fitted Curves
# --------------------------------

# We use horsepower for the X-axis
# Other features are kept at their median values

horsepower_values = np.linspace(
    data["horsepower"].min(),
    data["horsepower"].max(),
    200
)

curve_data = pd.DataFrame({
    "enginesize": data["enginesize"].median(),
    "horsepower": horsepower_values,
    "curbweight": data["curbweight"].median(),
    "citympg": data["citympg"].median()
})


plt.figure(figsize=(10, 6))

# Actual data
plt.scatter(
    data["horsepower"],
    y,
    alpha=0.4,
    label="Actual Data"
)


# Linear Regression curve
linear_curve = linear_model.predict(curve_data)

plt.plot(
    horsepower_values,
    linear_curve,
    label="Linear Regression"
)


# Polynomial Regression curves
for degree in [2, 3, 4]:

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False
    )

    X_train_poly = poly.fit_transform(X_train)

    model = LinearRegression()

    model.fit(X_train_poly, y_train)

    curve_poly = poly.transform(curve_data)

    predictions = model.predict(curve_poly)

    plt.plot(
        horsepower_values,
        predictions,
        label=f"Polynomial Degree {degree}"
    )


plt.xlabel("Horsepower")
plt.ylabel("Car Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()