from flask import Flask, render_template, request
import joblib
import pandas as pd
import csv
import os


app = Flask(__name__)


# =========================================
# LOAD TRAINED MODEL
# =========================================

model = joblib.load(
    "models/car_price_model.pkl"
)

metrics = joblib.load(
    "models/metrics.pkl"
)
feature_importance = joblib.load("models/feature_importance.pkl")


# =========================================
# HOME
# =========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================
# PREDICT
# =========================================


        # =========================================
        # VALIDATION
        # =========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get form data
        year = int(request.form["year"])
        present_price = float(request.form["present_price"])
        driven_kms = int(request.form["driven_kms"])
        fuel_type = request.form["fuel_type"]
        selling_type = request.form["selling_type"]
        transmission = request.form["transmission"]
        owner = int(request.form["owner"])


        # -----------------------------
        # Input Validation
        # -----------------------------

        if year < 2000 or year > 2026:
            return render_template(
                "error.html",
                message="Please enter a valid car year between 2000 and 2026."
            )

        if present_price <= 0:
            return render_template(
                "error.html",
                message="Present price must be greater than 0."
            )

        if driven_kms < 0:
            return render_template(
                "error.html",
                message="Driven kilometers cannot be negative."
            )

        if owner < 0 or owner > 3:
            return render_template(
                "error.html",
                message="Owner value must be between 0 and 3."
            )


        # -----------------------------
        # Prepare Input Data
        # -----------------------------

        input_data = pd.DataFrame([{

            "Year": year,

            "Present_Price": present_price,

            "Driven_kms": driven_kms,

            "Owner": owner,

            "Fuel_Type_Diesel":
                1 if fuel_type == "Diesel" else 0,

            "Fuel_Type_Petrol":
                1 if fuel_type == "Petrol" else 0,

            "Selling_type_Individual":
                1 if selling_type == "Individual" else 0,

            "Transmission_Manual":
                1 if transmission == "Manual" else 0

        }])


        # -----------------------------
        # Prediction
        # -----------------------------

        prediction = float(
            model.predict(input_data)[0]
        )


        # -----------------------------
        # Price Analysis
        # -----------------------------

        depreciation = present_price - prediction

        depreciation_percentage = (
            (depreciation / present_price) * 100
        )


        # -----------------------------
        # Prediction Category
        # -----------------------------

        if prediction < present_price * 0.50:

            price_category = "High Depreciation"

        elif prediction < present_price * 0.75:

            price_category = "Moderate Depreciation"

        else:

            price_category = "Lower Depreciation"


        # -----------------------------
        # Save Prediction History
        # -----------------------------

        history_file = "prediction_history.csv"

        file_exists = os.path.isfile(history_file)


        with open(
            history_file,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(file)


            if not file_exists:

                writer.writerow([
                    "Year",
                    "Present_Price",
                    "Driven_kms",
                    "Fuel_Type",
                    "Selling_type",
                    "Transmission",
                    "Owner",
                    "Predicted_Price"
                ])


            writer.writerow([

                year,

                present_price,

                driven_kms,

                fuel_type,

                selling_type,

                transmission,

                owner,

                round(prediction, 2)

            ])


        # -----------------------------
        # Result Page
        # -----------------------------

        return render_template(

            "result.html",

            prediction=prediction,

            present_price=present_price,

            depreciation=depreciation,

            depreciation_percentage=depreciation_percentage,

            price_category=price_category,

            year=year,

            driven_kms=driven_kms,

            fuel_type=fuel_type,

            selling_type=selling_type,

            transmission=transmission,

            owner=owner

        )


    except Exception as e:

        return render_template(

            "error.html",

            message=(
                "Unable to process prediction. "
                f"Error: {str(e)}"
            )

        )
# =========================================
# DASHBOARD
# =========================================

@app.route("/dashboard")
def dashboard():

    history_file = (
        "prediction_history.csv"
    )


    # =========================================
    # LOAD HISTORY
    # =========================================

    if os.path.exists(
        history_file
    ):

        history_data = pd.read_csv(
            history_file
        )

    else:

        history_data = pd.DataFrame()


    # =========================================
    # KPI CALCULATIONS
    # =========================================

    if (
        not history_data.empty
        and "Predicted_Price"
        in history_data.columns
    ):

        total_predictions = len(
            history_data
        )


        average_price = float(
            history_data[
                "Predicted_Price"
            ].mean()
        )


        highest_price = float(
            history_data[
                "Predicted_Price"
            ].max()
        )


        lowest_price = float(
            history_data[
                "Predicted_Price"
            ].min()
        )


    else:

        total_predictions = 0

        average_price = 0.0

        highest_price = 0.0

        lowest_price = 0.0


    # =========================================
    # CHART DATA
    # =========================================

    prediction_labels = []

    prediction_values = []


    if (
        not history_data.empty
        and "Predicted_Price"
        in history_data.columns
    ):

        prediction_labels = list(
            range(
                1,
                len(history_data) + 1
            )
        )


        prediction_values = [

            float(value)

            for value in
            history_data[
                "Predicted_Price"
            ].tolist()

        ]


    # =========================================
    # DASHBOARD
    # =========================================

    return render_template(

        "dashboard.html",

        metrics=metrics,

        total_predictions=
            total_predictions,

        average_price=
            average_price,

        highest_price=
            highest_price,

        lowest_price=
            lowest_price,

        prediction_labels=
            prediction_labels,

        prediction_values=
            prediction_values,
        feature_importance=
            feature_importance,

    )


# =========================================
# HISTORY
# =========================================

@app.route("/history")
def history():

    history_file = (
        "prediction_history.csv"
    )


    if os.path.exists(
        history_file
    ):

        history_data = pd.read_csv(
            history_file
        )

    else:

        history_data = pd.DataFrame()


    return render_template(

        "history.html",

        history=history_data

    )


# =========================================
# ABOUT
# =========================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# =========================================
# RUN APPLICATION
# =========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )