from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
# Feature Importance


# Load dataset
data = pd.read_csv("customer_churn.csv")
data = data.drop("CustomerID", axis=1)

label_cols = ["Gender", "Location", "ContractType", "PaymentMethod", "AutoPayEnabled"]

encoders = {}
for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

X = data.drop("Churn", axis=1)
y = data["Churn"]

model = RandomForestClassifier()
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""

    if request.method == "POST":
        input_data = {
            "Tenure": int(request.form["Tenure"]),
            "Age": int(request.form["Age"]),
            "Gender": request.form["Gender"],
            "Location": request.form["Location"],
            "ContractType": request.form["ContractType"],
            "PaymentMethod": request.form["PaymentMethod"],
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"]),
            "AvgMonthlyUsageGB": float(request.form["AvgMonthlyUsageGB"]),
            "NumSupportCalls": int(request.form["NumSupportCalls"]),
            "NumAppLogins": int(request.form["NumAppLogins"]),
            "TimeSinceLastLogin_Days": int(request.form["TimeSinceLastLogin_Days"]),
            "NumLatePayments": int(request.form["NumLatePayments"]),
            "FailedPayments": int(request.form["FailedPayments"]),
            "AutoPayEnabled": request.form["AutoPayEnabled"],
            "NumComplaints": int(request.form["NumComplaints"]),
            "AvgTicketResolution_Hours": float(request.form["AvgTicketResolution_Hours"]),
            "SatisfactionScore": float(request.form["SatisfactionScore"])
        }

        df = pd.DataFrame([input_data])

        for col in label_cols:
            df[col] = encoders[col].transform(df[col])

        prediction = model.predict(df)[0]

        if prediction == 1:
            prediction_text = "⚠ Customer is likely to CHURN"
        else:
            prediction_text = "✅ Customer is NOT likely to churn"

    return render_template("index.html", prediction_text=prediction_text)
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(10,5))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance - Customer Churn")
plt.show()
if __name__ == "__main__":
    app.run(debug=True)