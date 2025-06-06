import tkinter as tk
from tkinter import ttk, messagebox
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import pickle

# Load scaler and columns
scaler = joblib.load("scaler.save")
with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

# Define the model
class IncomeRegressor(nn.Module):
    def __init__(self, input_dim):
        super(IncomeRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# Load model
model = IncomeRegressor(len(columns))
model.load_state_dict(torch.load("income_model.pth", map_location=torch.device('cpu')))
model.eval()

# List of input fields
# NOTE: Capital Gain and Loss are not included here because of some bugs related to them in this GUI version.
fields = ["age", "workclass", "education.num", "marital.status", "occupation", "relationship", "race", "sex", "hours.per.week", "native.country"]

# Categorical options
categorical_options = {
    "age": ["17-30", "31-50", "51-70", "70+"],
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Local-gov", "State-gov", "Federal-gov", "Without-pay", "Never-worked"],
    "marital.status": ["Never-married", "Divorced", "Separated", "Widowed", "Married-civ-spouse", "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Prof-specialty", "Exec-managerial", "Machine-op-inspct", "Other-service", "Adm-clerical", "Craft-repair", "Transport-moving", "Handlers-cleaners", "Sales", "Farming-fishing"],
    "relationship": ["Not-in-family", "Unmarried", "Own-child", "Other-relative", "Husband", "Wife"],
    "race": ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
    "sex": ["Male", "Female"],
    "native.country": ["United-States", "Other"]
}
numerical_fields = ["education.num", "hours.per.week"]

def predict_income(inputs):
    # Build a DataFrame for one row with all columns from training
    data = {col: [0] for col in columns}
    for field in fields:
        val = inputs[field]
        # If this field is one-hot encoded (categorical)
        if any(col.startswith(f"{field}_") for col in columns):
            col_name = f"{field}_{val}"
            if col_name in data:
                data[col_name] = [1]
        # If this field is a numerical column
        elif field in columns:
            data[field] = [float(val)]
    df = pd.DataFrame(data)
    X = scaler.transform(df)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X_tensor).item()
    return pred

def on_predict():
    inputs = {}
    for field in fields:
        if field in categorical_options:
            inputs[field] = entries[field].get()
        else:
            try:
                inputs[field] = float(entries[field].get())
            except ValueError:
                messagebox.showerror("Input Error", f"Invalid value for {field}")
                return
    prob = predict_income(inputs)
    result = ">50K" if prob > 0.5 else "<=50K"
    messagebox.showinfo("Prediction", f"Predicted Income: {result}\nProbability: {prob:.2f}")

# Tkinter GUI
root = tk.Tk()
root.title("Income Prediction")

entries = {}
for idx, field in enumerate(fields):
    tk.Label(root, text=field).grid(row=idx, column=0, padx=5, pady=5, sticky="w")
    if field in categorical_options:
        cb = ttk.Combobox(root, values=categorical_options[field])
        cb.current(0)
        cb.grid(row=idx, column=1, padx=5, pady=5)
        entries[field] = cb
    else:
        ent = tk.Entry(root)
        ent.grid(row=idx, column=1, padx=5, pady=5)
        entries[field] = ent

tk.Button(root, text="Predict", command=on_predict).grid(row=len(fields), column=0, columnspan=2, pady=10)
root.mainloop()