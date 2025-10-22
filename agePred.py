import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns


pd.set_option("display.max_columns", None)
pd.set_option("mode.copy_on_write", True)

# Data Cleaning


df = pd.read_csv("crocodile_dataset.csv")

def dataCleaner(df):

    df["Observed Length (m)"] = pd.to_numeric(df["Observed Length (m)"], errors="coerce")
    df["Observed Weight (kg)"] = pd.to_numeric(df["Observed Weight (kg)"], errors="coerce")

    df = df.astype({"Common Name": "string", "Scientific Name": "string", "Family": "string", "Genus": "string", "Age Class": "string", 
            "Sex": "string", "Habitat Type": "string", "Conservation Status": "string", "Country/Region": "string"},)

    df = df.drop(columns=["Observer Name", "Notes", "Date of Observation"])


    df = df.apply(lambda col: col.str.strip().str.lower() if col.dtype == "string" else col)

    df["Sex"] = df["Sex"].map({'male':1,'female':0})
    df["Age Class"] = df["Age Class"].map({'hatchling':0,'juvenile':1,'subadult':2,'adult':3})

    return df

data = dataCleaner(df)

# Create Features / Target Variables

x = data.drop(columns=["Age Class"])
y = data["Age Class"]


xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.25)

# ML Preprocessing

scaler = MinMaxScaler()

xTrain = pd.get_dummies(xTrain)
xTest = pd.get_dummies(xTest)

xTest = xTest.reindex(columns=xTrain.columns, fill_value=0)

xTrain = xTrain.fillna(0)
xTest  = xTest.fillna(0)

xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

# Tune Model

def tuneModel(xTrain, yTrain):
    paramGrid = {
        "n_neighbors": range(1,21),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }

    model = KNeighborsClassifier()
    gridSearch  = GridSearchCV(model, paramGrid, cv=5, n_jobs=-1)
    gridSearch.fit(xTrain, yTrain)
    return gridSearch.best_estimator_

bestModel = tuneModel(xTrain, yTrain)


# Evaluation

def evaluate(model, xTest, yTest):
    prediction = model.predict(xTest)
    accuracy = accuracy_score(yTest, prediction)
    matrix = confusion_matrix(yTest, prediction)

    return accuracy, matrix

accuracy, matrix = evaluate(bestModel, xTest, yTest)

print(f'Accuracy: {accuracy*100:.2f}')
print("Matrix")
print(matrix)

