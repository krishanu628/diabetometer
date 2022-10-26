from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
def home(request):
    return render(request, 'home.html')
def predict(request):
    return render(request, 'predict.html')
def result(request):
    diabetes_dataset = pd.read_csv(r"C:\\Users\\krishanu roy\\Desktop\\krish1122\\diabetes.csv")
    x = diabetes_dataset.drop('Outcome', axis=1)
    y = diabetes_dataset[['Outcome']]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = model.predict([val1, val2, val3, val4, val5, val6, val7, val8])

    result1 = ""
    if pred ==[1]:
        result1 = "The person is diabetic"
    else:
        result1 = "The person is non-diabetic"
    return render(request, 'predict.html',{"result2":result1})