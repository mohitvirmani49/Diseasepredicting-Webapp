from flask import jsonify
import requests
import pickle
import numpy as np
import sys
import os
import re
import sklearn
from flask import Flask, render_template, url_for, flash, redirect, request, send_from_directory
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras import backend
from tensorflow.keras import backend
import tensorflow as tf
global graph
graph=tf.compat.v1.get_default_graph()
from skimage.transform import resize
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


app = Flask(__name__)

model_heartdisease = pickle.load(open('heartdisease.pkl', 'rb'))
model_liverdisease = pickle.load(open('liverdisease.pkl', 'rb'))
model_cancer = pickle.load(open('breastcancer.pkl', 'rb'))
model_diabetes = pickle.load(open('diabetes.pkl', 'rb'))
model_kidney = pickle.load(open('kidneydisease.pkl', 'rb'))





@app.route('/',methods=['GET'])
@app.route('/home',methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/about',methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/heartdisease', methods=['GET','POST'])
def heartdisease():
    if request.method == 'POST':
        Age=int(request.form['age'])
        Gender=int(request.form['sex'])
        ChestPain= int(request.form['cp'])
        BloodPressure= int(request.form['trestbps'])
        ElectrocardiographicResults= int(request.form['restecg'])
        MaxHeartRate= int(request.form['thalach'])
        ExerciseInducedAngina= int(request.form['exang'])
        STdepression= float(request.form['oldpeak'])
        ExercisePeakSlope= int(request.form['slope'])
        MajorVesselsNo= int(request.form['ca'])
        Thalassemia=int(request.form['thal'])
        prediction=model_heartdisease.predict([[Age, Gender, ChestPain, BloodPressure, ElectrocardiographicResults, MaxHeartRate, ExerciseInducedAngina, STdepression, ExercisePeakSlope, MajorVesselsNo, Thalassemia]])
        if prediction==1:
            return render_template('heartdisease.html', prediction_text="Oops! The person seems to have Heart Disease.", title='Heart Disease')
        else:
            return render_template('heartdisease.html', prediction_text="Great! The person does not have any Heart Disease.", title='Heart Disease')
    else:
        return render_template('heartdisease.html', title='Heart Disease')

    
@app.route('/liverdisease', methods=['GET','POST'])
def liverdisease():
    if request.method == 'POST':
        Age=int(request.form['Age'])
        Gender=int(request.form['Gender'])
        Total_Bilirubin= float(request.form['Total_Bilirubin'])
        Direct_Bilirubin= float(request.form['Direct_Bilirubin'])
        Alkaline_Phosphotase= int(request.form['Alkaline_Phosphotase'])
        Alamine_Aminotransferase= int(request.form['Alamine_Aminotransferase'])
        Aspartate_Aminotransferase= int(request.form['Aspartate_Aminotransferase'])
        Total_Protiens= float(request.form['Total_Protiens'])
        Albumin= float(request.form['Albumin'])
        Albumin_and_Globulin_Ratio= float(request.form['Albumin_and_Globulin_Ratio'])
        prediction=model_liverdisease.predict([[Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]])
        if prediction==1:
            return render_template('liverdisease.html', prediction_text="Oops! The person seems to have Liver Disease.", title='Liver Disease')
        else:
            return render_template('liverdisease.html', prediction_text="Great! The person does not have any Liver Disease.", title='Liver Disease')
    else:
        return render_template('liverdisease.html', title='Liver Disease')

@app.route('/breastcancer', methods=['GET','POST'])
def breastcancer():
    if request.method == 'POST':
        texture_mean = float(request.form['texture_mean'])
        perimeter_mean = float(request.form['perimeter_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        compactness_mean = float(request.form['compactness_mean'])
        concavity_mean = float(request.form['concavity_mean'])
        concave_points_mean = float(request.form['concave_points_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        radius_se = float(request.form['radius_se'])
        compactness_se = float(request.form['compactness_se'])
        concavity_se = float(request.form['concavity_se'])
        concave_points_se = float(request.form['concave_points_se'])
        texture_worst = float(request.form['texture_worst'])
        smoothness_worst = float(request.form['smoothness_worst'])
        compactness_worst = float(request.form['compactness_worst'])
        concavity_worst = float(request.form['concavity_worst'])
        concave_points_worst = float(request.form['concave_points_worst'])
        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst = float(request.form['fractal_dimension_worst'])
        prediction=model_cancer.predict([[texture_mean, perimeter_mean, smoothness_mean, compactness_mean,
           concavity_mean, concave_points_mean, symmetry_mean, radius_se,
           compactness_se, concavity_se, concave_points_se, texture_worst,
           smoothness_worst, compactness_worst, concavity_worst,
           concave_points_worst, symmetry_worst, fractal_dimension_worst]])
        if prediction==1:
            return render_template('cancer.html', prediction_text="Oops! The tumor is malignant.", title='Breast Cancer')
        else:
            return render_template('cancer.html', prediction_text="Great! The tumor is benign.", title='Breast Cancer')
    else:
        return render_template('cancer.html',title='Breast Cancer')
    
@app.route('/diabetes', methods=['GET','POST'])
def diabetes():
    if request.method == 'POST':
        Pregnancies=int(request.form['Pregnancies'])
        Glucose=int(request.form['Glucose'])
        BloodPressure=int(request.form['BloodPressure'])
        SkinThickness=int(request.form['SkinThickness'])
        Insulin=int(request.form['Insulin'])
        BMI=float(request.form['BMI'])
        DiabetesPedigreeFunction=float(request.form['DiabetesPedigreeFunction'])
        Age=int(request.form['Age'])
        prediction=model_diabetes.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        if prediction==1:
            return render_template('diabetes.html', prediction_text="You seem to be suffering from Diabetes.", title='Diabetes')
        else:
            return render_template('diabetes.html', prediction_text="Great!You are diabetes free.", title='Diabetes')
    else:
        return render_template('diabetes.html',title='Diabetes')
    
@app.route('/kidneydisease', methods=['GET','POST'])
def kidneydisease():
    if request.method == 'POST':
        age=float(request.form['age'])
        bp=float(request.form['bp'])
        al=float(request.form['al'])
        pcc=float(request.form['pcc'])
        bgr=float(request.form['bgr'])
        bu=float(request.form['bu'])
        sc=float(request.form['sc'])
        hemo=float(request.form['hemo'])
        pcv=int(request.form['pcv'])
        htn=float(request.form['htn'])
        dm=int(request.form['dm'])
        appet=float(request.form['appet'])
        prediction=model_kidney.predict([[age,bp,al,pcc,bgr,bu,sc,hemo,pcv,htn,dm,appet]])
        if prediction==1.0:
            return render_template('kidney.html', prediction_text="OOPS! The person seems to have Kidney Disease.", title='Kidney Disease')
        else:
            return render_template('kidney.html', prediction_text="Great!You are not suffering from Kidney disease.", title='Kidney Disease')
    else:
        return render_template('kidney.html',title='Kidney Disease')    
 
    
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename)

if __name__=='__main__':
	app.run(debug=True)

