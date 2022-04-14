from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.classification import *
# import pycaret
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

data = pd.read_csv('https://raw.githubusercontent.com/maryclarecc/mentalhealthML/main/data.csv')
regr = setup(data = data, target = 'dx_dep1', session_id=123,
                  normalize = True, 
                  silent = True,
                  transformation = True, 
                  ignore_low_variance = True,
                  feature_selection = True,
                  feature_selection_threshold = 0.8,
                  bin_numeric_features = ["hours_work_paid", "age", "bmi"],
                  high_cardinality_features = ["residenc", "timeclass", "BRS_1", "BRS_3", "BRS_5", "belong8", "educ_par2"],
                  remove_multicollinearity = True, multicollinearity_threshold = 0.9) 

# model = "log_regr_pipeline.pkl"
# print(model)

# model = pickle.load(open("log_regr_pipeline.pkl","rb"))
# model = load_model('log_regr_pipeline_v2')
model = load_model('log_regr_pipeline_v2')
# columns need to be original column names at setup - not transformed dataset
cols = ["hours_work_paid", "age", "bmi", "residenc", "timeclass", "BRS_1", "BRS_3", "BRS_5", "belong8", "educ_par2"]
# prediction = "Test"


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/detecting')
def mentalhealth():    
    return render_template("detecting.html")


@app.route('/resources')
def resources():
    return render_template("resources.html")

# prediction function
# def ValuePredictor(to_predict_list):
#     to_predict = np.array(to_predict_list).reshape(1,2)
#     loaded_model = pickle.load(open("model.pkl","rb")) # load the model
#     result = loaded_model.predict(to_predict) # predict the values using loded model
#     return result[0]

# @app.route('/predict', methods = ["POST"])


@app.route('/results.html', methods = ['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    # int_features = ["4", "22", "22", "1", "1","1","1","1","1","1"]

    # l = len(int_features)
    # print(l)

    final = np.array(int_features)

    # prediction = model.predict(final)
    # prediction = int_features[0]  # Uncomment to show first feature in output - for test purposes only
    # prediction = "Test"


    data_unseen = pd.DataFrame([final], columns = cols)
    # prediction = data_unseen[0,0]
    # print(data_unseen)
    prediction = predict_model(model, data=data_unseen, round = 0)
    prediction = int(prediction.Score[0])

    if int(prediction)== 1:
        result ='You are at risk for depression - please see resource page'
    else:
        result ='You are not at risk for depression'           



    # return render_template('/predict.html', pred = 'Expected Diagnosis: {}'.format(prediction))
    # return render_template('/detecting.html',prediction)
    return render_template('/results.html', prediction_text = result)
    # return render_template('/detecting.html')

# @flask_app.route("/predict", methods = ["POST"])
# def predict():
#     float_features = [float(x) for x in request.form.values()]
#     features = [np.array(float_features)]
#     prediction = model.predict(features)
#     return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))



if __name__ == "__main__":
    app.run()
