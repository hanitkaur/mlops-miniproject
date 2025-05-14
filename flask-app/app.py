from flask import Flask,render_template,request
import mlflow
from mlflow import MlflowClient
from preprocessing_utility import normalize_text
import dagshub
import pickle
dagshub.init(repo_owner='hanitkaur', repo_name='mlops-miniproject', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/hanitkaur/mlops-miniproject.mlflow')

app= Flask(__name__)

#load model
def get_latest_model_version(model_name):
    client=MlflowClient()
    latest_version=client.get_latest_versions(model_name,stages=["Production"])
    if not latest_version:
        client.get_latest_versions(model_name,stages=["None"])
    return latest_version[0].version if latest_version else None

model_name="my_model" 
model_version=get_latest_model_version(model_name)
model_uri=f'models:/{model_name}/{model_version}'
model= mlflow.pyfunc.load_model(model_uri)
vectorizer= pickle.load(open('models/vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html',result=None)


@app.route('/predict',methods=['POST'])
def predict():
    text=request.form['text']
    text=normalize_text(text)
    features=vectorizer.transform([text])
    result=model.predict(features)

    return render_template('index.html',result=result[0])


app.run(debug=True)