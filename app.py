from flask import Flask, request, jsonify
import joblib
import requests
from bs4 import BeautifulSoup
from requests.exceptions import SSLError
from BayesianFilter import BayesianFilter

app = Flask(__name__)

# 홈 경로
@app.route("/", methods=['GET'])
def home():
    return "Prediction API", 200

# 모델 예측 경로
@app.route("/model", methods=['POST'])
def model():
    if request.method == 'POST':
        data = request.get_json()  
        title = data.get('title')

        title_input = title

        model_data = joblib.load("model_and_mapping_1125.pkl")  # 모델 로드
        clf = model_data['bayesian_filter']
        keyword_to_category = model_data['keyword_to_category']

        prediction, _ = clf.predict(title_input)  # 모델에 제목 입력 후 예측

        result = {
            'prediction': prediction,  # 예측 결과
            'title': title_input        # 가져온 제목
        }

        return jsonify(result), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
