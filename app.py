from flask import Flask, jsonify, request
from src.treeRepository import *

app = Flask(__name__)


@app.route('/', methods=['GET'])
def base():  # put application's code here
    return "saas api"

@app.route('/export', methods=['POST'])
def export():  # put application's code here
    return export_model()


@app.route('/model/information', methods=['GET'])
def get_information():  # put application's code here
    return show_model_information()


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    return predict_model(data)


if __name__ == '__main__':
    app.run()
