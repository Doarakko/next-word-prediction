
import json

from flask import Flask, jsonify, render_template, request

import predict

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict-next-words', methods=["POST"])
def predict_next_words():
    body = request.json['body']

    language = request.json['language']
    res = predict.predict_next_words(body, language)

    return jsonify({'nextWords': res})


@ app.route('/predict-language', methods=["POST"])
def predict_language():
    body = request.json['body']
    res = predict.predict_language(body)

    return jsonify({'language': res})


if __name__ == '__main__':
    app.run()
