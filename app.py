import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
from flask import Flask
import os
app = Flask(__name__)
# model = pickle.load(open("model.pkl", "rb"))


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width= float(request.form['petal_width'])

        val = np.array([sepal_length, sepal_width, petal_length, petal_width])

        final_features = [np.array(val)]
        model_path = os.path.join('models', 'modelNB.pkl')
        model = pickle.load(open(model_path, 'rb'))
        res = model.predict(final_features)

        return render_template('index.html', prediction_text=res)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
