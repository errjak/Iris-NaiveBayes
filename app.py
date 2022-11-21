import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
from flask import Flask
import os
app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        PMu = float(request.form['PM2.5'])
        PMs = float(request.form['PM10'])
        No = float(request.form['NO'])
        N02 = float(request.form['NO2'])
        NOx = float(request.form['NOx'])
        Nh3 = float(request.form['NH3'])
        AqI = float(request.form['AQI'])


        val = np.array([PMu, PMs, No, N02, NOx, Nh3, AqI])

        final_features = [np.array(val)]
        model_path = os.path.join('models', 'modelDelhi.pkl')
        model = pickle.load(open(model_path, 'rb'))
        res = model.predict(final_features)

        return render_template('index.html', prediction_text=res)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
