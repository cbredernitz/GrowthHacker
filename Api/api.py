from flask import Flask, render_template, request
from sklearn.externals import joblib
import numpy as np
import math 
app = Flask(__name__)

@app.route('/')
def marketing():
    return render_template('customer.html')

@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        mdl = joblib.load('modl_GBT_num.joblib')
        result_form = request.form

        # Returns a list we can pass into a model
        result_list = list(result_form.values()) 
        result = mdl.predict(np.asarray(result_list).reshape(1,-1))

        # Log Inverse Result and convert to millions
        result = np.exp(result)*1000000
        result = str(round(result, 2))

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)