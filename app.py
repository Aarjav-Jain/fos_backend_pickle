import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
modelML = pickle.load(open('model_ml.pkl','rb'))
modelSC = pickle.load(open('model_safety_chart.pkl','rb'))

def makePredictionML(data):
    newData = np.array(data)
    prediction = modelML.predict(newData.reshape(1,11))
    return(prediction[0])
    # return(prediction[0][0])

def makePredictionSafetyChart(data):
    newData = np.array(data)
    prediction = modelSC.predict(newData.reshape(1,3))
    # print(prediction)
    return (prediction[0][0])

@app.route("/models/machine_learning",methods=['POST'])
def MLResult():
    if(request.method == 'POST'):
        data = request.get_json(force=True)
        # print(data)
        predictedValue = makePredictionML(data)
        return str(predictedValue)
        # return 'machine learninig'
    
@app.route("/models/safety_chart",methods=['POST'])
def SafetyChartResult():
    if(request.method == 'POST'):
        data = request.get_json(force=True)
        # print(data)
        predictedValue = makePredictionSafetyChart(data)
        return str(predictedValue)
        # return 'data recieved'
        # return 'safety chart'

@app.route('/test')
def test():
    return jsonify('test successful')

if __name__ == '__main__':
    app.run(debug=True)