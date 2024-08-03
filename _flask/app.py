from flask import Flask,request,jsonify
import pickle
import numpy as np

model = pickle.load(open('_flask\model.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello World!'

@app.route('/predict',methods=['POST'])
def predict():
    Age=request.form.get('Age')
    EstimatedSalary=request.form.get('EstimatedSalary')
    EstimatedSalary = float(EstimatedSalary)
    Age = float(Age)
     
    
    input_query=np.array([[Age,EstimatedSalary]])
    result=model.predict(input_query)[0]

    if(result==1):
        return jsonify({'purchased':int(result)})
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)