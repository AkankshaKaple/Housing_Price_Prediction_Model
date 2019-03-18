from flask import request
import pickle
import math
import locale
from flask import Flask, render_template, jsonify
# from forms import RegistrationForm, LoginForm

app = Flask(__name__)

app.config["SECRET_KEY"] = 'c8faaf33aaa751c1270e70af5c21fa75'

@app.route('/', methods=['GET'])
def index():
    return render_template("data.html")


model = pickle.load(open('model_housing_data.pkl','rb'))

@app.route('/result', methods=['POST'])
def predict():
    bathrooms = float(request.form['bathrooms'])
    bedrooms = float(request.form['bedrooms'])
    area = float(request.form['area'])
    output = math.floor(model.predict([[bathrooms, bedrooms, area]])[0][0])

    locale.setlocale(locale.LC_ALL, '')
    output = locale.format("%d", output, grouping=True)
    return "Predicted Price for " + str(bathrooms) + " bathrooms " + str(bedrooms) + " bedrooms " \
           + str(area) + " area " + " is:-  rupees " + str(output)




if __name__ == '__main__':
    app.run(debug=True)