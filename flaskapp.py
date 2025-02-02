from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
import json

app = Flask(__name__)

# Sample training data (height, weight, age, activity_level) and corresponding valve sizes
training_data = np.array([
    [5, 148, 45, 1],
    [6, 160, 50, 1],
    [5.5, 155, 40, 0],
    [5.8, 150, 35, 1],
    [5.2, 140, 60, 0]
])
valve_sizes = np.array([2.5, 3.0, 2.8, 3.2, 2.0])

# Train a linear regression model
model = LinearRegression()
model.fit(training_data, valve_sizes)


def predict_valve_size(height, weight, age, activity_level):
    features = np.array([[height, weight, age, activity_level]])
    size = model.predict(features)[0]
    return size


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            height = float(request.form['height'])
            weight = float(request.form['weight'])
            age = int(request.form['age'])
            activity_level = int(request.form['activity_level'])

            size_prediction = predict_valve_size(height, weight, age, activity_level)

            # Store data
            data = {
                "height": height,
                "weight": weight,
                "age": age,
                "activity_level": activity_level,
                "predicted_size": size_prediction
            }

            with open("valve_data.json", "w") as json_file:
                json.dump(data, json_file, indent=4)

            return render_template('index.html', prediction=size_prediction, data=data)

        except ValueError:
            return render_template('index.html', error="Please enter valid numeric values.")

    return render_template('index.html')


@app.route('/data', methods=['GET'])
def show_data():
    try:
        with open("valve_data.json", "r") as json_file:
            data = json.load(json_file)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"error": "No data file found."}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Error reading the data file."}), 500


if __name__ == '__main__':
    app.run(debug=True)
