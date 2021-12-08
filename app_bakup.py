from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

model = joblib.load("models/model.joblib")


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/documentation")
def more_love():
    things_i_love = [
        "Star Wars",
        "Coffee",
        "Cookies",
    ]
    return render_template("documentation.html", things_i_love=things_i_love)


# @app.route("/documentation")
# def prediction():
#     things_i_love = [
#         "Star Wars",
#         "Coffee",
#         "Cookies",
#     ]
#     return render_template("prediction.html", things_i_love=things_i_love)


@app.route("/predict", methods=["POST"])
def predict():
    # Check if request has a JSON content
    if request.json:
        # Get the JSON as dictionnary
        data = request.get_json()
        # Check mandatory key
        if "input" in data.keys():
            # Load model
            # Predict
            input = np.array(data['input'])
            # print(input)
            # prediction = model.predict([data["input"]])

            prediction = model.predict(input)

            # Return the result as JSON but first we need to transform the
            # result so as to be serializable by jsonify()
            # prediction = str(prediction[0])

            # result = {'result': prediction.tolist()}
            result = {'result': str(prediction)}

            # return jsonify({"predict": prediction}), 200

            return jsonify(result), 200
            # return input
    return jsonify({"msg": "Error: not a JSON or no email key in your request"})


if __name__ == "__main__":
    app.run(debug=True)
