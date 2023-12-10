from flask import Flask, render_template, request
import keras
import numpy as np
from sklearn import datasets

main = Flask(__name__)

iris = datasets.load_iris()

model = keras.models.load_model('models/model.h5')


@main.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':

        parameters = [request.form['sepal-length'], request.form['sepal-width'], request.form['petal-length'], request.form['petal-width']]

        for i in range(len(parameters)):
            try:
                parameters[i] = float((parameters[i]).replace(',', '.'))
            except:
                parameters[i] = 'значение не указано'

        try:
            values = model.predict([parameters])
            answer = iris.target_names[np.argmax(values)]
            return render_template('index.html', name=answer, val_array=parameters)
        except:
            answer = 'Значения не указаны'
            return render_template('index.html', name=answer, val_array=parameters)

    return render_template('index.html')


if __name__ == "__main__":
    main.run(debug=False)
