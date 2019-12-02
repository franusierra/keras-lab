import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras import optimizers

models = []


def generate_models():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optim=optimizers.SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
    models.append(model)

    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optim=optimizers.RMSprop(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

    models.append(model)

    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optim=optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

    models.append(model)

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

trainingData={
    "x_fit" : dataset[0:500,0:8],
    "y_fit" : dataset[0:500,8],
    "x_evaluate" : dataset[500:-68,0:8],
    "y_evaluate" : dataset[500:-68,8],
    "x_predict" : dataset[-68:,0:8],
    "y_predict" : dataset[-68:,8]
}
if __name__ == "__main__":
    generate_models()

    for i,model in enumerate(models):
        model.fit(trainingData["x_fit"], trainingData["y_fit"], epochs=10, batch_size=10, verbose=1)

        _, accuracy = model.evaluate(trainingData["x_evaluate"], trainingData["y_evaluate"])
        print('Accuracy: %.2f' % (accuracy*100))

        path="Models/"
        # serialize model to JSON
        model_json = model.to_json()
        with open(f"{path}model{i}.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(f"{path}model{i}.h5")
        print("Saved model to disk")


        # load json and create model
        json_file = open(f"{path}model{i}.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(f"{path}model{i}.h5")
        print("Loaded model from disk")
        
        predictions = loaded_model.predict_classes(trainingData["x_predict"])

        for j in range(5):
            print('%s => %d (expected %d)' % (trainingData["x_predict"][j].tolist(), predictions[j], trainingData["y_predict"][j]))

