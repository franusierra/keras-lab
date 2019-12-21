import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras import optimizers
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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


optimizadoresText=[
    "SGD",
    "RMSprop",
    "ADAM"
]
if __name__ == "__main__":
    
    data = pd.read_csv('pima-indians-diabetes.csv', header=None, names=['n_emb','conc_gluc','pres_art','piel_tric','2h_ins','imc','pedi_fun','edad','SALIDA'])
    D=pd.get_dummies(data)
    train, test = train_test_split(data, test_size=0.2)
    Xtrain=train.loc[:,'n_emb':'edad']
    Ytrain=train.loc[:,'SALIDA']
    Xtest=test.loc[:,'n_emb':'edad']
    Ytest=test.loc[:,'SALIDA']
    predictX=Xtest.head(5)
    predictY=Ytest.head(5)
    normD=(D-D.min())/(D.max()-D.min())
    normD.head()
    plt.matshow(normD.corr())
    plt.xticks(range(len(normD.columns)), normD.columns)
    plt.yticks(range(len(normD.columns)), normD.columns)
    plt.colorbar()
    plt.show()
    
    generate_models()
    
    for i,model in enumerate(models):
        history=model.fit(Xtrain, Ytrain, epochs=150, batch_size=10, verbose=1 )

        _, accuracy = model.evaluate(Xtest,Ytest)
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
        predictions = loaded_model.predict(predictX)
        print('Realizando predicciones')
        for data,predicted,expected in zip(predictX.iterrows(),predictions,predictY):
           print('==================================')
           print('Datos de entrada:')
           print('{}'.format(list(data)))
           print('Predicción:')
           print('{} (expected {})'.format(predicted,expected))
           print('==================================')
        print(history.history.keys())
        acc = history.history['accuracy']
        loss = history.history['loss']
        epochs = range(1, len(acc) + 1)
        fig, axs = plt.subplots(2)
        fig.tight_layout() 
        axs[0].plot(epochs, acc, 'b', label='Training acc')
        axs[0].set_title(f'Precision de entrenamiento - Optimizador {optimizadoresText[i]}')
        axs[0].legend()
    
        axs[1].plot(epochs, loss, 'b', label='Training loss')
        axs[1].set_title(f'Suma de las perdidas - Optimizador {optimizadoresText[i]}')
        axs[1].legend()
    
        plt.show()
    fig.clear()
    plt.close(fig)

