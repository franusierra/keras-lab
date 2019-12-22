from keras.models import model_from_json
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

FILENAME = 'pima-indians-diabetes.csv'
#Cargamos el dataset
data = pd.read_csv(FILENAME, header=None, names=['n_emb','conc_gluc','pres_art','piel_tric','2h_ins','imc','pedi_fun','edad','SALIDA']) 

D=pd.get_dummies(data)

#Aplicamos la normalizacion scaling para tener los datos con valores
#entre 0 y 1
normD=(D-D.min())/(D.max()-D.min())


#Separamos el dataset en conjunto de entrenamiento y de test
#80 % Training ..... 20% Test
train, test = train_test_split(normD, test_size=0.2)

#Conjunto de entrenamiento
Xtrain=train.loc[:,'n_emb':'edad']
Ytrain=train.loc[:,'SALIDA']

#Conjunto de test
Xtest=test.loc[:,'n_emb':'edad']
Ytest=test.loc[:,'SALIDA']

#Cogemos las 5 primeras filas del conjunto de test para hacer
#las predicciones
predictX=Xtest.head()
predictY=Ytest.head()

#Creamos el modelo con tres capas Dense
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Compilamos el modelo de la red neuronal (Optimizador Adam)
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

#Entrenamos el modelo
history = model.fit(Xtrain, Ytrain, epochs=200, batch_size=16, shuffle = False, validation_split=0.2)

#Evaluamos el modelo
_, accuracy = model.evaluate(Xtest, Ytest)

#Mostramos la precision en porcentaje
print('Accuracy: %.2f' % (accuracy*100))

path="Models/"
# Serializamos el modelo a formato JSON
model_json = model.to_json()
with open(f"{path}model_adam.json", "w") as json_file:
    json_file.write(model_json)
# Serializamos los pesos a HDF5
model.save_weights(f"{path}model_adam.h5")
print("Saved model to disk")

# Cargamos el JSON y creamos el modelo a partir del JSON
json_file = open(f"{path}model_adam.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Cargamos los pesos en el nuevo modelo
loaded_model.load_weights(f"{path}model_adam.h5")

print("Loaded model from disk")

#Realizamos las predicciones
predictions = loaded_model.predict(predictX)

#Mostramos los predicciones realizadas a traves de la red neuronal
#y mostramos algunas gráficas

print('Realizando predicciones')
for data,predicted,expected in zip(predictX.iterrows(),predictions,predictY):
    print('==================================')
    print('Datos de entrada:')
    print('{}'.format(list(data)))
    print('Predicción:')
    print('{} (expected {})'.format(predicted,expected))
    print('==================================')

acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)
fig, axs = plt.subplots(2)
fig.tight_layout() 
axs[0].plot(epochs, acc, 'b', label='Training acc')
axs[0].set_title(f'Precision de entrenamiento - Optimizador Adam')
axs[0].legend()

axs[1].plot(epochs, loss, 'b', label='Training loss')
axs[1].set_title(f'Suma de las perdidas - Optimizador Adam')
axs[1].legend()

plt.show()
fig.clear()
plt.close(fig)