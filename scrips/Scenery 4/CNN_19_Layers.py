from LoadData import loadData, loadDataTest
from GeneratorConfusionMatrix import plotConfusionMatrix
import Functions
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import time
from keras import backend as K
from SplitData import splitData
K.set_image_data_format('channels_last')

inicio = time.time()

# Configuração do Modelo
batch_size = 32
epochs = 100

# Separa as imagens de teste
splitData(["cercosporiose","ferrugem", "mancha", "saudavel"], 15)

#Carrega o BD das imgagens (Teste)
inicio_aux = time.time()
(X_test, y_test) = loadDataTest()
fim = time.time()
Functions.printTime("Load Dataset Test", inicio_aux, fim)

#Carrega o BD das imagens (Treino e Validação) 
inicio_aux = time.time()
(X, y) = loadData()
fim = time.time()
Functions.printTime("Load Dataset Treino", inicio_aux, fim)

#Redimensiona os dados para ficar no formato que o tensorflow trabalha
X = X.astype('float32') 
X_test = X_test.astype('float32')

#Normalizando os valores de 0-255 to 0.0-1.0
X /= 255.0
X_test /= 255.0

#Divide os dados em 82% para treino e 18% para teste **Os dados de testes já foram separados
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.17, train_size=0.83, stratify=y)


#Transformando os rótulos de decimal para vetores com valores binários
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)

#Criação do Modelo
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(150, 150, 3), activation="relu", padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (3, 3), activation="relu", padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (3, 3), activation="relu", padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, (3, 3), activation="relu", padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(256, (3, 3), activation="relu", padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(512, (3, 3), activation="relu", padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(1024, (3, 3), activation="relu", padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

# Executa o treinamento do modelo
inicio_aux = time.time()
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
fim = time.time()
Functions.printTime("Training", inicio_aux, fim)

# Plota o histórico da acurácia 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Learning curve')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig('curve_accuracy.png', dpi=300)
plt.show()

plotConfusionMatrix(model, X_test, y_test)

#Transformando os rótulos dos dados de teste
#de decimal para vetores com valores binários
y_test = np_utils.to_categorical(y_test)

# Mostra a potuação da acurácia
scores = model.evaluate(X_test, y_test, verbose=0)
result_error = str("%.2f"%(1-scores[1]))
result = str("%.2f"%(scores[1]))
print("CNN Score:", result)
print("CNN Error:", result_error)

# Salva o modelo no formato JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Salva os pesos em HDF5
model.save_weights("model_w.h5")
print("Modelo salvo no disco")

# Salva os resultados da acurácia em arquivo CSV
index = []
for i in range(1, epochs+1):
    index.append(f'epoca{i}')
result_train = pd.DataFrame(history.history['accuracy'], index=index)
result_test = pd.DataFrame(history.history['val_accuracy'], index=index)
result_train.to_csv('accuracy_train.csv', header=False)
result_test.to_csv('accuracy_test.csv', header=False)

fim = time.time()
Functions.printTime("Time Run Model", inicio, fim)