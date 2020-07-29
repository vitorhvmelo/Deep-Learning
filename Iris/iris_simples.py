import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder #IMPORTA A BIBLIOTECA DO ENCODER
from keras.utils import np_utils #IMPORTA A BIBLIOTECA PARA COMPLEMENTAR O ENCODER
import numpy as np
from sklearn.metrics import confusion_matrix

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelEncoder = LabelEncoder() #CRIA O OBJETO DO ENCODER
classe = labelEncoder.fit_transform(classe) #TRANSFORMA AS CLASSES DAS PLANTAS EM VALORES NUMERICOS
classe_dummy = np_utils.to_categorical(classe)
"""
iris setosa 1 0 0
iris virginica 0 1 0
iris versicolor 0 0 1
"""
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, 
                                                                                              classe_dummy, 
                                                                                              test_size=0.25)
classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dense(units = 3, activation = 'softmax'))
classificador.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 1000)

resultado = classificador.evaluate(previsores_teste, classe_teste)
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

classe_teste2 = [np.argmax(t) for t in classe_teste] #Pega o codigo da planta e transforma ela pelos indices 0 1 e 2
previsoes2 = [np.argmax(t) for t in previsoes]
matriz = confusion_matrix(previsoes2, classe_teste2)  