from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

dados = pd.read_csv('personagens.csv')

previsores = dados.iloc[:, 0:6].values
classe = dados.iloc[:, 6].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

classificador = Sequential()

classificador.add(Dense(units = 4, activation = 'relu', input_dim = 6))
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                      metrics = ['binary_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, 
                  epochs = 2000)
resultado = classificador.evaluate(previsores_teste, classe_teste)