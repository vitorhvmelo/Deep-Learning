import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

classificador = Sequential()
classificador.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'normal', input_dim=30)) #primeira camada oculta
classificador.add(Dense(units = 8, activation = 'relu', 
                        kernel_initializer = 'normal')) #segunda camada oculta
classificador.add(Dense(units = 1, activation = 'sigmoid')) #camada de saida
classificador.compile('adam', loss='binary_crossentropy', metrics = ['binary_accuracy']) 
classificador.fit(previsores, classe, batch_size = 10, epochs = 100)

classificador_json = classificador.to_json()
with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_breast.h5')