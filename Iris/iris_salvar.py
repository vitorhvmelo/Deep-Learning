import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, 
                                                                                              classe_dummy, 
                                                                                              test_size=0.25)
classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dense(units = 3, activation = 'softmax'))
classificador.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 1000)


classificador_json = classificador.to_json()
with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_breast.h5')

                        
