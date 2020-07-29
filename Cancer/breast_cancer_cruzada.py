import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim=30)) #primeira camada oculta
    classificador.add(Dropout(0.2)) #Dropout ajuda a diminuir o overfiting
    classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform')) #segunda camada oculta
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 1, activation = 'sigmoid')) #camada de saida
    classificador.compile('adam', loss='binary_crossentropy', metrics = ['binary_accuracy'])    
    return classificador

classificador = KerasClassifier(build_fn = criarRede, epochs = 100, batch_size = 10)
resultados = cross_val_score(estimator = classificador, X = previsores, y =classe, cv = 10, scoring = 'accuracy')

media = resultados.mean()
dvpadrao = resultados.std()