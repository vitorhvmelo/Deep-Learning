import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score #importa a matriz confusao e a matriz de acerto

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

classificador = Sequential()
classificador.add(Dense(units = 16, activation = 'relu',
                        kernel_initializer = 'random_uniform', input_dim=30)) #primeira camada oculta
classificador.add(Dense(units = 16, activation = 'relu',
                        kernel_initializer = 'random_uniform')) #segunda camada oculta
classificador.add(Dense(units = 1, activation = 'sigmoid')) #camada de saida
classificador.compile('adam', loss='binary_crossentropy', metrics = ['binary_accuracy']) #optimizer padrao
            #otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5) #mudando parametros do optimizer => abaixou muito a precisao
            #classificador.compile(otimizador, loss='binary_crossentropy', metrics = ['binary_accuracy']) #usando o optimizer configurado
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)

pesos0 = classificador.layers[0].get_weights() #mostra os pesos que a IA encontrou, o 0 indica a primeira camada
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()
print(pesos0)
print(len(pesos0))

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

precisao = accuracy_score(classe_teste, previsoes) #essa matriz mostra a precisao da rede
matriz = confusion_matrix(classe_teste, previsoes) #essa matriz mostra quantos a rede errou ou acertou
resultado = classificador.evaluate(previsores_teste, classe_teste) #outra maneira de achar a precisao, mas agr usando o keras