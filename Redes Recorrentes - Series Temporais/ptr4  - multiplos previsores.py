from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, LSTM 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint 


# ==== Estrutura dos dados ====
base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()
base_treinamento = base.iloc[:, 1:7].values #Dessa vez vamos pegar 6 previsores

normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

normalizador_previsao = MinMaxScaler(feature_range=(0,1))
normalizador_previsao.fit_transform(base_treinamento[:,0:1])

previsores = []
preco_real = []
for i in range(90, 1242):
    previsores.append(base_treinamento_normalizada[i-90:i, 0:6]) #6 previsores
    preco_real.append(base_treinamento_normalizada[i, 0])
previsores, preco_real = np.array(previsores), np.array(preco_real)
#previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1)) Nao precisamos mais pq ja saiu  em Numpy


# ==== Estrutura da rede convolucinal ====
regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 6))) #6 previsores de entrada
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1, activation = 'sigmoid')) #Como os dados estao entre 0 e 1, para testar, podemos utilizar a sigmoid, ja que faremos a 'desnormalizaçao'

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error']) #Testando o adam ao inves do RMSdrop

# ==== callbacks ====

'''EarlyStopping é uma funcao callbacks que é util para fazer pausas em nossa rede em determinadas situaçoes que desejarmos e olhar os aspectos da rede
1- monitor: damos a funçao que queremos monitorar
2- min_deta: definimos o minimo de alteraçao na loss
3- patience: numero de epocas que devemos esperar antes de parar 
verbose serve so para mostrar algumas mensagens na tela'''

es = EarlyStopping(monitor = 'loss', min_delta = 1e-10, patience = 10, verbose = 1)  

rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.2, patience = 5, verbose = 1) #Diminui a taxa de aprendizagem quando uma metrica para de melhorar (no caso a loss), factor damos o tanto que a taxa diminuira

mcp = ModelCheckpoint(filepath = 'pesos.h5', monitor = 'loss', 
                      save_best_only = True, verbose = 1) #Salva os melhores pesos a cada epoca em um arquivo para que possamos utiliza-los posteriormente

regressor.fit(previsores, preco_real, epochs = 30, batch_size = 32,
              callbacks = [es, rlr, mcp])


# ==== Estrutura dos dados de teste ====
base_teste = pd.read_csv('petr4_teste.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values
frames = [base, base_teste] #Como temos mais de um previsor temos fazer esse processo que seria a lista entre esses dois frames
base_completa = pd.concat(frames)
base_completa = base_completa.drop('Date', axis = 1) #Dropamos a coluna date

entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = normalizador.transform(entradas) #Temos que criar um normalizador diferente

X_teste = []
for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0:6])
X_teste = np.array(X_teste)

previsoes = regressor.predict(X_teste)
previsoes = normalizador_previsao.inverse_transform(previsoes) #Fazendo a desnormalizaçao

previsoes.mean()
preco_real_teste.mean()

plt.plot(preco_real_teste, color = 'red', label = 'Preço real')
plt.plot(previsoes, color = 'blue', label = 'Previsoes')
plt.title('Previsao preço das açoes')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()