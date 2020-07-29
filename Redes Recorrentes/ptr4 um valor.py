from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, LSTM 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#==== Estrutura dos dados ====
base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna() #Dropa os NA

base_treinamento = base.iloc[:, 1:2].values # Seleciona todas as linhas da primeira coluna
normalizador = MinMaxScaler(feature_range=(0,1)) #Contruimos o normalizador com funçao que normaliza o preço das açoes entre 0 e 1
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento) #Aplicamos o normalizador na base de treinamento

'''Aqui é a parte mais importante de series temporais. Para fazer a previsão de um preço real, precisamos de uma certa quantidade de dados antes dele, nesse exemplo utilizaremos 90 dados antes do preço real analisado. Entao criamos um for que pegara fara uma lista de listas, com essa ultima tendo lotes de 90 dias, sendo que os lotes tem um dia de diferença. Tambem sera criada uma lista com os preços reais a partir do 90 ate o 1242'''

previsores = []
preco_real = []
for i in range(90, 1242):
    previsores.append(base_treinamento_normalizada[i-90:i, 0])
    preco_real.append(base_treinamento_normalizada[i, 0])
previsores, preco_real = np.array(previsores), np.array(preco_real)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1)) #Tanto de registros(1152), tanto de registros por lote, somente um atributo previsor

#==== Estrutura da rede Recorrente ====
regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 1))) #Usamos LSTM que é um tipo de camada de recorrencia
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50)) #Colocamos return_sequences =True quando vamos passar os dados de uma camada recorrente para outra recorrente, no caso a proxima é densa
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1, activation = 'linear')) #Camada de saida com funçao linear, nao queremos fazer nenhuma transformaçao na saida

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error',
                  metrics = ['mean_absolute_error']) #RMSprop é o optimizer mais recomendado para Redes Recorrentes
regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32)

# ==== Estrutura dos dados de teste ====
base_teste = pd.read_csv('petr4_teste.csv')
preco_real_teste = base_teste.iloc[:, 1:2].values #Pegamos somente a primeira coluna
base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0) #Aqui concatenamos a base teste com a de treinamento
entradas = base_completa[len(base_completa) - len(base_teste) - 90 : ].values #Precisamos somente da base de teste e os 90 dias anteriores ao primeiro dia de teste
entradas = entradas.reshape(-1, 1) #Pra ficar no formato numpy, que é exigido no normalizador
entradas = normalizador.transform(entradas) #Normalizando as entradas

X_teste = []
for i in range(90, 112): #for parecido com o anterior pra pegar lotes de 90 dias
    X_teste.append(entradas[i-90:i, 0])

X_teste = np.array(X_teste) #Passando pra numpy
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1)) #Tanto de registros(1152), tanto de registros por lote, somente um atributo previsor

previsoes = regressor.predict(X_teste) #Testando na rede

previsoes = normalizador.inverse_transform(previsoes) #Processo inverso da normalizaçao
previsoes.mean()
preco_real_teste.mean()
    
# ==== Fazendo grafico de comparaçao ====
plt.plot(preco_real_teste, color = 'red', label = 'Preço real')
plt.plot(previsoes, color = 'blue', label = 'Previsoes')
plt.title('Previsao preço das açoes')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()