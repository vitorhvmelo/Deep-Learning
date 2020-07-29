#!/usr/bin/env python
# coding: utf-8

# # Importações
# 
#   Prestar **atenção** que o KERAS foi incorporado no TensorFLow 2.0, dessa madeira a importaçao dos modulos keras deve ser feita como **tensorflow.keras**
# 
# 
# 
# 
# 

# In[1]:



import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Dropout 
from tensorflow.python.keras.utils import np_utils 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.python.keras.layers.normalization import BatchNormalization 


# # Tratamento dos dados
# 
#  Obs: As variáveis **dummys** devem ser utilizadas sempre que desejarmos incluir variáveis **categóricas** em modelos que aceitam apenas variáveis **numéricas**.

# In[2]:


(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data() #Fazemos a importaçao do MNIST
plt.imshow(X_treinamento[1], cmap = 'gray') #abrimos como exemplo o treinamento [1], e colocacamos e  escala preto e branco
plt.title('Classe ' + str(y_treinamento[1])) #Titulo

previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],
                                               28, 28, 1) #Fazemos o reshape para que o TF consiga ler os dados
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
previsores_treinamento = previsores_treinamento.astype('float32') #Mudamos para float para podermos dividir logo abaixo
previsores_teste = previsores_teste.astype('float32')

previsores_treinamento /= 255 #Normalizaçao (1) Pra diminuir o custo operacional, dividimos os valores RGB por 255, 
                              #dessa forma temos uma escala de 0 ate 1
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento, 10) #Transformamos os dados em variaveis dummy
classe_teste = np_utils.to_categorical(y_teste, 10)


# # Estrutura da Rede Neural
# 
# #       Conv2d:
# 
# 
# 1.  Primeiro parametro: filters 32 Kernels diferents que serao testados, o recomendavel é começar com 64
# 2.  Segundo parametro: kernel_size: o tamanho do kernel
# 
# 

# In[4]:


classificador = Sequential()
#       ======Primeira Camada Convolucional ======
classificador.add(Conv2D(32, (3,3), 
                         input_shape=(28, 28, 1),
                         activation = 'relu')) #Operador de Convoluçao
classificador.add(BatchNormalization()) #Segue uma forma parecida da normalizaçao anterior(1), mas agora nas camadas de conv.
classificador.add(MaxPooling2D(pool_size = (2,2))) #Operador de Pooling
#classificador.add(Flatten()) Flatening somente na ultima camada de convoluçao

#       ======Segunda Camada Convolucional ======
classificador.add(Conv2D(32, (3,3), activation = 'relu')) #Vale notar que precisamos passar o shape so na primeira camada
classificador.add(BatchNormalization()) 
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Flatten()) #Operador de Flatening


classificador.add(Dense(units = 128, activation = 'relu')) #1 camada oculta
classificador.add(Dropout(0.2)) #Recomendavel colocar pq redes conv tem muitas entradas, evitando overfittining
classificador.add(Dense(units = 128, activation = 'relu')) #2 camada oculta
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, 
                        activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam', metrics = ['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 128, epochs = 2,
                  validation_data = (previsores_teste, classe_teste))


# # Teste de uma unica amostra
# 

# In[26]:


# Nesse exemplo escolhi a primeira imagem da base de teste e abaixo você
# pode visualizar que trata-se do número 7
num_test = int(input('Digite a posiçao do dado que deseja analisar:  '))
plt.imshow(X_teste[num_test], cmap = 'gray')
plt.title('Classe ' + str(y_teste[num_test]))

# Criamos uma única variável que armazenará a imagem a ser classificada e
# também fazemos a transformação na dimensão para o tensorflow processar
imagem_teste = X_teste[num_test].reshape(1, 28, 28, 1)

# Convertermos para float para em seguida podermos aplicar a normalização
imagem_teste = imagem_teste.astype('float32')
imagem_teste /= 255

# Fazemos a previsão, passando como parâmetro a imagem.
#Como temos um problema multiclasse e a função de ativação softmax, será
# gerada uma probabilidade para cada uma das classes. A variável previsão
# terá a dimensão 1, 10 (uma linha e dez colunas), sendo que em cada coluna
# estará o valor de probabilidade de cada classe
previsoes = classificador.predict(imagem_teste)

# Como cada índice do vetor representa um número entre 0 e 9, basta agora buscarmos qual é o maior índice e o retornarmos. 
resultado = np.argmax(previsoes)
print(f'''\nA classe analisada foi a CLASSE {y_teste[num_test]} e a rede detectou a CLASSE {resultado}

O peso com os resultados foi: {previsoes}''')


# In[ ]:




