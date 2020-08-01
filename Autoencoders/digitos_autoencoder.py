import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.layers import Input, Dense 

# ==== Estrutura dos Dados ====

(previsores_treinamento, _), (previsores_teste, _) = mnist.load_data()  # _ pq nao queremos criar essas variaveis
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:]))) #cada uma das linhas tem uma img
previsores_teste = previsores_teste.reshape((len(previsores_teste), np.prod(previsores_teste.shape[1:]))) #cada uma das linhas tem uma img

# 784 - 32 - 784 #Camada de entrada - Camada oculta - cada de saida
fator_compactacao = 784 / 32 

autoencoder = Sequential()
autoencoder.add(Dense(units = 32, activation = 'relu', input_dim = 784)) 
autoencoder.add(Dense(units = 784, activation = 'sigmoid')) #Numero de neuroneos da saida deve ser igual da entrada
autoencoder.summary() #Mostra a estrutura do encoder
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                    metrics = ['accuracy'])
autoencoder.fit(previsores_treinamento, previsores_treinamento,
                epochs = 50, batch_size = 256,
                validation_data = (previsores_teste, previsores_teste)) #Comparamos treinamento com treinamento para sabermos se esta sendo bem reconstruido e depois comparamos com o teste

dimensao_original = Input(shape=(784,))
camada_encoder = autoencoder.layers[0]
encoder = Model(dimensao_original, camada_encoder(dimensao_original))
encoder.summary()

imagens_codificadas = encoder.predict(previsores_teste)
imagens_decodificadas = autoencoder.predict(previsores_teste)

numero_imagens = 10
imagens_teste = np.random.randint(previsores_teste.shape[0], size = numero_imagens)
plt.figure(figsize=(18,18))
for i, indice_imagem in enumerate(imagens_teste):
    #print(i)
    #print(indice_imagem)
    
    # imagem original
    eixo = plt.subplot(10,10,i + 1)
    plt.imshow(previsores_teste[indice_imagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
    
    # imagem codificada
    eixo = plt.subplot(10,10,i + 1 + numero_imagens)
    plt.imshow(imagens_codificadas[indice_imagem].reshape(8,4))
    plt.xticks(())
    plt.yticks(())
    
     # imagem reconstruída
    eixo = plt.subplot(10,10,i + 1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
    