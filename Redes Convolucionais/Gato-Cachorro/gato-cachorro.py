from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.layers.normalization import BatchNormalization 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import numpy as np
from tensorflow.keras.preprocessing import image 


classificador = Sequential()
# ====Primeira Camada Conv====
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu')) #Aqui temos tamanhos variados de imagens,  que serao 
                                                                                     #convertidas para 64x64, colocamos 3 canais pq trabalharemos com RGB'''
classificador.add(BatchNormalization()) #Normalizando na camada convolucional
classificador.add(MaxPooling2D(pool_size = (2,2))) #Operador de Pooling
# ====Segunda Camada Conv====
classificador.add(Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu')) 
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten()) #Operador de Flatening

#=== Duas camadas ocultas com Dropout ====
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 1, activation = 'sigmoid')) #Camada de saida, como é uma classificaçao binaria gato/cachorro usamos sigmoid

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

'''Observaçao interessante sobre o ImageDataGenerator: mesmo se nao quisermos fazer alguma transformaçao na imagem (como na base de teste), é legal utilizar essa funçao porque nós conseguimos fazer a normalizaçao dos dados (repare no rescale = 1./255) sem precisar carregar todas as imagens do disco, transforma-las em matrizes numpy e posteriormente fazer a normalizaçao. Desta forma economizamos varias linhas de codigo'''

gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)
gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'binary')
base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size = (64, 64),
                                               batch_size = 32,
                                               class_mode = 'binary')
classificador.fit_generator(base_treinamento, steps_per_epoch = 4000 / 32,
                            epochs = 5, validation_data = base_teste,
                            validation_steps = 1000 / 32)

imagem_teste = image.load_img('dataset/test_set/gato/cat.3519.jpg',
                              target_size = (64,64)) #Carregamos uma imagem qualquer do disco
imagem_teste = image.img_to_array(imagem_teste) #transformamos a imagem em um array
imagem_teste /= 255 #Normalizaçao do Array
imagem_teste = np.expand_dims(imagem_teste, axis = 0) #Usamos o Numpy para aumentar uma dimensao do array, que é o padrao que o tf trabalha
previsao = classificador.predict(imagem_teste) #Fazemos a classificaçao
base_treinamento.class_indices #Quanto mais proximo de 0 é cachorro e quanto mais proximo de 1 é gato

if previsao > 0.5:
    print ('É um gato')
else:
    print('É um cachorro')





