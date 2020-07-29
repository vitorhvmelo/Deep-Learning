import numpy as np
from keras.models import model_from_json

arquivo = open('classificador_breast.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_breast.h5')

novo = np.array([[6.4,3.1,5.5,1.8]])
 

previsao = classificador.predict(novo)
previsaoT = (previsao > 0.5)