from minisom import MiniSom 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pylab import pcolor, colorbar, plot

# ==== Estrutura dos dados ====
base = pd.read_csv('entradas-breast.csv')
saidas = pd.read_csv('saidas-breast.csv')
X = base.iloc[:,1:30].values #Todas as colunas  menos a primeira e a ultima
y = saidas.iloc[:,0].values #Pegando a primeira coluna - classes

normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X) # Normalizando

# ==== Estrutura do Mapa ==== 
'''
MiniSom:
x e y sao as linhas e as colunas, achamos essa dimensao com a formula 5*raiz(n), sendo n o numero de dados
input_len é a quantidade de atributos
sigma é o raio do neuroneo
learning_rate é a taxa de aprendizagem
random_seed inicializacao dos pesos sera as mesmas
'''
som = MiniSom(x = 10, y = 10, input_len = 29, sigma = 2, learning_rate = 0.2, random_seed = 2) 
som.random_weights_init(X) #inicializaçao randomica dos pesos
som.train_random(data = X, num_iteration = 10000) #num_iteration = epocas

a = som._weights #Mosta os pesos colocados no mapa: 13 elementos * 8 linhas * 8 blocos = cubo de 832 pesos, que seria um ponto novo no mapa
som._activation_map #Os valores de ativaçao do mapa
q = som.activation_response(X) #Mostra quantas vezes cada neuroneo foi selecionado como BMU(melhor neuroneo)


pcolor(som.distance_map().T) # MID - mean inter neuron distance, quanto mais distante (diferente) dos seus vizinhos mais claro
colorbar()                   #Quanto mais diferente menos confiavel é o vizinho

w = som.winner(X[2]) #Mostra qual o BMU para cada registro
markers = ['o', 's'] # o = bolinha, s = square, D = Logango
color = ['r', 'g']
#y[y == 1] = 0
#y[y == 2] = 1

for i, x in enumerate(X):
    w = som.winner(x) #volta uma tupla de coordenas, vamos colocar +0,5 para o marcador ficar no centro do quadradinho
    
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = color[y[i]], markeredgewidth = 2)