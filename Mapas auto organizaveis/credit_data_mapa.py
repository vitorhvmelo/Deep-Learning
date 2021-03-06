from minisom import MiniSom
import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')
base = base.dropna()
base.loc[base.age < 0, 'age'] = 40.92 #Trocamos idades negativas para a media de idade do Banco de dados

X = base.iloc[:, 0:4].values
y = base.iloc[:, 4].values

from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)

som = MiniSom(x = 15, y = 15, input_len = 4, random_seed = 0) #Deixaremos o learning_rate e sigma default sendo 0.5 e 0 respectivamente
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

from pylab import pcolor, colorbar, plot
pcolor(som.distance_map().T) #Os amarelos podem ser outliers
colorbar()

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = colors[y[i]], markeredgewidth = 2)

# ==== Detectando Fraudes ====
    
mapeamento = som.win_map(X)
suspeitos = np.concatenate((mapeamento[(13,9)], mapeamento[(1,10)]), axis = 0) #Pegamos dois quadradinhos amarelos como suspeitos, pode ser diferente dependendo do resultado
suspeitos = normalizador.inverse_transform(suspeitos)

classe = []
for i in range(len(base)):
    for j in range(len(suspeitos)):
       if base.iloc[i, 0] == int(round(suspeitos[j,0])): #Roud serve para arredondar
           classe.append(base.iloc[i,4])
classe = np.asarray(classe)

suspeitos_final = np.column_stack((suspeitos, classe))
suspeitos_final = suspeitos_final[suspeitos_final[:, 4].argsort()]