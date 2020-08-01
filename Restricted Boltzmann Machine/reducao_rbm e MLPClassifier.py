import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB



base = datasets.load_digits()
previsores = np.asarray(base.data, 'float32')
classe = base.target

normalizador = MinMaxScaler(feature_range=(0,1))
previsores = normalizador.fit_transform(previsores)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.2, random_state=0)

rbm = BernoulliRBM(random_state = 0) #Bernoulli para reduzir a dimensionalidade
rbm.n_iter = 25 #Numero de iteraçoes
rbm.n_components = 50 #Neuroneos na camada escondida, nao precisamos colocar a de entrada pq ja tem os 64 pixels que el ja vai pegar 

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)
classificador_rbm = Pipeline(steps = [('rbm', rbm), ('mlp', mlp)]) #faz a reduçao com o Bernoulli e depois manda para o de Gauss
classificador_rbm.fit(previsores_treinamento, classe_treinamento)


# ==== Visualizando as imagens reduzidas =====
plt.figure(figsize=(20,20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8,8)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
plt.show()


 # ==== Comparando as precisoes ==== 
previsoes_rbm = classificador_rbm.predict(previsores_teste)
precisao_rbm = metrics.accuracy_score(previsoes_rbm, classe_teste)

naive_simples = GaussianNB()
naive_simples.fit(previsores_treinamento, classe_treinamento) #Aqui nao fazemos o RBM primeiro
previsoes_naive = naive_simples.predict(previsores_teste)
precisao_naive = metrics.accuracy_score(previsoes_naive, classe_teste)

# Criação da rede neural simples sem aplicação de rbm
mlp_simples = MLPClassifier(hidden_layer_sizes = (37, 37),
                        activation = 'relu', 
                        solver = 'adam',
                        batch_size = 50,
                        max_iter = 1000,
                        verbose = 1)
mlp_simples.fit(previsores_treinamento, classe_treinamento)
previsoes_mlp = mlp_simples.predict(previsores_teste)
precisao_mlp = metrics.accuracy_score(previsoes_mlp, classe_teste)

# Comparando os resultados, com RBM chegamos em 0.93 e sem RBM o percentual é de 0.98
# Com isso chegamos a conclusão que usar RBM com essa base de dados e com redes
# neurais piora os resultados