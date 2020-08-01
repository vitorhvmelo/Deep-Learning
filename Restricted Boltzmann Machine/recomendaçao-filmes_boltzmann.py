from rbm import RBM #Importaçao do modulo, tem no github
import numpy as np

rbm = RBM(num_visible = 6, num_hidden = 2) #Temos 6 filmes entao 6 neuroneos de entrada(visible) temos duas categorias de filmes entao dois neuroneos escondidos

base = np.array([[1,1,1,0,0,0],
                 [1,0,1,0,0,0],
                 [1,1,1,0,0,0],
                 [0,0,1,1,1,1],
                 [0,0,1,1,0,1],
                 [0,0,1,1,0,1]]) #Essa seria a a classificaçao dos filmes de 6 usuarios, sendo 1 = viu e gostou e 0 = nao viu

filmes = ["A bruxa", "Invocação do mal", "O chamado",
          "Se beber não case", "Gente grande", "American pie"]

rbm.train(base, max_epochs=5000)
rbm.weights #Mostra os pesos, a segunda e terceira coluna mostra o valor dos pesos dos dois neuroneos, numero mais altos mostram que aquele filme ativou mais aquele determinado neuroneo, da pra notar que o filmes de terror ficaram com o segundo neuroneo e o de comedio com o primeiro

usuario1 = np.array([[1,1,0,1,0,0]])
usuario2 = np.array([[0,0,0,1,1,0]])

neo1 = rbm.run_visible(usuario1) #Ativou mais o segundo neuroneo, entao serao recomendados filmes de terror
neo2 = rbm.run_visible(usuario2) #Ativou mais o primeiro neuroneo, entao serao recomendados filmes de comedia

camada_escondida1 = np.array([[0, 1]]) #Resultado do neo1
recomendacao1 = rbm.run_hidden(camada_escondida1) #Sai um vetor com as recomendaçoes
camada_escondida2 = np.array([[1,0]]) #Resultado do neo2
recomendacao2 = rbm.run_hidden(camada_escondida2)

print('Recomendaçoes usuario1:')
for i in range(len(usuario1[0])):
    #print(usuario1[0,i])
    
    if usuario1[0,i] == 0 and recomendacao1[0,i] == 1: #Se ele nao viu e tem a recomedaçao, printamos o nome do filme
        print(filmes[i])

print('\nRecomendaçoes usuario2:')
for i in range(len(usuario1[0])):
    #print(usuario1[0,i])
    if usuario2[0,i] == 0 and recomendacao2[0,i] == 1: #Se ele nao viu e tem a recomedaçao, printamos o nome do filme
        print(filmes[i])
    