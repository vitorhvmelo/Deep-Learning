import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model

base = pd.read_csv('games.csv')

base = base.drop('Other_Sales', axis = 1) #axis = 1 apaga coluna, axis = 0 apara linha
base = base.drop('Global_Sales', axis = 1)
base = base.drop('Developer', axis = 1)
base = base.dropna(axis = 0)

a = base.loc[base['NA_Sales'] < 1]
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]

base['Name'].value_counts()
nome_jogos = base.Name #vamos reservar o nome para uma futura avaliacao
base = base.drop('Name', axis = 1) #nome nao é importante para a classificaçao

previsores = base.iloc [:,[0,1,2,3,7,8,9,10,11]].values
venda_na=base.iloc[:]