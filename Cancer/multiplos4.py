pares = open('pares.txt', 'r')
multiplos4 = open('multiplos4.txt', 'w')
for c in pares.readlines(): #readlines retorna uma lista cujos elementos sao as linhas do arquivo
    if int(c) % 4 == 0:
        multiplos4.write(c)

pares.close()
multiplos4.close()
