#importa a biblioteca pandas como pd
import pandas as pd
#Importa o numpy como np
import numpy as np

#Cria um array numpy
data = np.array(['a','b','c','d'])

#Usa o array numpy criado acima para 
#gerar um objeto Series do pandas
s1 = pd.Series(data)

s1

#########################################
#Cria um array numpy
data2 = np.array(['a','b','c','d'])
s2 = pd.Series(data2, index = [70, 71, 72, 73])
print(s2)

#########################################
data3 = {'a' : 0., 'b' : 1., 'c' : 2.}
s3 = pd.Series(data3)
s3

#########################################
data4 = {'a' : 0., 'b' : 1., 'c' : 2.}
s4 = pd.Series(data4, index = ['b','c','d','a'])
print(s4)

#########################################
s = pd.Series([1, 2, 3, 4, 5], index = ['a','b','c','d','e'])

#Recupera um único elemento
print(s['a'])

#Recupera vários elementos
s[['a','c','d']]

#Tentar acessar uma chave que não existe, dá um KeyError.
print(s['f'])

#########################################
#Cria um dataframe vazio e atribui a df
df = pd.DataFrame()
print(df)

#Um DataFrame pode ser criado usando uma única lista ou uma lista de listas.
data = [1, 2, 3, 4, 5]
df = pd.DataFrame(data)
print(df)

#Exemplo 2
data = [ ['Maria', 10], ['Carlos', 12], ['Paulo', 13] ]
df = pd.DataFrame(data, columns = ['Nome', 'Idade'])
print(df)

#Criando um data frame a partir de um dicionário ndarrays / Lists
data = {'Nome':['Marcos', 'Paula', 'Lia', 'Carlos'], 'Idade':[28, 34, 29, 42]}
df = pd.DataFrame(data)
print(df)

#Outro exemplo
data = {'Nome': ['Marcos', 'Paula', 'Lia', 'Carlos'], 'Pontuação': [7.5, 6.8, 5.9, 8.3]}
df = pd.DataFrame(data, index = ['rank1', 'rank2', 'rank3', 'rank4'])
print(df)

#Criando um data frame a partir de uma lista de dicionários.	
data = [ {'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20} ]
df = pd.DataFrame(data)
print(df)

#Criando um data frame a partir de uma lista de dicionários e os índices de linhas.
data = [ {'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20} ]
df = pd.DataFrame(data, index = ['primeiro', 'segundo'])
print(df)


#Criando um DataFrame com uma lista de dicionários, índices de linhas e índices de colunas.

#Com dois índices de colunas, os valores são iguais aos das chaves do dicionário
df1 = pd.DataFrame(data, index = ['primeiro', 'segundo'], columns = ['a', 'b'])

#Com dois índices de colunas com um índice com outro nome
df2 = pd.DataFrame(data, index = ['primeiro', 'segundo'], columns = ['a', 'b1'])
print(df1)
print(df2)

######################################################################
#Criando um DataFrame a partir do Dicionário de Series

dic = {'um' : pd.Series([1, 2, 3], index = ['a', 'b', 'c']),
   'dois' : pd.Series([1, 2, 3, 4], index = ['a', 'b', 'c', 'd'])}

df = pd.DataFrame(dic)
print(df)

#Seleciona a chave "um"
df = pd.DataFrame(dic)
print(df ['um'])

dic = {'um' : pd.Series([1, 2, 3], index = ['a', 'b', 'c']),
   'dois' : pd.Series([1, 2, 3, 4], index = ['a', 'b', 'c', 'd'])}

df = pd.DataFrame(dic)

# Adicionando uma nova coluna a um objeto DataFrame existente 
# com rótulo de coluna passando nova Series

print ("Adicionando uma nova coluna passando como Série:")
df['três'] = pd.Series([10, 20, 30], index = ['a','b','c'])
print(df)

print ("Adicionando uma nova coluna usando as colunas existentes no DataFrame")
df['quatro'] = df['um'] + df['três']

print(df)

################################################################
#Apagando colunas
d = {'um' : pd.Series([1, 2, 3], index = ['a', 'b', 'c']), 
   'dois' : pd.Series([1, 2, 3, 4], index = ['a', 'b', 'c', 'd']), 
   'três' : pd.Series([10,20,30], index = ['a','b','c'])}

df = pd.DataFrame(d)
print ("Nosso dataframe é:")
print(df)

# usando a função del
print ("Excluindo a primeira coluna usando a função DEL:")
del df['um']
print(df)

# usando a função pop
print ("Excluindo outra coluna usando a função POP:")
df.pop('dois')

print(df)

##################################################################
#Seleção, adição e exclusão de linha

#As linhas podem ser selecionadas passando o rótulo da linha para uma função loc.
dic = {'um' : pd.Series([1, 2, 3], index = ['a', 'b', 'c']), 
   'dois' : pd.Series([1, 2, 3, 4], index = ['a', 'b', 'c', 'd'])}

df = pd.DataFrame(dic)
print(df.loc['b'])

#As linhas podem ser selecionadas passando a localização inteira para uma função iloc.
dic = {'um' : pd.Series([1, 2, 3], index = ['a', 'b', 'c']),
   'dois' : pd.Series([1, 2, 3, 4], index = ['a', 'b', 'c', 'd'])}

df = pd.DataFrame(dic)
print(df.iloc[2])

#Várias linhas podem ser selecionadas usando o operador “:“.
d = {'um' : pd.Series([1, 2, 3], index = ['a', 'b', 'c']), 
   'dois' : pd.Series([1, 2, 3, 4], index = ['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print(df[2:4])

#Adicione novas linhas a um DataFrame usando a função append.Esta função irá anexar as linhas no final.
df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])

df = df.append(df2)
print(df)

#Vamos deletar um rótulo e ver quantas linhas serão descartadas.
df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])

df = df.append(df2)

#Deleta as linhas com o rótulo 0
df = df.drop(0)

print(df)