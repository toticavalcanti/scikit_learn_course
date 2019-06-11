#Importa os datasets que vem com o scikit learn
from sklearn import datasets
import pandas as pd
iris = datasets.load_iris()
#Cria o data frame e atribui a irs
irs = pd.DataFrame(iris.data, columns = iris.feature_names)
irs['class'] = iris.target
#Mostra só o início do data frame
irs.head()

#Mostra todas colunas do conjunto de dados
irs.columns
#conta quantas linhas tem a base de dados
irs.count()
#O describe mostra o desvio padrão, média, valor mínimo e máximo de colunas
irs.describe()

#x recebe sepal length (cm), sepal width (cm), 
# petal length (cm) e petal width (cm) e o y recebe os rótulos, 
#isto é, as classes de cada flor (0, 1 e 2).
x = irs.iloc[:, :-1].values
y = irs.iloc[:, 4].values

from sklearn.model_selection import train_test_split  
#cria as divisões de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)


from sklearn.preprocessing import StandardScaler
# redimensionamento as features
scaler = StandardScaler()  
scaler.fit(x_train)

x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)

#Treinamento e Previsões
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors = 5)  
classifier.fit(x_train, y_train)

#faz as previsões sobre os dados de teste
y_pred = classifier.predict(x_test)

#Avaliando o Algoritmo
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  

#ajusta para 9 o n_neighbors = 9 para ver se o resultado muda.
classifier = KNeighborsClassifier(n_neighbors = 9)
classifier.fit(x_train, y_train)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))