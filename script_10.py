#Importa os datasets que vem com o scikit learn
from sklearn import datasets
#Importa o pandas como pd
import pandas as pd
#Carrega a base na forma como ela tá no Scikit-Learn, 
#ou seja, ela tá como dicionário
iris = datasets.load_iris()
#Transforma o iris dicionário em dataframe pandas 
#e atribui a variável irs
irs = pd.DataFrame(iris.data, columns = iris.feature_names)
#Separa os atributos, sepal length (cm), sepal width (cm), 
#petal length (cm) e petal width (cm) na variável x.
x = irs.iloc[:, :-1].values
#Separa o campo com as classificações: 0(setosa), 1(versicolor) e 2(virgínica) na variável y
y = irs.iloc[:, 4].values
#Importa o train_test_split do sklearn.model_selection 
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
#importa o StandardScaler do sklearn.preprocessing
#para normalizar a distribuição, com média próxima de zero e um desvio padrão próximo a um.
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(x_train)
#Importa o KNeighborsClassifier do sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 
#A classe é KNeighborsClassifier inicializada com um parâmetro: 
#n_neigbours igual a 5. 
knn_classifier = KNeighborsClassifier(n_neighbors = 5)
#Faz as previsões
knn_classifier.fit(x_train, y_train)
#Importa confusion_matrix e classification_report do sklearn.metrics
from sklearn.metrics import classification_report, confusion_matrix
#Imprime a confusion matrix 
print(confusion_matrix(y_test, y_pred))
#Imprime o relatório de classificação  
print(classification_report(y_test, y_pred)) 
#######Vamos salvar o modelo usando o joblib#############
#Salva o modelo em disco
filename = 'kNeighborsClassifier_iris_model.sav'
joblib.dump(knn_classifier, filename)
#Algumas horas ou talvez dias dias depois...
#Carrega o modelo do disco
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)