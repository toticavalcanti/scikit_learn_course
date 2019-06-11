#Importa os datasets que vem com o scikit learn
from sklearn import datasets
#importa a biblioteca pandas como pd
import pandas as pd
iris = datasets.load_iris()
irs = pd.DataFrame(iris.data, columns = iris.feature_names)
irs['class'] = iris.target

x = irs.iloc[:, :-1].values
y = irs.iloc[:, 4].values

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(x_train)

x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test) 

#importa o numpy como np
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  
error = []

# Calculando erro para valores de K entre 1 e 40
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

#importa o matplotlib como plt
import matplotlib.pyplot as plt  
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')
#Para mostrar o gráfico
plt.show()