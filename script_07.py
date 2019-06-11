import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
# importar os dados do iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2] # Nós só pegamos os dois primeiros recursos. Poderíamos
 # evitar esse corte feio usando um conjunto de dados bidimensional
y = iris.target
# Criamos uma instância do SVM e ajustamos os dados. Nós não dimensionamos nosso
# dados já que queremos traçar os vetores de suporte
C = 1.0 # Parâmetro de regularização do SVM
svc = svm.SVC(kernel='rbf', C=1,gamma=0.5).fit(X, y)
# Cria uma malha para traçar o gráfico
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')