"""
================================
Faces recognition example using eigenfaces and SVMs
================================

An example showing how the scikit-learn can be used to faces recognition with eigenfaces and SVMs

================================
================================

================================
Exemplo de reconhecimento de faces usando autofaces e SVMs
================================

Exemplo mostrando como o scikit-learn pode ser usado para reconhecimento de faces com autofaces e SVMs


"""
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC


print(__doc__)

# O logging exibe o progresso da execução no stdout
# adicionando as informações de data e hora
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# Download dos dados, se ainda não estiver em disco e carregue-a como uma matriz numpy.
# Nesse caso, só pega a pessoa que tiver no mínimo 70 imagens na base.
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Inspeciona as matrizes de imagens para encontrar os formatos das imagens( para plotagem )
n_samples, h, w = lfw_people.images.shape

# para aprendizado de máquina, usamos 2 dados diretamente( esse modelo ignora
# informações relativas a posição do pixel )
X = lfw_people.data
# Pega o número de características das imagens
n_features = X.shape[1]

# A rótulo( label ) a prever é o ID da pessoa
# Carrega a parte target dos dados em y
y = lfw_people.target
# target_names são as pessoas mais representadas no LFW, no caso 7 pessoas.
target_names = lfw_people.target_names
# ['Ariel Sharon', 'Colin Powell', 'Donald Rumsfeld', 'George W Bush','Gerhard Schroeder', 'Hugo Chavez', 'Tony Blair']
n_classes = target_names.shape[0] # n_classes é 7

print("Total dataset size:")
print(f"n_samples: {n_samples}")
print(f"n_features: {n_features}")
print(f"n_classes: {n_classes}")


# #############################################################################
# Divide a base em um conjunto de treinamento e um conjunto de teste usando k fold
# 25% da base para o conjunto de teste os 75% restantes para o treino
# random_state é para inicializar o gerador interno de números aleatórios
# Definir random_state com um valor fixo garantirá que a mesma sequência de números 
# aleatórios seja gerada cada vez que você executar o código
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


# #############################################################################
# Computa o principal component analysis - PCA (eigenfaces) na base de faces 
# ( tratado como dataset não rotulado ): extração não supervisionada / redução de dimensionalidade
n_components = 150
print(f"Extracting the top {n_components} eigenfaces from {X_train.shape[0]} faces")
# t0 guarda o tempo zero, para cálculo do tempo de execução do PCA
# O cálculo é feito no time() - t0 do print da linha 81
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in {0:.3f}s".format(time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in {0:.3f}s".format(time() - t0))


# #############################################################################
# Treinando um modelo de classificação SVM

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5, iid=False)
clf = clf.fit(X_train_pca, y_train)
print("done in {0:.3f}s".format(time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# #############################################################################
# Avaliação quantitativa da qualidade do modelo sobre o conjunto de testes

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in {0:.3f}s".format(time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# #############################################################################
# Avaliação qualitativa das previsões usando matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Função Helper para plotar a galeria de fotos"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# Plota o resultado da previsão em uma parte do conjunto de testes
def title(y_pred, y_test, target_names, i):
    # imagine o target_names[y_pred[i]] = 'George W Bush'
    # definir o parâmetro maxsplit como 1 no rsplit
    # retornará uma lista com 2 elementos ['George W', 'Bush']
    # o -1 pega só o último elemento ('Bush)
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1] # resultados que o modelo previu
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1] # Respostas corretas
    return f'predicted: {pred_name}\ntrue:      {true_name}'

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plota a galeria das autofaces mais significativas
eigenface_titles = [f"eigenface {i}" for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
# inicia um loop de eventos, procura todos os objetos de figura ativos
# no momento e abre uma ou mais janelas interativas que exibem sua figura
# ou figuras
plt.show()