# importa os dataset que vem com o sklearn
from sklearn import datasets
import pandas as pd
# Carrega o digits dataset na variável digits
digits = datasets.load_digits()
# Transforma os dados em pandas dataframe
data_x = pd.DataFrame(digits.data)
target_y = pd.DataFrame(digits.target)
#Importa o svm(support vector machine) do sklearn
from sklearn import svm
# Instancia um objeto svm em clf
clf = svm.SVC(gamma=0.001, C=100.)
#---------- Dividinda a base (Treino e Teste)
test_size = 0.33
seed = 7
# Importa do sklearn o train_test_split
from sklearn.model_selection import train_test_split
#Divisão do digits dataset em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(data_x, target_y, test_size = test_size, random_state = seed)
#-------- Treinando o modelo -------------
# Treina o modelo clf
clf.fit(X_train, Y_train)
# Prevê o valor do registro 1791 do dígito do dataset, que é um 4.
clf.predict(data[:][1791:1792])
# Vamos conferir se o modelo acertou com
target[:][1791:1792]
# Mostra o quanto o modelo é acertivo.
clf.score(X_test, Y_test)
#------   Salvando o modelo -----------------
# importa o pickle
import pickle
# Define o nome do arquivo em disco que irá 
# guardar o nosso modelo
filename = 'predict_digits_model.sav'
# salva o modelo no disco
pickle.dump(clf, open(filename, 'wb'))
# Carregando o modelo do disco
loaded_model = pickle.load(open(filename, 'rb'))
# Atribui a variável result o score do modelo
result = loaded_model.score(X_test, Y_test)
#Imprime o resultado
print(result)