from sklearn import datasets
#load iris dataset
iris = datasets.load_iris()

x = iris.data
y = iris.target

x.shape
y.shape

iris

targets.reshape(targets.shape[0],-1)
targets.shape

featuresAll=[]
features = iris.data[: , [0,1,2,3]]
features.shape
iris.feature_names

for observation in features:
    featuresAll.append([observation[0] + observation[1] + observation[2] + observation[3]])

print(featuresAll)

####################################################################
#Plotando o gráfico de dispersão (Relação entre comprimento e largura sépala)
import matplotlib.pyplot as plt
plt.scatter(featuresAll, targets, color='red', alpha =1.0)
plt.rcParams['figure.figsize'] = [10,8]
plt.title('Iris Dataset scatter Plot')
plt.xlabel('Features')
plt.ylabel('Targets')

plt.show()

####################################################################	
#Gráfico de Dispersão com Dataset Iris (Relação entre o Comprimento e a Largura da Sépala)
#Encontrando o relacionamento entre o comprimento e a largura da sépala
sepal_len = []
sepal_width = []
for feature in features:
    sepal_len.append(feature[0]) #Comprimento da sépala
    sepal_width.append(feature[1]) #Largura da sépala

groups = ('Iris-setosa','Iris-versicolor','Iris-virginica')
colors = ('blue', 'green','red')
data = ((sepal_len[:50], sepal_width[:50]), (sepal_len[50:100], sepal_width[50:100]), 
        (sepal_len[100:150], sepal_width[100:150]))

for item, color, group in zip(data, colors, groups): 
    #item = (sepal_len[:50], sepal_width[:50]), (sepal_len[50:100], sepal_width[50:100]), 
    #(sepal_len[100:150], sepal_width[100:150])
    x0, y0 = item
    plt.scatter(x0, y0,color=color,alpha=1)
    plt.title('Iris Dataset scatter Plot')

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

#####################################################################
#Gráfico de Dispersão com Conjunto de Dados Iris (Relação entre o Comprimento e a Largura da Pétala)
#Encontrando o relacionamento entre o comprimento e a largura da pétala
petal_len = []
petal_width = []
for feature in features:
    petal_len.append(feature[2]) #Comprimento da pétala
    petal_width.append(feature[3]) #Largura da pétala


groups = ('Iris-setosa','Iris-versicolor','Iris-virginica')
colors = ('blue', 'green','red')
data = ((petal_len[:50], petal_width[:50]), (petal_len[50:100], petal_width[50:100]), 
        (petal_len[100:150], petal_width[100:150]))

for item, color, group in zip(data,colors,groups): 
    #item = (petal_len[:50], petal_width[:50]), (petal_len[50:100], petal_width[50:100]), 
    #(petal_len[100:150], petal_width[100:150])
    x0, y0 = item
    plt.scatter(x0, y0,color=color,alpha=1)
    plt.title('Iris Dataset scatter Plot')

plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.show()