from sklearn import datasets
#load iris dataset
iris = datasets.load_iris()
#show the type of iris
type(iris)
#print the data of iris dataset, you can use iris['data'] too
print(iris.data)
#print feature_names
print(iris.feature_names)
#or without print
iris.feature_names
#show a list of target names
list(iris.target_names)
#print target
print(iris.target)
#or without print
iris.target
#print type of iris.data and iris.target
print(type(iris.data))
print(type(iris.target))
#print the shape of data and target
print(iris.data.shape)
print(iris.target.shape)
#assign data to x and target to y
x = iris.data
y = iris.target