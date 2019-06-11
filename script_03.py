import numpy as np 
import pandas as pd 
from sklearn.datasets import load_iris 
#Carrega o iris dataset em iris 
iris = load_iris() 
#Cria o DataFrame em df_iris utilizando um numpy array (np) 
df_iris = pd.DataFrame(np.column_stack((iris.data, iris.target)), 
    columns = iris.feature_names + ['target'])
df_iris.describe()