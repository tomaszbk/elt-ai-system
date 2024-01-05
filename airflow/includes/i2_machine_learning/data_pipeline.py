from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TypeOfLaptopDropper(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y= None):
        return self
    
    def transform(self, X):
        return X.drop('type_of_laptop', axis=1)
    

class ValueMapper(BaseEstimator, TransformerMixin):
    def map_resolution(value: str):
        value = value.upper()
        if '4K' in value:
            return 2
        return 1

    def map_os(value: str):
        value = value.upper().split(' ')[0]
        if value == 'NO':
            return None
        return value

    def map_gpu(value: str):
        value = value.upper()
        if 'NVIDIA' in value:
            return 'NVIDIA'
        elif 'INTEL' in value:
            return 'INTEL'
        elif 'AMD' in value:
            return 'AMD'

    def map_memory(value: str):
        value = value.upper()
        if 'SSD' in value:
            disk_type = 'SSD'
        else:
            disk_type = 'HDD'
        disk_size = value.split('GB')[0]
        try:
            return disk_type, int(disk_size)
        except:
            return disk_type, 256
    
    def fit(self, X, y= None):
        return self
    
    def transform(self, X):
        X[['disk_type', 'disk_size']] = X['memory'].apply(self.map_memory).apply(pd.Series)
        X['screen_resolution'] = X['screen_resolution'].apply(self.map_resolution)
        X['os'] = X['os'].apply(self.map_os)
        X['gpu'] = X['gpu'].apply(self.map_gpu)
        X.cpu = X.cpu.map(lambda x: x.split()[0])
        X['disk_type'] = X['disk_type'].map({'SSD': 1, 'HDD': 0})
        return X
    

class MemoryDropper(BaseEstimator, TransformerMixin):
        
        def fit(self, X, y= None):
            return self
        
        def transform(self, X):
            return X.drop('memory', axis=1) 
 

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y= None):
        return self
    
    def transform(self, X):
        return pd.get_dummies(X, columns=['company_name', 'ram', 'os', 'gpu', 'cpu'], prefix = ['laptop_brand', 'ram', 'os', 'gpu', 'cpu']).head()


data_transform_pipeline = Pipeline([
     'drop_type_of_laptop_column', TypeOfLaptopDropper(),
     'map_values', ValueMapper(),
    'drop_memory_column', MemoryDropper(),
    'one_hot_encoding', CustomOneHotEncoder(),
    'standardize', StandardScaler()
])