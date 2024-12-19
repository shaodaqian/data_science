import numpy as np
import os
import tarfile
import urllib.request
import pandas as pd
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler,LabelEncoder,RobustScaler,PolynomialFeatures
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,IterativeImputer

from sklearn.feature_selection import SelectKBest, f_regression,mutual_info_regression

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

from scipy.stats.contingency import association


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("./datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data()
dataset=load_housing_data()

print(dataset.head())
print(dataset.describe())
print(dataset.info())

dataset.hist(bins=50,figsize=(20,15))

dataset.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=dataset["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
# plt.show()

cat_cols=[]
num_cols=[]
for column in dataset:
    data=dataset[column]
    if not is_numeric_dtype(data):
        cat_cols.append(column)
    else:
        num_cols.append(column)

print('category features:', len(cat_cols))
print('numeric features:', len(num_cols))


for column in cat_cols:
    data=dataset[column]
    encoder=LabelEncoder()
    test=encoder.fit_transform(data)
    test=np.expand_dims(test,-1)
    cat_cols.append(column)

    nulls= np.expand_dims(dataset['price [€]'].isnull().to_numpy(),-1)
    concat=np.concatenate((nulls,test),-1)
    concat+=1
    asso=association(concat,method="cramer")
    print(column, asso)

corr_matrix = dataset[num_cols].corr()

corr_matrix = dataset[num_cols].corr("kendall")
corr_matrix = dataset[num_cols].corr("spearman")

plt.matshow(corr_matrix)
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_matrix, linewidths=.5)

# plt.show()

print(corr_matrix["median_house_value"].sort_values(ascending=False))
print(corr_matrix["total_rooms"].sort_values(ascending=False))

dataset_num=dataset[num_cols]
mice=IterativeImputer()
dataset_num_imputed=mice.fit_transform(dataset_num)
f_test,_=f_regression(dataset_num_imputed,dataset["median_house_value"])
f_test/=np.max(f_test)
print(f_test)
mi=mutual_info_regression(dataset_num_imputed,dataset["median_house_value"])
mi/=np.max(mi)
print(mi)

# max=dataset['housing_median_age'].max()
# TobitLinear:
# x=dataset.drop('housing_median_age',axis=1)
# y=dataset['housing_median_age']
# cens=np.zeros_like(y)
# cens[y==max]=1
# tobit=TobitLinear()
# tobit.fit(x,y,cens)

train_data,test_data=train_test_split(dataset,test_size=0.2,stratify=dataset["ocean_proximity"])

train_x=train_data.drop("median_house_value",axis=1)
train_y=train_data["median_house_value"]
num_cols.remove("median_house_value")

num_pipeline=Pipeline([('imputer',IterativeImputer()),('scaler',RobustScaler())])
cat_pipeline=Pipeline([('encoder',OneHotEncoder())])

preprocessor=ColumnTransformer([('num',num_pipeline,num_cols),('cat',cat_pipeline,cat_cols)])

train_x=preprocessor.fit_transform(train_x)

y_scalar=RobustScaler()
train_y=y_scalar.fit_transform(train_y.values.reshape(-1,1))

ridge_grid=GridSearchCV(Ridge(),{'alpha':[0.1,1,10]},scoring='neg_mean_squared_error',cv=10)
ridge_grid.fit(train_x,train_y)
cvres=ridge_grid.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

param_grid=[{'n_estimators':[10,30,60],'max_features':[4,6,8]},{'bootstrap':[False],'n_estimators':[10,30],'max_features':[4,8]}]
forest_grid=GridSearchCV(RandomForestRegressor(),param_grid,scoring='neg_mean_squared_error',cv=10,n_jobs=-1)
forest_grid.fit(train_x,train_y)
cvres=forest_grid.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


param_grid=[{'n_estimators':[10,30,60],'max_features':[4,6,8]}]
gboost_grid=GridSearchCV(GradientBoostingRegressor(),param_grid,scoring='neg_mean_squared_error',cv=10,n_jobs=-1)
gboost_grid.fit(train_x,train_y)
cvres=gboost_grid.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = forest_grid.best_estimator_.feature_importances_

test_x=test_data.drop("median_house_value",axis=1)
test_y=test_data["median_house_value"]
test_x=preprocessor.transform(test_x)

test_y=y_scalar.transform(test_y.values.reshape(-1,1))

pred=forest_grid.best_estimator_.predict(test_x)
pred=y_scalar.inverse_transform(pred.reshape(-1,1))
test_y=y_scalar.inverse_transform(test_y.reshape(-1,1))

print(pred.shape,test_y.shape)
print('rmse', np.sqrt(mean_squared_error(test_y,pred)))
print('mae', mean_absolute_error(test_y,pred))
