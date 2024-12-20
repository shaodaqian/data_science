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




org_dataset=dataset.copy()

def split_feet(row):
    # nan check
    if row != row :
        return None
    if "”" in row:
        meter = row[row.rfind("”")+1:][:-2]
        feet = round(float(row[:row.rfind("”")+1].split("’")[0]) + float(row[row.rfind("’")+1:row.rfind("”")])/12,2)
    elif "’" in row:
        meter = row[row.rfind("’")+1:][:-2]
        feet = row[:row.rfind("’")]
    return float(feet)

def split_meter(row):
    # nan check
    if row is None:
        return None
    if row != row :
        return None
    if "”" in row:
        meter = row[row.rfind("”")+1:][:-2]
    elif "’" in row:
        meter = row[row.rfind("’")+1:][:-2]
    return float(meter)

def split_price(row):
    # nan check
    if row == "N/A €" :
        return None
    return int(row[6: row.rfind("€")].replace(" ", ""))

def split_ft2(row):
    # nan check
    if row != row :
        return None
    return float(row.split(" ft²")[0].strip())

def split_m2(row):
    # nan check
    if row != row :
        return None
    return float(row.split("ft²")[-1].split("m²")[0].strip())

def split_kg(row):
    # nan check
    if row != row :
        return None
    return float(row.split("lb")[-1].split("kg")[0].strip())

def split_liters(row,lit):
    if row != row :
        return None
    return float(row.split("gal")[-1].split(lit)[0].strip())

def split_displacement(row):
    if row != row :
        return None
    return float(row.split("ft²/T")[-1].split("m²/T")[0].strip())

def split_kgm(row):
    if row != row :
        return None
    return float(row.split("lb.ft")[-1].split("kg.m")[0].strip())

def split_hp(row):
    if row is None:
        return None
    if row != row :
        return None
    if len(row)==0:
        return None
    return float(row.split("HP")[0])

def remove_about(row):
    if row is None or row!=row:
        return None
    if 'About' in row:
        return float(row.split("About")[-1])
    else:
        return float(row)


# dataset["Engine(s) power [HP]"] = dataset["Engine(s) power"].apply(lambda x:int(x.split("HP")[0]) if x == x else None)
# dataset.drop("Engine(s) power",axis=1,inplace=True)

dataset["Number of hulls built"] = dataset["Number of hulls built"].apply(lambda x: remove_about(x))

dataset["price [€]"] = dataset["Standard public price ex. VAT (indicative only)"].apply(lambda x: split_price(x))
dataset.drop("Standard public price ex. VAT (indicative only)",axis=1,inplace=True)

dataset["Critical hull speed [knots]"] = dataset["Critical hull speed"].apply(lambda x:float(x.split("knots")[0]) if x == x else None)
dataset.drop("Critical hull speed",axis=1,inplace=True)

dataset["Ballast ratio%"] = dataset["Ballast ratio"].apply(lambda x:float(x.split("%")[0]) if x == x else None)
dataset.drop("Ballast ratio",axis=1,inplace=True)

dataset["French customs tonnage"] = dataset["French customs tonnage"].apply(lambda x:float(x.split("Tx")[0]) if x == x else None)
dataset.drop("French customs tonnage",axis=1,inplace=True)


for column in ["Upwind sail area to displacement","Downwind sail area to displacement"]:
    dataset[f"{column} [num]"] = dataset[column].apply(lambda x: split_displacement(x) )
    dataset.drop(column,axis=1,inplace=True)
    
for column in ["Righting moment @ 1°","Righting moment @ 30°"]:
    dataset[f"{column}"] = dataset[column].apply(lambda x: split_kgm(x) )
    dataset.drop(column,axis=1,inplace=True)

for column in dataset:
    if "(min./max.)" in column:
        dataset[column[:-11]+'min'] = dataset[column].apply(lambda x: x.split("/")[0].strip() if x == x else None)
        dataset[column[:-11]+'max'] = dataset[column].apply(lambda x: x.split("/")[1].strip() if x == x else None)
        # print(dataset[column[:-11]+'min'])
        # for index, value in dataset[column[:-11]+'min'].items():
        #     if value is not None and len(value)>0:
        #         print(value)
        #         print(float(value))
        try:
            dataset[column[:-11]+'min']= dataset[column[:-11]+'min'].apply(lambda x: float(x) if x is not None and len(x)>0 else None)
            dataset[column[:-11]+'max']= dataset[column[:-11]+'max'].apply(lambda x: float(x) if x is not None and len(x)>0 else None)
        except:
            print(column,'not int')
        dataset.drop(column,axis=1,inplace=True)


for column in dataset:
    mask=dataset[column].notna()
    # print(dataset[column][mask])
    # print(dataset[column][mask].iloc[0])
    d=dataset[column][mask].iloc[0]
    if isinstance(d, str) and column not in ["Chart table","Berth width (head/elbows/knees/feet)"]:
        # if "(min./max.)" in column:
        if column=="Berth width (head/feet)":
            continue
        if "m"==d[-1]:
            # print('meter', column)
            dataset[f"{column} [meter]"] = dataset[column].apply(lambda x: split_meter(x) )
            dataset.drop(column,axis=1,inplace=True)
        if "m²"==d[-2:]:
            # print('meter square',column)
            dataset[f"{column} [m²]"] = dataset[column].apply(lambda x: split_m2(x) )    
            dataset.drop(column,axis=1,inplace=True)
        if "liters"==d[-6:] or "Liters"==d[-6:]:
            print('liters',column)
            dataset[f"{column} [l]"] = dataset[column].apply(lambda x: split_liters(x,d[-6:]) )
            dataset.drop(column,axis=1,inplace=True)
        if "kg" == d[-2:]:
            # print('kg',column)
            dataset[f"{column} [kg]"] = dataset[column].apply(lambda x: split_kg(x) )
            dataset.drop(column,axis=1,inplace=True)
        if "HP" == d[-2:]:
            # print('hp',column)
            dataset[f"{column} [HP]"] = dataset[column].apply(lambda x: split_hp(x) )
            dataset.drop(column,axis=1,inplace=True)
