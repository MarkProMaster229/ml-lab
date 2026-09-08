from catboost import CatBoostClassifier
import pandas as pd
X = pd.read_csv('/home/chelovek/Music/modelWork/titanic_features_X.csv')
y = pd.read_csv('/home/chelovek/Music/modelWork/titanic_target_y.csv')

y = y['Survived'] 

cat_features = ['Sex', 'Cabin', 'Embarked', 'Title']
X[cat_features] = X[cat_features].fillna('Missing')

model = CatBoostClassifier(iterations=100, verbose=100, random_seed=42)
model.fit(X, y, cat_features=cat_features)

model.save_model('/home/chelovek/Music/modelWork/catboost_titanic_model.cbm')
