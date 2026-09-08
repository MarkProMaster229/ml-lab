import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('secret_math_pattern.csv')

X = df[['Feature_X', 'Feature_Y', 'Feature_Z_Noise']]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=12, random_seed=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Точность (Accuracy) на скрытом тесте: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
print("Детальный отчет по классам:")
print(classification_report(y_test, predictions))



importances = model.feature_importances_
print("Важность признаков по мнению Случайного леса:")
for name, importance in zip(X.columns, importances):
    print(f"  {name}: {importance:.4f}")
