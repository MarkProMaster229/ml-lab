import pandas as pd
df = pd.read_csv('/home/chelovek/Downloads/7890/train.csv')
df = df.drop(columns=['PassengerId', 'Ticket'])
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df = df.drop(columns=['SibSp', 'Parch'])

df['Cabin'] = df['Cabin'].fillna('X')

df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
df = df.drop(columns=['Name'])

df['Cabin'] = df['Cabin'].fillna('X').astype(str).str[0]


df['Age'] = df['Age'].fillna(df['Age'].median())

print(df.head())

X = df.drop(columns=['Survived'])
y = df['Survived']

X.to_csv('titanic_features_X.csv', index=False)
y.to_csv('titanic_target_y.csv', index=False)
