import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

# -------------- Loading Data ------------------
train = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("test.csv", dtype={"Age": np.float64}, )

train['Gender'] = train['Sex'].map({'female':0, 'male':1}).astype(int)

median_ages = np.zeros((2,3))

for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = train[(train['Gender'] == i) & \
                            (train['Pclass'] == j+1)]['Age'].dropna().median()

train['AgeFill'] = train['Age']

for i in range(0,2):
    for j in range(0,3):
         train.loc[(train.Age.isnull()) & (train.Gender == i) & (train.Pclass == j+1),'AgeFill'] = median_ages[i,j]
#print train.loc[train.Age.isnull(), ('Gender','Pclass','Age','AgeFill')].head(10)

train['AgeIsNull'] = pd.isnull(train.Age).astype(int)

def changeTitles(data, array, arg):
    for i in range(0,len(array)):
        data.loc[data.Title == array[i], 'Title'] = arg
    return data.Title;

# ------------- Feature Engineering ---------------

def featureEngineer(data):
    data['Gender'] = data['Sex'].map({'female':0, 'male':1}).astype(int)
    median_ages = np.zeros((2,3))

    for i in range(0, 2):
        for j in range(0, 3):
            median_ages[i,j] = data[(data['Gender'] == i) & \
                                (data['Pclass'] == j+1)]['Age'].dropna().median()
    
    data['AgeFill'] = data['Age']

    for i in range(0,2):
        for j in range(0,3):
             data.loc[(data.Age.isnull()) & (data.Gender == i) & (data.Pclass == j+1),'AgeFill'] = median_ages[i,j]
             
    data['FamilySize'] = data['SibSp'] + data['Parch']
    data['Title'] = data.Name.str.split(',',expand=True, n=2)[1]
    # print train.loc[647, 'Title']
    data['Title'] = data.Title.str.split('.',expand=True, n=2)[0].str[1:]
    # print(data.loc[train.Age.isnull(), ('Gender','Pclass','Age','AgeFill')].head(10))
    
    data['Title'] = changeTitles(data,['the Countess','Ms'],'Mrs')
    data['Title'] = changeTitles(data,['Col','Capt','Dona','Dr','Jonkheer',\
                                    'Major','Rev','Sir','Lady'],'Noble')
    data['Title'] = changeTitles(data,["Mlle", "Mme"],"Miss")
    # print train.Title.value_counts()

    data['Title'] = data.Title.map({'Mrs':1, 'Noble':2, 'Miss':3, 'Master':4, 'Mr':5})
    data['Embarked'] = data.Embarked.map({'S':1, 'C':2, 'Q':3})
    return data;
    
train = featureEngineer(train)
test = featureEngineer(test)

# ------------- Pre ML Preparation ---------------
#print train.dtypes[train.dtypes.map(lambda x: x=='object')]
#axis=0 : along the rows; axis=1: along the columns
train = train.dropna()
train = train.drop(['PassengerId','Name','Age','Sex','Ticket','Cabin','AgeIsNull','Fare'], axis=1)
y_test = test.drop(['PassengerId','Name','Age','Sex','Ticket','Cabin','Fare'], axis=1).copy()
# checking data
# print x_train.info()
# print x_train.isnull().sum()
# print x_test.info()
# print x_test.isnull().sum()
train = train.values
y_test = y_test.values

# ------------- Partition --------------
x_train, x_test = train_test_split(train, test_size=0.33)

# ------------- Building Random Forest -------------
forest = RandomForestClassifier(n_estimators = 100, max_features = 0.33, oob_score = True)
forest = forest.fit(x_train[:,1::],x_train[:,0] )
pred = forest.predict(y_test)
#print forest.score(x_test[:,1::],x_test[:,0])

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": pred})
submission.to_csv('titanic.csv', index=False)