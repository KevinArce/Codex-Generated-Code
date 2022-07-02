#Create an implementation of the PCA algorithm in Python using pandas, sklearn and matplotlib to train and test a dataset called heart.csv. You can use sklearn.descomposition, sklearn.linear_model, sklearn.preprocessing and sklearn.model_selection. The dataset contains the following columns:

#age: age in years

#sex: sex (1 = male; 0 = female)

#cp: chest pain type

#trestbps: resting blood pressure (in mm Hg on admission to the hospital)

#chol: serum cholestoral in mg/dl

#fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

#restecg: resting electrocardiographic results

#thalach: maximum heart rate achieved

#exang: exercise induced angina (1 = yes; 0 = no)

#oldpeak: ST depression induced by exercise relative to rest

#slope: the slope of the peak exercise ST segment

#ca: number of major vessels (0-3) colored by flourosopy

#thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

#target: 1 or 0

#The goal is to predict the target column.

#You can use the following code to load the dataset:

import pandas as pd

df = pd.read_csv('heart.csv')

df.head()

#You can use the following code to split the dataset into train and test:

from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)

y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#You can use the following code to scale the dataset:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

#You can use the following code to create a PCA instance:

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

#You can use the following code to fit the PCA instance to the train dataset:

pca.fit(X_train)

#You can use the following code to transform the train and test datasets:

X_train = pca.transform(X_train)

X_test = pca.transform(X_test)

# You can use the following code to create a Logistic Regression instance:

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# You can use the following code to fit the Logistic Regression instance to the train dataset:

model.fit(X_train, y_train)

# You can use the following code to predict the test dataset:

y_pred = model.predict(X_test)

# You can use the following code to evaluate the model:

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

# You can use the following code to visualize the train dataset:

import matplotlib.pyplot as plt

plt.scatter(X_train[:,0], X_train[:,1], c=y_train)

plt.xlabel