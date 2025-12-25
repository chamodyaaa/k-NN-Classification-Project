import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']
df = pd.read_csv('chronic_kidney_disease.csv', skiprows=29, names=columns, on_bad_lines='skip')
#df.head()

df.replace('?', np.nan, inplace=True)

df = df.dropna()

df = df.drop_duplicates()

# Numerical columns
num_cols = ['age','bp','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc']
for col in num_cols:
    df[col] = df[col].astype(float)  # convert to float first
    df[col].fillna(df[col].median(), inplace=True)

# Categorical columns
cat_cols = ['sg','al','su','rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','class']
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)




# Binary columns
binary_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane','class']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# Nominal with multiple values (sg, al, su)
df = pd.get_dummies(df, columns=['sg','al','su'])




scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


from sklearn.decomposition import PCA

pca = PCA(n_components=10)  # reduce to 10 features
df_pca = pca.fit_transform(df.drop('class', axis=1))


df.head()




# Save cleaned dataset to a new CSV file
df.to_csv('dataset_cleaned.csv', index=False)

print("Cleaned dataset saved successfully!")
print(df.head())

print(df.shape)  # e.g., (400, 25)
print(df.info())
print(df.columns)


df = pd.read_csv('dataset_cleaned.csv')
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))


accuracies = []
for k in range(1, 51):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Best k
best_k = np.argmax(accuracies) + 1
print("Best k:", best_k)

#Graphing accuracy vs k
plt.figure(figsize=(10,6))
plt.plot(range(1, 51), accuracies, marker='o', linestyle='--', color='b')
plt.title('k-NN Accuracy vs k')
plt.xlabel('Number of Neighbors: k')
plt.ylabel('Accuracy')
plt.xticks(range(1, 51))
plt.grid(True)
plt.show()