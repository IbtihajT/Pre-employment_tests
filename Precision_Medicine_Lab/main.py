# Required Libararies
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from mpl_toolkits.mplot3d import axes3d, Axes3D
# Random Seed for reprouction of results
seed = 7

# Function to check if there are any categorical variables in featureset


def is_categorical(array_like):
    return array_like.dtype.name == 'category'


# Load the data with tab-separation
data = pd.read_csv("data/Dmel_matrix__pml.txt", delimiter="\t", header=None)
print("Dataset Loaded")

# Check if data has missing values
print("Missing values in data: {}".format(data.isnull().sum().sum()))

# Capture the Dependent and Independent Variable
X = data.iloc[:, 1:-1].values  # Independent Varibale with 28 features
y = data.iloc[:, -1].values  # Dependent Variable with 2 classes

# Checking Categorical variables
print("Categorical Variables present: {}".format(is_categorical(X)))

# Scale the Values
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Plot the data

# Will apply principle component analysis (PCA) for dimentionality reduction
pca = PCA(n_components=3)  # For 3 Dimenstions
pca_result = pca.fit_transform(X)
print('Explained variation per principal component: {}'.format(
    pca.explained_variance_ratio_))

# 3D Plot
ax = plt.figure(figsize=(15, 10)).gca(projection='3d')
ax.scatter(
    xs=pca_result[:, 0],
    ys=pca_result[:, 1],
    zs=pca_result[:, 2],
    c=y
    # cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()

# Split the data in train_set and test_set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=seed)

# Model Training

# 1) Logistic Regression
classifier = LogisticRegression(random_state=seed)
print()
print("----------Training Logistic Regression----------")
print()
classifier.fit(X_train, y_train)

# Pridiciton of Testset
y_pred = classifier.predict(X_test)

# Confusion Matrix
acc_score = accuracy_score(y_test, y_pred)
apc = average_precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print('{:<23}: {:>10.2f}'.format('Accuracy Score', acc_score), sep='')
print('{:<23}: {:>10.2f}'.format('Avarage Precision Score', apc), sep='')
print('{:<23}: {:>10.2f}'.format('f1 Score:', f1), sep='')
print('{:<23}: {:>10.2f}'.format('Precision Score:', ps), sep='')
print('{:<23}: {:>10.2f}'.format('Recall Score:', rs), sep='')
print()
print('Confusion Matrix: {}'.format(cm))
print()

# 2) k-NN
classifier = KNeighborsClassifier(n_neighbors=5)  # using 5 neighbors
print()
print("----------Training k-NN----------")
print()
classifier.fit(X_train, y_train)

# Pridiciton of Testset
y_pred = classifier.predict(X_test)

# Confusion Matrix
acc_score = accuracy_score(y_test, y_pred)
apc = average_precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print('{:<23}: {:>10.2f}'.format('Accuracy Score', acc_score), sep='')
print('{:<23}: {:>10.2f}'.format('Avarage Precision Score', apc), sep='')
print('{:<23}: {:>10.2f}'.format('f1 Score:', f1), sep='')
print('{:<23}: {:>10.2f}'.format('Precision Score:', ps), sep='')
print('{:<23}: {:>10.2f}'.format('Recall Score:', rs), sep='')
print()
print('Confusion Matrix: {}'.format(cm))
print()

# 3) Support Vector Mahcine (SVM)
classifier = SVC(random_state=seed)
print()
print("----------Training SVM----------")
print()
classifier.fit(X_train, y_train)

# Pridiciton of Testset
y_pred = classifier.predict(X_test)

# Confusion Matrix
acc_score = accuracy_score(y_test, y_pred)
apc = average_precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print('{:<23}: {:>10.2f}'.format('Accuracy Score', acc_score), sep='')
print('{:<23}: {:>10.2f}'.format('Avarage Precision Score', apc), sep='')
print('{:<23}: {:>10.2f}'.format('f1 Score:', f1), sep='')
print('{:<23}: {:>10.2f}'.format('Precision Score:', ps), sep='')
print('{:<23}: {:>10.2f}'.format('Recall Score:', rs), sep='')
print()
print('Confusion Matrix: {}'.format(cm))
print()

# 4) Decision Trees
classifier = DecisionTreeClassifier(random_state=seed)
print()
print("----------Training Decision Trees----------")
print()
classifier.fit(X_train, y_train)

# Pridiciton of Testset
y_pred = classifier.predict(X_test)

# Confusion Matrix
acc_score = accuracy_score(y_test, y_pred)
apc = average_precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print('{:<23}: {:>10.2f}'.format('Accuracy Score', acc_score), sep='')
print('{:<23}: {:>10.2f}'.format('Avarage Precision Score', apc), sep='')
print('{:<23}: {:>10.2f}'.format('f1 Score:', f1), sep='')
print('{:<23}: {:>10.2f}'.format('Precision Score:', ps), sep='')
print('{:<23}: {:>10.2f}'.format('Recall Score:', rs), sep='')
print()
print('Confusion Matrix: {}'.format(cm))
print()

# 5) Random Forest
classifier = RandomForestClassifier(random_state=0)
print()
print("----------Training Random Forest----------")
print()
classifier.fit(X_train, y_train)

# Pridiciton of Testset
y_pred = classifier.predict(X_test)

# Confusion Matrix
acc_score = accuracy_score(y_test, y_pred)
apc = average_precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print('{:<23}: {:>10.2f}'.format('Accuracy Score', acc_score), sep='')
print('{:<23}: {:>10.2f}'.format('Avarage Precision Score', apc), sep='')
print('{:<23}: {:>10.2f}'.format('f1 Score:', f1), sep='')
print('{:<23}: {:>10.2f}'.format('Precision Score:', ps), sep='')
print('{:<23}: {:>10.2f}'.format('Recall Score:', rs), sep='')
print()
print('Confusion Matrix: {}'.format(cm))
print()

# 6) Naive Bayes
classifier = GaussianNB()
print()
print("----------Training Naive Bayes----------")
print()
classifier.fit(X_train, y_train)

# Pridiciton of Testset
y_pred = classifier.predict(X_test)

# Confusion Matrix
acc_score = accuracy_score(y_test, y_pred)
apc = average_precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print('{:<23}: {:>10.2f}'.format('Accuracy Score', acc_score), sep='')
print('{:<23}: {:>10.2f}'.format('Avarage Precision Score', apc), sep='')
print('{:<23}: {:>10.2f}'.format('f1 Score:', f1), sep='')
print('{:<23}: {:>10.2f}'.format('Precision Score:', ps), sep='')
print('{:<23}: {:>10.2f}'.format('Recall Score:', rs), sep='')
print()
print('Confusion Matrix: {}'.format(cm))
print()
