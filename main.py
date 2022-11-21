import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

(X_train, y_train), (X_pred, y_pred) = mnist.load_data()
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=2022)

dim = 784  # 28*28
X_train = X_train.reshape(len(X_train), dim)
X_test = X_test.reshape(len(X_test), dim)

pca = PCA(svd_solver='full')
modelPCA = pca.fit(X_train)

X_train = modelPCA.transform(X_train)

explained_variance = np.round(np.cumsum(pca.explained_variance_ratio_), 3)

# Пункт 1
component_count = -1
for k, v in enumerate(explained_variance):
    if v > 0.9 and k + 1 > component_count:
        component_count = k + 1
        break
print(component_count)
pca = PCA(n_components=component_count, svd_solver='full')
modelPCA = pca.fit(X_train)

X_train = modelPCA.transform(X_train)

plt.plot(np.arange(len(explained_variance)), explained_variance, ls='-')

# Пункт 2
tree = RandomForestClassifier(criterion='gini', min_samples_leaf=10, max_depth=20, n_estimators=10, random_state=2020)
clf = OneVsRestClassifier(tree).fit(X_train, y_train)

modelPCA = pca.fit(X_test)
X_test_original = X_test
X_test = modelPCA.transform(X_test)
y_pred = clf.predict(X_test)
CM = confusion_matrix(y_test, y_pred)

print(CM[1][1])

# Пункт 3
from random import sample

indexes = sample(range(len(X_test)), k=5)
print(f"Random picked object indexes {indexes}")
objects = list()
for i in indexes:
    objects.append(X_test[i])
i = 0
for prob in clf.predict_proba(objects):
    print(f"Object id {indexes[i] + 1}'s predicted class is {y_pred[indexes[i]]}, probability: {round(max(prob), 3)}")
    i += 1

# Пункт 4
print(classification_report(y_test, y_pred))