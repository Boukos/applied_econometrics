from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

# Loading some example data
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target

print y

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=1)
clf2 = DecisionTreeClassifier(max_depth=4)
clf3 = SVC(kernel='rbf',random_state=0, gamma=1.0, C=1.0)
eclf = SVC(kernel='rbf',random_state=0, gamma=100.0, C=1.0)

"""
# Training classifiers
clf1 = RandomForestClassifier(max_depth=1)
clf2 = DecisionTreeClassifier(max_depth=2)
clf3 = DecisionTreeClassifier(max_depth=5)
eclf = DecisionTreeClassifier(max_depth=10)

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3)],
                        voting='soft', weights=[2, 1, 2])
"""


clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
eclf.fit(X, y)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))
i = 1
for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1, clf2, clf3, eclf],
                        ['Underfit', 'Good compromise',
                         'Beginning to overfit', 'Heavily overfit']):
   
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    axarr[idx[0], idx[1]].set_title(tt)
    if i == 1 or i == 3:
        axarr[idx[0], idx[1]].set_ylabel("Petal length")
    if i == 3 or i == 4:
        axarr[idx[0], idx[1]].set_xlabel("Petal width")
        
    if i == 1:
         axarr[idx[0], idx[1]].legend()
    
    i +=1
    
plt.show()
