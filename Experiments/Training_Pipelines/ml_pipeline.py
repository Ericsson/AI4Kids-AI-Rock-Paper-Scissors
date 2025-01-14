from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle
from ml_pipeline_utils import metric_values, Data_load

print('Loading data from files ...')
X, y = Data_load(max_items_per_class=10000)
data = [X, y]

# Saving data to disk
with open('data_flipped_dt_zs.pkl', 'wb') as output:  # Overwrites any existing file.
    pickle.dump(data, output)
print('Data saved')

# Loading data from disk
with open('data_flipped_dt_zs.pkl', 'rb') as input:
    data = pickle.load(input)
print('Data loaded.')

[X, y] = data

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Model creation
print('Training model ...')
# clf = DecisionTreeClassifier()
clf = RandomForestClassifier(n_estimators=100, n_jobs=8)
print('Model trained.')

# Model fit
clf.fit(X_train, y_train)

# Saving model to disk
with open('model_updated_dt_zs.pkl', 'wb') as output:  # Overwrites any existing file.
    pickle.dump(clf, output)
print('Model saved')

# Loading model from disk
with open('model_updated_dt_zs.pkl', 'rb') as input:
    clf = pickle.load(input)
print('Model loaded')

print('Testing model ...')
# Model test set prediction
y_pred = clf.predict(X_test)
print('Model testd.')

# Calculating and printing metrics
metric_values(y_test, y_pred)

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred, normalize='true')

plt.matshow(cm, cmap='Blues')
for (i, j), z in np.ndenumerate(cm):
    plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')
plt.show()


# # Decision tree model plot
# plt.figure()
# plot_tree(clf, filled=True)
# plt.title("Decision tree trained on all the iris features")
# plt.show()
