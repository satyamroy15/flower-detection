from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm

# import data
data = pd.read_csv('IRIS.csv')
# sparte date as classes to visualize
setosa = data[data['species'] == 'Iris-setosa']
versicolor = data[data['species'] == 'Iris-versicolor']
virginica = data[data['species'] == 'Iris-virginica']

# visualize all data together
plt.figure()
fig, ax = plt.subplots(1, 2, figsize=(21, 10))
# visualize with sepal
setosa.plot(x="SepalLengthCm", y="SepalWidthCm",
            kind="scatter", ax=ax[0], label='setosa', color='r')
versicolor.plot(x="SepalLengthCm", y="SepalWidthCm",
                kind="scatter", ax=ax[0], label='versicolor', color='b')
virginica.plot(x="SepalLengthCm", y="SepalWidthCm",
               kind="scatter", ax=ax[0], label='virginica', color='g')
# visualize with petal
setosa.plot(x="PetalLengthCm", y="PetalWidthCm",
            kind="scatter", ax=ax[1], label='setosa', color='r')
versicolor.plot(x="PetalLengthCm", y="PetalWidthCm",
                kind="scatter", ax=ax[1], label='versicolor', color='b')
virginica.plot(x="PetalLengthCm", y="PetalWidthCm",
               kind="scatter", ax=ax[1], label='virginica', color='g')

ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
ax[1].set(title='Petal Comparasion', ylabel='petal-width')

# plt.show()
# sparte data to train and test
x = data.iloc[:, :-1]
y = data.iloc[:, 4]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
# initialize svm
clf = svm.SVC(gamma='scale')
# initialize model
model = clf.fit(x_train, y_train)
# get model accuracy
predicted_data = model.predict(x_test)


# loop for predictedData to model accuracy
accuracy = 0
for i, d in enumerate(predicted_data):
    if (y_test.iloc[i] == d):
        accuracy += 1/len(predicted_data) * 100  # get accuracy

print("model accuracy : ", accuracy)
