



import numpy as np
import pandas as pd
import PIL
from PIL import Image
import sklearn
import glob
from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




def initialization():
    neurons = None
    R = 1
    lr = 1
    epochs = 100
    R_update = 20
    lr_update = 10
    converge_rate = 0.0001
    return neurons, R, lr,  epochs, R_update, lr_update, converge_rate



def LoadData():
    dataframe = pd.read_excel('Data.xlsx', header=None)
    X = dataframe.iloc[:, 0:-1]
    Y = dataframe.iloc[:, -1]

    # print(np.shape(X))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)#, random_state=42
    # print(np.shape(X_train))

    return X_train, X_test


def normalization(features, min_max_range):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=min_max_range)
    features = min_max_scaler.fit_transform(features)
    return features


def SOM_Architecture(row_size, column_size, features_num):
    global neurons
    neurons = np.random.uniform(low=-1, high=1, size=(row_size, column_size, features_num))
    sorted_indexes = np.argsort(np.linalg.norm(neurons, axis=2))
    for row in range(row_size):
        neurons[row] = neurons[row, sorted_indexes[row]]
    return



def Training(lr, R, dataset):
    global neurons
    (row_num, column_num, feature_num) = neurons.shape
    max_delta_w = 0

    data = np.copy(dataset)
    np.random.shuffle(data)
    for sample in data:
        winner_row = 0
        winner_column = 0
        winner_dist = -1
        for row in range(row_num):
            for column in range(column_num):
                neuron = neurons[row][column]
                dist = np.linalg.norm(sample - neuron)
                if dist < winner_dist or winner_dist == -1:
                    winner_dist = dist
                    winner_row = row
                    winner_column = column
        for row in range(winner_row - R, winner_row + R + 1):
            for column in range(winner_column - R, winner_column + R + 1):
                if 0 <= row < row_num and 0 <= column < column_num:
                    delta_w = lr * (sample - neurons[row][column])
                    neurons[row][column] += delta_w
                    if np.linalg.norm(delta_w) > max_delta_w:
                        max_delta_w = np.linalg.norm(delta_w)
    return max_delta_w





def Weights(dataset, R, lr, epochs, R_update, lr_update, converge_rate, colors):
    global neurons
    for epoch_num in range(1, epochs + 1):
        max_delta_w = Training(lr, R, dataset)
        if max_delta_w < converge_rate:
            print('epoch_number:', epoch_num)
            return
        if epoch_num % R_update == 0 and R > 0:
            R -= 1
        if epoch_num % lr_update == 0:
            lr *= 0.9
    print('epoch_number:', epoch_num)
    return

def testData(testdata, xcenters, ycenters):
    indx = 0
    s = []
    testdata = np.array(testdata)
    m, n = np.shape(testdata)
    r = len(xcenters)
    for i in range(m):
        for j in range(r):
            s.append(np.sqrt((testdata[i][0]-xcenters[j])**2 + (testdata[i][1]-ycenters[j])**2))
        mnindex = np.argmin(s)
        indx += s[mnindex]

    return indx







def visualization(dataset, colors):
    global neurons
    (row_num, column_num, feature_num) = neurons.shape
    clusters_list = []
    for row in range(row_num):
        for column in range(column_num):
            clusters_list.append([neurons[row][column], [], []])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for sample in dataset:
        min_cluster_index = -1
        min_cluster_dist = -1
        for cluster_index, cluster in enumerate(clusters_list):
            dist = np.linalg.norm(sample - cluster[0])
            if dist < min_cluster_dist or min_cluster_dist == -1:
                min_cluster_dist = dist
                min_cluster_index = cluster_index
        clusters_list[min_cluster_index][1].append(sample[0])
        clusters_list[min_cluster_index][2].append(sample[1])
        ax1.scatter(sample[0], sample[1], s=10, marker="o", c=colors[min_cluster_index])
    x_centers = []
    y_centers = []
    for cluster_index, cluster in enumerate(clusters_list):
        x_centers.append(cluster[0][0])
        y_centers.append(cluster[0][1])

    ax1.scatter(x_centers, y_centers, s=30, marker="*", c=colors)

    plt.show()
    print("x_centers are:", x_centers)
    print("y_centers are:", y_centers)


    return x_centers, y_centers


neurons, R, lr,  epochs, R_update, lr_update, converge_rate = initialization()
Xtrain, Xtest = LoadData()
values = Xtrain.values

values = normalization(values, (-1, 1))

valuesTest = Xtest.values
valuesTest = normalization(valuesTest, (-1, 1))

SOM_Architecture(10, 10, len(values[0]))
colors = [np.random.rand(3, ) for x in range(neurons.shape[0] * neurons.shape[1])]

Weights(values, R, lr, epochs, R_update, lr_update, converge_rate, colors)
xcenters, ycenters = visualization(values, colors)
indx = testData(valuesTest, xcenters, ycenters)


print("#################################################")
print("#################################################")

print("Index for this Radius and Learning_rate is:", indx)



















