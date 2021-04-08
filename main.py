import os
from Utils import DataLoader
from Model import KMeans
import matplotlib.pyplot as plt
import numpy as np

mnist_data = DataLoader("dataset")

tr_data, tr_class_labels, tr_subclass_labels = mnist_data.loaddata()

print(tr_data.shape)

mnist_data.plot_imgs(tr_data,25,True)

Kmeans = KMeans(n_clusters=10,max_iter=200)

Kmeans.fit(tr_data,tr_class_labels)

mnist_data.plot_imgs(Kmeans.centroids,len(kmeans.centeroids))

plt.show()

for key,data in list(Kmeans.clusters['data'].items()):
    print('Cluster:',key,'Label:',Kmeans.clusters_labels[key])

    mnist_data.plot_imgs(data[:min(25,data.shape[0])],min(25,data.shape[0]))

print('[cluster_label,no_occurrence_of_label,total_sample_in_cluster,cluster_accuracy]\n',Kmeans.clusters_info)
print('Accuracy:',Kmeans.accuracy)
