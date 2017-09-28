import scipy.io
from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time
import matplotlib.pyplot as plt

### reading data
X = scipy.io.mmread("test.mtx.txt")

### number of clusters
kValues = range(2,8)
print kValues
timeSpendForEachK = []
numberOfDocumentsInEachCluster = {}

for k in kValues:
	### Appyling Kmeans on X
	km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1,
	                verbose=False, random_state=8)

	print("Clustering sparse data with %s" % km)
	t0 = time()
	km.fit(X)
	timeSpendForEachK.append(time() - t0)
	# print("done in %0.3fs" % (time() - t0))

	### printing the number of documents in each label
	dictOfDocumentsInEachLable = {}
	for i in range(k):	
		dictOfDocumentsInEachLable[i] = len([label for label in km.labels_ if label == i])
	numberOfDocumentsInEachCluster[k] = dictOfDocumentsInEachLable

### plotting Time Spend vs K Values

print ("plotting Time Spend vs K Values")
XValues = []
YValues = []

XValues = kValues
YValues = timeSpendForEachK
plt.figure(4)
plt.plot(XValues, YValues, 'ro')
plt.axis([0,8,0,2])
plt.show()

### Plotting Number of documents in each cluster vs cluster number(label)

print ("Plotting Number of documents in each cluster vs cluster")
XValues = []
YValues = []

listOfColors = ['r-','b-','g-','y-','k-','m-']
plt.figure(5)

for k in kValues:
	listOfDocumentsInEachk = []
	for lable in range(k):	
		listOfDocumentsInEachk.append(numberOfDocumentsInEachCluster[k][lable])
	XValues.append(range(k))
	YValues.append(listOfDocumentsInEachk)

plt.plot(XValues[0],YValues[0],listOfColors[0], XValues[1],YValues[1],listOfColors[1], \
	XValues[2],YValues[2],listOfColors[2], XValues[3],YValues[3],listOfColors[3], \
	XValues[4],YValues[4],listOfColors[4], XValues[5],YValues[5],listOfColors[5])
plt.axis([0,8,0,2225])
plt.show()

