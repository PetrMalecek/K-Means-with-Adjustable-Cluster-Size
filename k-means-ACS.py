import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# ======= Controls ======= #
K = 3                         # number of clusters, example works for up to 5 clusters
adj = np.array([1, 1, 1])     # cluster size ratios
stop = 1                      # stopping criterion: % deviation from the desired number of items in each cluster
seed = 1234                   # seed number


# ======= Generate sample data ======= #
np.random.seed(seed)
centers = [[1, 1], [-1, -1], [1, -1], [0.8, -0.8]]           # generated centers
locs, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.65)


# ======= Calculate k-means ======= #
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=K, init='k-means++', random_state=0).fit(locs)
centers_kmeans = kmeans.cluster_centers_   # locations of centroids
labels_kmeans = kmeans.labels_   # assign cases to centroids
cases_kmeans = np.array([sum(labels_kmeans == k) for k in range(K)])   # number of items


# ======= Adjustable cluster size k-means algorithm ======= #
adj = sum(adj)/(K*adj)                     # standardize adjustment weights
weights = adj*cases_kmeans/(len(locs)/K)   # initialize weights
C = np.ones(K)
labels_eq = np.ones(len(locs))
cases_eq = cases_kmeans                    # initialize for the loop
iters = 0                                  # traces the number of iterations

while np.max(100*np.abs(adj*cases_eq-(len(locs)/K))/(len(locs)/K)) > stop:
    for j in range(len(locs)):
        for i in range(K):
            # assign cases to k-means centroids
            C[i] = weights[i]*(np.dot(locs[j]-centers_kmeans[i],locs[j]-centers_kmeans[i])) # distance measure
        labels_eq[j] = np.argmin(C)
    cases_eq = np.array([sum(labels_eq == k) for k in range(K)])   # number of cases
    weights = adj*weights*cases_eq/(len(locs)/K)
    iters = iters + 1

centers_eq = np.array([locs[labels_eq == k].mean(axis = 0) for k in range(K)]) # shift centroids
cases_eq = np.array([sum(labels_eq == k) for k in range(K)])


# ======= Plot and print results ======= #
colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#852087', '#2D262D']
def colornames(argument):
    switcher = {0: "Blue", 1: "Orange", 2: "Green", 3: "Violet", 4: "Black"}
    return switcher.get(argument)

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('KMeans')
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('ACS KMeans')

print('Center color | Cluster size (ACS) | Cluster size (KMeans)')
print('=========================================================')

for k, col in zip(range(K), colors):
    my_members = labels_kmeans == k
    ax1.plot(locs[my_members, 0], locs[my_members, 1], 'w', linestyle='none',
             markerfacecolor=col, marker='.')
    ax1.plot(centers_kmeans[k, 0], centers_kmeans[k, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=10)
    my_members = labels_eq == k
    ax2.plot(locs[my_members, 0], locs[my_members, 1], 'w', linestyle='none',
            markerfacecolor=col, marker='.')
    ax2.plot(centers_eq[k,0], centers_eq[k,1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=10)
    print(colornames(k),' '*(11-len(colornames(k))),'|', sum(labels_eq == k),' '*(17-len(str(sum(labels_eq == k)))),'|',
            sum(labels_kmeans == k))
for k, col in zip(range(K), colors):
    ax2.plot(centers_kmeans[k,0], centers_kmeans[k,1], 'x', markeredgecolor='k', markersize=20)

print('=========================================================')
print('Number of iterations:', iters)

dot_patch = mlines.Line2D([], [], markerfacecolor='#d1d1e0', markeredgecolor='k', markersize=10, marker='o',linestyle = 'None')
cross_patch = mlines.Line2D([], [], markeredgecolor='k', markersize=10, marker='x',linestyle = 'None')
ax1.legend([dot_patch],['KMeans'])
ax2.legend([cross_patch, dot_patch],['KMeans','ACS KMeans'])
ax1.tick_params(labelsize=8)
plt.show()
