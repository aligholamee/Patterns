import numpy as np
import scipy.cluster.hierarchy as hac
import matplotlib.pyplot as plt

def augmented_dendrogram(*args, **kwargs):

    ddata = hac.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        for i, d in zip(ddata['icoord'], ddata['dcoord']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            plt.plot(x, y, 'ro')
            plt.annotate("%.3g" % y, (x, y), xytext=(0, -8),
                         textcoords='offset points',
                         va='top', ha='center')

    return ddata

data = np.array([
    [1, 1],
    [2, 1],
    [5, 4],
    [6, 5],
    [6.5, 6]
])

distance_matrix = hac.linkage(data, metric='euclidean', method='centroid')
show_leaf_counts = False
ddata = augmented_dendrogram(distance_matrix,
               color_threshold=1,
               p=6,
               truncate_mode='lastp',
               show_leaf_counts=show_leaf_counts,
               )
plt.title("show_leaf_counts = %s" % show_leaf_counts)
plt.show()