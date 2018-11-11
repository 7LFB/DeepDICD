# visualization using tsne algorithm
# visualization for analysis
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.datasets import fetch_mldata
import pandas as pd
from sklearn.manifold import TSNE
import time
from matplotlib import pyplot as plt
import colorsys
import pdb

def visualization_using_tsne(X,y,n_class,num_samples,save_name='./output/cluster_result.png'):

	rndperm=np.random.permutation(X.shape[0])

	hsv_tuples = [(x / n_class, 1., 1.)
	              for x in range(n_class)]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(
	    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
	        colors))

	# T-SNE
	n_sne=num_samples
	
	tsne=TSNE(n_components=2,verbose=1,perplexity=40,n_iter=300)
	tsne_results=tsne.fit_transform(X[rndperm[:n_sne],:])
	y_select=y[rndperm[:n_sne]]
	
	# Visualization
	# plt.figure(figsize=(6,5))
	
	for i in range(int(n_class)):
	    plt.scatter(tsne_results[y_select==i,0],tsne_results[y_select==i,1],cmap=colors[i],label=str(i))
	
	plt.yticks([])
	plt.xticks([])
	plt.legend(loc='upper left')
	plt.savefig(save_name)
