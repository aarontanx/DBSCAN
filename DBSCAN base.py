from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhoutte_samples, silhouette_score
import numpy as np
import pandas as pd
import seaborn as sns

# Prep the data and create features - remember to normalize
# DF IS FEATURES IN THIS EXERCISE, NOT THE RAW DATA
# df_scaled = # df after normalized

# Plot corr plot (pair plot)
corr_plot = sns.pairplot(data = df_scaled)
corr_plot.savefig('corr_plot_fig.png')

# Plot kde plot
g = sns.PairGrid(df_scaled)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=3, legend= False)
g.savefig('pairgrid_fig.png')

# Determine how many clusters and separations of clusters
distance = []
silhouette_avg = []

# to test from 2-10 clusters
cluster = list(np.arange(11))[2:]
max_sil = 0
n_clus = 0

for i in cluster:
	# initialize clsuter with n_clusters value and random seed 123 for reproducibility
	kmeans = KMeans(n_clusters = i, random_state = 123).fit(df_scaled)
	cluster_labels = kmeans.predict(df_scaled)
	cluster_distances = kmeans.precompute_distances
	
	# Calculate avg silhoutte value for all samples
	# Gives a perspective into density and separation of formed clusters
	silhouette_avg = silhouette_score(df_scaled, cluster_labels)
	
	# Avg silhoutte score
	print(' '.join(["For n_clusters = ", str(i), "The average silhouette_score is : ", str(silhouette_avg)]))
	
	if silhouette_avg > max_sil:
		max_sil = silhouette_avg
		n_clus = i

print(' '.join(['n_cluster', str(n_clus), "with highest silhoutte_score : ", str(max_sil)]))



# create 10 different epsilon sizes and 40 different min sample size
eps = list(np.linspace(0.1, 0.4, 10))
min_samples = list(np.arange(80, 250, 40))
a = 0

df_results = pd.DataFrame(columns = ['iteration', 'eps', 'min_samples', 'error', 'num_error'])


# Loop through the EPS and min sample list to find optimal outlier size
for i in eps:
	population = df_scaled.shape[0]
	
	for j in min_samples:
		a += 1
		iteration = 'Iteration %0.0f' % a
		print('')
		print(iteration)
		
		db = DBSCAN(eps = i, min_samples = j, algorithm = 'auto', n_jobs = -1).fit(df_scaled)
		
		core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
		core_samples_mask[db.core_sample_indices_] = True
		labels = db.labels_
		
		# If no outlier will continue to run instead of stopping
		try:
			anomaly = pd.Series(list(labels)).value_counts()[-1] 
		except:
			anomaly = 0

		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		n_noise_ = list(labels).count(-1)
		
		error = anomaly / population
		
		if n_clusters_ == 0:
			print(' '.join(['eps = ', str(i), 'min, samples =', str(j)]))
			print('')
			continue
		else:
			print(' '.join(['eps= ', str(i), 'min_samples = ', str(j)]))
			print(' '.join(['Estimated number of clusters:', str(n_clusters_)]))
			print(' '.join(['Estimated number of noise points', str(n_noise_)]))
			print(' '.join(['Homogeneity:', str(round(metrics.homogeneity_score(labels_true, labels), 3))]))
			print(' '.join(['Completeness:', str(round(metrics.completeness_score(labels_true, labels), 3))]))
			print(' '.join(['V-measure:', str(round(metrics.v_measure_score(labels_true, labels), 3))]))
			print(' '.join(['Adjusted Rand Index:', str(round(metrics.adjusted_rand_score(labels_true, labels), 3))]))
			print(' '.join(['Percentage Outliers:', str(round(error*100, 3)), '%']))
			print('')
			
			param = pd.DataFrame({'iteration': [iteration_num],
                'eps': [i],
                'min_samples': [j],
                'error': [error],
                'num_error': [anomaly]})
				
			df_results = pd.concat([df_results, param])
		
##### PICK THE OUTLIER COUNT YOU WANT for sampling
# Rerun the code above without loop, replacing the value you want


## Label data
df_scaled['outlier'] = list(db.labels_)

# THE LABEL IS THE GROUP, IF ONLY WANT TO FLAG OUTLIERS ONLY USE THE CODE BELOW

fimm_summary.loc[fimm_summary.outlier >= 0, 'outlier'] = 'Group'
fimm_summary.loc[fimm_summary.outlier == -1, 'outlier'] = 'Outlier'
		
# Plot pair plots with outliers

outlier_plot_1 = sns.pairplot(data = df
    , palette =['lightblue', 'Red']
    , hue = 'outlier'
    , vars = [#### put in your list of features])
 
outlier_plot_1.fig.suptitle('Clustering', y = 1)
outlier_plot_1.savefig('outlier_plot.png')

df_outlier = fimm_summary.loc[fimm_summary['outlier'] == 'Outlier', : ]

# This part is raw data label
fimm_actual_RFM['outlier'] = list(db.labels_)
complete_df = pd.merge(fimm_actual_RFM, df_fimm, on = [#joining key]
    , how = 'outer')
    
    
 writer = pd.ExcelWriter(#location to save
    , engine='xlsxwriter')
 df.to_excel(writer, sheet_name='name')
 writer.save()
 