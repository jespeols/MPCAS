from matplotlib.colors import to_rgb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from classification_functions import *

# import clustering methods
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture # Gaussian Mixture Model

# import metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(X, y_pred, y_true, visualize=False, compute_scores=True, uses_noise=False):
    if compute_scores:
        if uses_noise:
            np.delete(y_pred, np.where(y_pred == -1)) # remove noise
            
        silh_score = silhouette_score(X, y_pred)
        db_score = davies_bouldin_score(X, y_pred)
        ch_score = calinski_harabasz_score(X, y_pred)    
        
        scores = [silh_score, db_score, ch_score]
    
    if visualize:
        tsne = TSNE(n_components=2, random_state=1234)
        tsne_results = pd.DataFrame(tsne.fit_transform(X), columns=['t-SNE 1', 't-SNE 2'])
        
        plt.figure(figsize=(14, 6))
        plt.ioff()
        color_palette = sns.color_palette('viridis', np.maximum(len(np.unique(y_pred)), len(np.unique(y_true)))+1)
        
        plt.subplot(1, 2, 1)
        plt.title('true labels', fontsize=14)
        sns.scatterplot(x='t-SNE 1', y='t-SNE 2', hue=y_true, data=tsne_results, palette=color_palette, legend='full', alpha=0.9)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        plt.subplot(1, 2, 2)
        if uses_noise:
            y_pred[y_pred == -1] = np.max(y_pred) + 1
            color_palette = sns.color_palette('viridis', len(np.unique(y_pred))-1) + [to_rgb('gray')]
        
        plt.title('clustering', fontsize=14)
        ax = sns.scatterplot(x='t-SNE 1', y='t-SNE 2', hue=y_pred+1, data=tsne_results, palette=color_palette, legend='full', alpha=0.9)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        if uses_noise:
            handles, labels = ax.get_legend_handles_labels()
            labels[-1] = 'noise'
            plt.legend(handles, labels)

        plt.show()
    
    return scores if compute_scores else None

def evaluate_clustering_multiple_runs(X, y_true, method, n_clusters_list, n_runs=5):
    scores_mean_list = []
    scores_std_list = []
    for n_clusters in n_clusters_list: 
        scores_runs = []
        for _ in range(n_runs):
            if method == 'kmeans':
                clm = KMeans(n_clusters=n_clusters).fit(X)
                y_pred = clm.labels_
            elif method == 'spectral':
                clm = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_init=10, n_jobs=-1).fit(X)
                y_pred = clm.labels_
            elif method == 'gmm':
                clm = GaussianMixture(n_components=n_clusters).fit(X)
                y_pred = clm.predict(X)
            
            scores_runs.append(evaluate_clustering(X, y_pred, y_true))

        scores_mean_list.append(np.mean(scores_runs, axis=0))
        scores_std_list.append(np.std(scores_runs, axis=0))
        
    scores_mean_list = np.array(scores_mean_list)
    scores_std_list = np.array(scores_std_list)
    
    return scores_mean_list, scores_std_list

def compare_clustering_methods(X, y_true, methods, n_clusters_list=[2, 3, 4, 5, 6, 7, 8, 9, 10], n_runs=5, visualize=False, path=None):
    
    df_scores_list = []
    df_scores_PCA_list = []
    df_scores_tsne_list = []
    for method in methods:
        print(f"Method: {method}")
        scores_mean_list, scores_std_list = evaluate_clustering_multiple_runs(X, y_true, method, n_clusters_list, n_runs=n_runs)

        df_dict = {"silhouette (mean)": scores_mean_list[:, 0], "silhouette (std)": scores_std_list[:, 0],
                "davies_bouldin (mean)": scores_mean_list[:, 1], "davies_bouldin (std)": scores_std_list[:, 1],
                "calinski_harabasz (mean)": scores_mean_list[:, 2], "calinski_harabasz (std)": scores_std_list[:, 2]}

        df_scores = pd.DataFrame(df_dict, index=n_clusters_list)
        df_scores.index.name = "n_clusters"
        df_scores_list.append(df_scores)
        
        # Do the same with PCA
        pca = PCA(n_components=10, random_state=1234)
        X_pca = pca.fit_transform(X)
        
        scores_mean_list_pca, scores_std_list_pca = evaluate_clustering_multiple_runs(X_pca, y_true, method, n_clusters_list, n_runs=n_runs)
        
        df_dict_PCA = {"silhouette (mean)": scores_mean_list_pca[:, 0], "silhouette (std)": scores_std_list_pca[:, 0],
                    "davies_bouldin (mean)": scores_mean_list_pca[:, 1], "davies_bouldin (std)": scores_std_list_pca[:, 1],
                    "calinski_harabasz (mean)": scores_mean_list_pca[:, 2], "calinski_harabasz (std)": scores_std_list_pca[:, 2]}
        
        df_scores_PCA = pd.DataFrame(df_dict_PCA, index=n_clusters_list)
        df_scores_PCA.index.name = "n_clusters"
        df_scores_PCA_list.append(df_scores_PCA)
        
        tsne = TSNE(n_components=2, random_state=1234)
        X_tsne = tsne.fit_transform(X)
        
        scores_mean_list_tsne, scores_std_list_tsne = evaluate_clustering_multiple_runs(X_tsne, y_true, method, n_clusters_list, n_runs=n_runs)
        
        df_dict_tsne = {"silhouette (mean)": scores_mean_list_tsne[:, 0], "silhouette (std)": scores_std_list_tsne[:, 0],
                        "davies_bouldin (mean)": scores_mean_list_tsne[:, 1], "davies_bouldin (std)": scores_std_list_tsne[:, 1],
                        "calinski_harabasz (mean)": scores_mean_list_tsne[:, 2], "calinski_harabasz (std)": scores_std_list_tsne[:, 2]}
        
        df_scores_tsne = pd.DataFrame(df_dict_tsne, index=n_clusters_list)
        df_scores_tsne.index.name = "n_clusters"
        df_scores_tsne_list.append(df_scores_tsne)
        
    if visualize:
        colors = ['blue', 'orange', 'green']
        colors_PCA = ['darkblue', 'darkorange', 'darkgreen']
        
        plt.figure(figsize=(17, 16))
        
        plt.subplot(3, 3, 1)
        plt.title('Silhouette score', fontsize=14)
        for i, method in enumerate(methods):
            plt.errorbar(n_clusters_list, df_scores_list[i]['silhouette (mean)'], yerr=df_scores_list[i]['silhouette (std)'], 
                         fmt='-o', markersize=4, label=method, color=colors[i])
        plt.ylabel('score', fontsize=13)
        plt.xticks(n_clusters_list, fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend()

        plt.subplot(3, 3, 2)
        plt.title('Davies-Bouldin score', fontsize=14)
        for i, method in enumerate(methods):
            plt.errorbar(n_clusters_list, df_scores_list[i]['davies_bouldin (mean)'], yerr=df_scores_list[i]['davies_bouldin (std)'], 
                         fmt='-o', markersize=4, label=method, color=colors[i])
        plt.xticks(n_clusters_list, fontsize=13)
        plt.yticks(fontsize=13)

        plt.subplot(3, 3, 3)
        plt.title('Calinski-Harabsz score', fontsize=14)
        for i, method in enumerate(methods):
            plt.errorbar(n_clusters_list, df_scores_list[i]['calinski_harabasz (mean)'], yerr=df_scores_list[i]['calinski_harabasz (std)'], 
                         fmt='-o', markersize=4, label=method, color=colors[i])
        plt.xticks(n_clusters_list, fontsize=13)
        plt.yticks(fontsize=13)
        
        plt.subplot(3, 3, 4)
        for i, method in enumerate(methods):
            plt.errorbar(n_clusters_list, df_scores_PCA_list[i]['silhouette (mean)'], yerr=df_scores_PCA_list[i]['silhouette (std)'], 
                         fmt='-o', markersize=4, label=method+' (PCA)', color=colors_PCA[i])
        plt.ylabel('score', fontsize=13)
        plt.xticks(n_clusters_list, fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend()

        plt.subplot(3, 3, 5)
        for i, method in enumerate(methods):
            plt.errorbar(n_clusters_list, df_scores_PCA_list[i]['davies_bouldin (mean)'], yerr=df_scores_PCA_list[i]['davies_bouldin (std)'], 
                         fmt='-o', markersize=4, label=method+' (PCA)', color=colors_PCA[i])
        plt.xticks(n_clusters_list, fontsize=13)
        plt.yticks(fontsize=13)

        plt.subplot(3, 3, 6)
        for i, method in enumerate(methods):
            plt.errorbar(n_clusters_list, df_scores_PCA_list[i]['calinski_harabasz (mean)'], yerr=df_scores_PCA_list[i]['calinski_harabasz (std)'], 
                         fmt='-o', markersize=4, label=method+' (PCA)', color=colors_PCA[i])
        plt.xticks(n_clusters_list, fontsize=13)
        plt.yticks(fontsize=13)
        
        plt.subplot(3, 3, 7)
        for i, method in enumerate(methods):
            plt.errorbar(n_clusters_list, df_scores_tsne_list[i]['silhouette (mean)'], yerr=df_scores_tsne_list[i]['silhouette (std)'], 
                         fmt='-o', markersize=4, label=method+' (t-SNE)', color=colors_PCA[i])
        plt.ylabel('score', fontsize=13)
        plt.xlabel('number of clusters', fontsize=13)
        plt.xticks(n_clusters_list, fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend()

        plt.subplot(3, 3, 8)
        for i, method in enumerate(methods):
            plt.errorbar(n_clusters_list, df_scores_tsne_list[i]['davies_bouldin (mean)'], yerr=df_scores_tsne_list[i]['davies_bouldin (std)'], 
                         fmt='-o', markersize=4, label=method+' (t-SNE)', color=colors_PCA[i])
        plt.xlabel('number of clusters', fontsize=13)
        plt.xticks(n_clusters_list, fontsize=13)
        plt.yticks(fontsize=13)

        plt.subplot(3, 3, 9)
        for i, method in enumerate(methods):
            plt.errorbar(n_clusters_list, df_scores_tsne_list[i]['calinski_harabasz (mean)'], yerr=df_scores_tsne_list[i]['calinski_harabasz (std)'], 
                         fmt='-o', markersize=4, label=method+' (t-SNE)', color=colors_PCA[i])
        plt.xlabel('number of clusters', fontsize=13)
        plt.xticks(n_clusters_list, fontsize=13)
        plt.yticks(fontsize=13)
        
        if path is not None:
            plt.savefig(path, dpi=300, bbox_inches='tight')
        
        plt.show()

def evaluate_num_clusters(X, y_true, method, n_clusters_list=[2, 3, 4, 5, 6, 7, 8, 9, 10], n_runs=5, 
                          n_pca_components=10, visualize=False, tick_spacing=1, return_df=False, path=None):
    
    scores_mean_list, scores_std_list = evaluate_clustering_multiple_runs(X, y_true, method, n_clusters_list, n_runs=n_runs)

    df_dict = {"silhouette (mean)": scores_mean_list[:, 0], "silhouette (std)": scores_std_list[:, 0],
            "davies_bouldin (mean)": scores_mean_list[:, 1], "davies_bouldin (std)": scores_std_list[:, 1],
            "calinski_harabasz (mean)": scores_mean_list[:, 2], "calinski_harabasz (std)": scores_std_list[:, 2]}

    df_scores = pd.DataFrame(df_dict, index=n_clusters_list)
    df_scores.index.name = "n_clusters"
    
    # Do the same with PCA
    pca = PCA(n_components=n_pca_components, random_state=1234)
    X_pca = pca.fit_transform(X)
    
    scores_mean_list_pca, scores_std_list_pca = evaluate_clustering_multiple_runs(X_pca, y_true, method, n_clusters_list, n_runs=n_runs)
    
    df_dict_PCA = {"silhouette (mean)": scores_mean_list_pca[:, 0], "silhouette (std)": scores_std_list_pca[:, 0],
                   "davies_bouldin (mean)": scores_mean_list_pca[:, 1], "davies_bouldin (std)": scores_std_list_pca[:, 1],
                   "calinski_harabasz (mean)": scores_mean_list_pca[:, 2], "calinski_harabasz (std)": scores_std_list_pca[:, 2]}
    
    df_scores_PCA = pd.DataFrame(df_dict_PCA, index=n_clusters_list)
    df_scores_PCA.index.name = "n_clusters"
    
    x_ticks = np.arange(n_clusters_list[0], n_clusters_list[-1]+2, tick_spacing)
    if visualize:
        plt.figure(figsize=(16, 5))
        plt.subplot(1, 3, 1)
        plt.title('Silhouette score', fontsize=14)
        plt.errorbar(n_clusters_list, df_scores['silhouette (mean)'], yerr=df_scores['silhouette (std)'], 
                     fmt='-o', markersize=4, label='without PCA')
        plt.errorbar(n_clusters_list, df_scores_PCA['silhouette (mean)'], yerr=df_scores_PCA['silhouette (std)'], 
                     fmt='-o', markersize=4, label='with PCA')
        plt.xlabel('number of clusters', fontsize=13)
        plt.ylabel('score', fontsize=13)
        plt.xticks(x_ticks, fontsize=13)
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.title('Davies-Bouldin score', fontsize=14)
        plt.errorbar(n_clusters_list, df_scores['davies_bouldin (mean)'], yerr=df_scores['davies_bouldin (std)'], fmt='-o', markersize=4)
        plt.errorbar(n_clusters_list, df_scores_PCA['davies_bouldin (mean)'], yerr=df_scores_PCA['davies_bouldin (std)'], fmt='-o', markersize=4)
        plt.xlabel('number of clusters', fontsize=13)
        plt.ylabel('score', fontsize=13)
        plt.xticks(x_ticks, fontsize=13)

        plt.subplot(1, 3, 3)
        plt.title('Calinski-Harabsz score', fontsize=14)
        plt.errorbar(n_clusters_list, df_scores['calinski_harabasz (mean)'], yerr=df_scores['calinski_harabasz (std)'], fmt='-o', markersize=4)
        plt.errorbar(n_clusters_list, df_scores_PCA['calinski_harabasz (mean)'], yerr=df_scores_PCA['calinski_harabasz (std)'], fmt='-o', markersize=4)
        plt.xlabel('number of clusters', fontsize=13)
        plt.ylabel('score', fontsize=13)
        plt.xticks(x_ticks, fontsize=13)

        if path is not None:
            plt.savefig(path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    if return_df:
        return df_scores, df_scores_PCA