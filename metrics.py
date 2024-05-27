from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import faiss

def calculate_silhouette_score(data, labels):
    score = silhouette_score(data, labels)
    if score > 0.7:
        quality = "Excellent"
    elif score > 0.5:
        quality = "Good"
    elif score > 0.3:
        quality = "Fair"
    else:
        quality = "Poor"
    return score, quality

def calculate_davies_bouldin_index(data, labels):
    score = davies_bouldin_score(data, labels)
    if score < 0.5:
        quality = "Excellent"
    elif score < 1.0:
        quality = "Good"
    elif score < 2.0:
        quality = "Fair"
    else:
        quality = "Poor"
    return score, quality

def calculate_calinski_harabasz_index(data, labels):
    score = calinski_harabasz_score(data, labels)
    # Higher is better, so the qualitative measure is relative
    if score > 3000:
        quality = "Excellent"
    elif score > 2000:
        quality = "Good"
    elif score > 1000:
        quality = "Fair"
    else:
        quality = "Poor"
    return score, quality



def do_spearman(X, X_embedded, sample_size = 2000):
  # Spearman Rank Correlation (using sampling)
  original_data = X
  reduced_data = X_embedded
  n_samples = min(sample_size, len(original_data))
  indices = random.sample(range(len(original_data)), n_samples)
    
  original_sample = original_data[indices, :]
  reduced_sample = reduced_data[indices, :]
    
  original_distances = pairwise_distances(original_sample).flatten()
  reduced_distances = pairwise_distances(reduced_sample).flatten()
    
  spearman_corr, _ = spearmanr(original_distances, reduced_distances)

  return spearman_corr

def compute_knn_faiss(data, k):
    """
    Compute k-nearest neighbors using Faiss.
    
    Parameters:
    data (np.ndarray): The data to compute k-nearest neighbors on.
    k (int): The number of nearest neighbors to find.
    
    Returns:
    np.ndarray: Indices of the k-nearest neighbors for each point.
    """
    index = faiss.IndexFlatL2(data.shape[1])
    index.add(data)
    _, indices = index.search(data, k + 1)  # k + 1 because the point itself is included
    return indices[:, 1:]  # Exclude the point itself

def trustworthiness(X, X_embedded, n_neighbors=5, extended_neighbors=20):
    n_samples = X.shape[0]

    # Compute the k-nearest neighbors using Faiss
    knn_X = compute_knn_faiss(X, n_neighbors)
    knn_X_extended = compute_knn_faiss(X_embedded, extended_neighbors)

    # Calculate the rank differences for neighbors
    rank_diff_sum = 0.0
    for i in range(n_samples):
        ranks_in_high_dim = {j: rank for rank, j in enumerate(knn_X[i])}
        for rank, j in enumerate(knn_X_extended[i]):
            if j not in ranks_in_high_dim:
                if rank < n_neighbors:
                    rank_diff_sum += extended_neighbors + 1  # Assign fixed penalty
            else:
                if rank < n_neighbors:
                    rank_diff_sum += ranks_in_high_dim[j] - n_neighbors

    T = 1.0 - (2.0 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))) * rank_diff_sum
    return T

