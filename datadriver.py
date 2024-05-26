import openml
from metrics import trustworthiness, do_spearman
from myumapper import myUMAP

def read_openml(id):
  ds = openml.datasets.get_dataset(id)
  X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
  print("This is the {} dataset", ds.name)
  return X, y

def process_data(X, y, do_viz = True, **kwargs):
  keydict = kwargs

  myu= myUMAP(dimension=2,
              n_neighbors=keydict.get('n_neighbors', 25),
              sigma_tol=1e-6,
              max_iter_sigma=200,
              max_iter_layout=keydict.get('max_iter_layout',200),
              symmetrize = keydict.get('symmetrize',True),
              multiplier = keydict.get('multiplier', 0.5)
              )
  myu.fit(X)
  myu.do_diffusion_embedding(alpha=keydict.get('alpha', 0.5),  t=keydict.get('t', 10.0))
  myu.do_layout(cutoff=keydict.get('cutoff', 6), df=keydict.get('df', 1))
  if do_viz:
    fig = px.scatter(pd.DataFrame(myu.init_embedding, columns=['x','y']), x='x', y='y', color = y)
    fig.show()

    fig = myu.visualize(labels = y)
    fig.show()

  reduced_data = myu.layout
  cluster_labels = y
  trusty = trustworthiness(X, reduced_data)
  spear = do_spearman(X, reduced_data, sample_size = 2000)
  """
  silhouette_score_value, silhouette_quality = calculate_silhouette_score(reduced_data, cluster_labels)
  davies_bouldin_value, davies_bouldin_quality = calculate_davies_bouldin_index(reduced_data, cluster_labels)
  calinski_harabasz_value, calinski_harabasz_quality = calculate_calinski_harabasz_index(reduced_data, cluster_labels)
  if do_viz:
    print(f"Silhouette Score: {silhouette_score_value:.2f} ({silhouette_quality})")
    print(f"Davies-Bouldin Index: {davies_bouldin_value:.2f} ({davies_bouldin_quality})")
    print(f"Calinski-Harabasz Index: {calinski_harabasz_value:.2f} ({calinski_harabasz_quality})")
  """
  return reduced_data, spear, trusty
  #, silhouette_score_value, -davies_bouldin_value, calinski_harabasz_value

def do_workflow(id, **kwargs):
  keydict = kwargs
  X, y = read_openml(id)
  process_data(X, y, **kwargs)