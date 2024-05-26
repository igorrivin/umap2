import optuna
from datadriver import process_data
# Generate synthetic data for demonstration
#data, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

class optunizer:
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def objective(self, trial):

      symmetrize = trial.suggest_categorical('symmetrize', [True, False])
      n_neighbors = trial.suggest_int('n_neighbors', 10, 100)
      max_iter_layout = trial.suggest_int('max_iter_layout', 50, 500)
      alpha = trial.suggest_float('alpha', 0.1, 1.0)
      t = trial.suggest_float('t', 1, 20)
      multiplier = trial.suggest_float('multiplier', 0.1, 10.0)
      cutoff = trial.suggest_float('cutoff', 0.5, 50)
      df = trial.suggest_float('df', 0.5, 5.0)
      #_, spear, trusty, silhouette, davies_bouldin, calinski_harabasz = process_data(self.X, self.y, do_viz = False, **trial.params)
                                   
      _, spear, trusty = process_data(self.X, self.y, do_viz = False, **trial.params)

      # We aim to maximize silhouette and calinski_harabasz, and minimize davies_bouldin
      return spear, trusty

