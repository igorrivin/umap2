import optuna
from loguru import logger
from datadriver import *

class optunizer:

  def __init__(self, X, y, my_logger=None):

    self.X = X
    self.y = y
    self.pareto_front = None

    if not(my_logger):
      logger.remove();
      logger.add(sys.stdout, level="INFO");
      self.logger = logger
    else:
      self.logger = my_logger

  def objective(self, trial):

      symmetrize = trial.suggest_categorical('symmetrize', [True, False])
      n_neighbors = trial.suggest_int('n_neighbors', 10, 100)
      max_iter_layout = trial.suggest_int('max_iter_layout', 50, 500)
      alpha = trial.suggest_float('alpha', 0.1, 1.0)
      t = trial.suggest_float('t', 1.0, 20.0)
      multiplier = trial.suggest_float('multiplier', 0.1, 10.0)
      cutoff = trial.suggest_float('cutoff', 1.0, 16.0)
      df = trial.suggest_float('df', 0.5, 5.0)
                                   
      _, _, spear, trusty = process_data(self.X, self.y, do_viz = False, **trial.params)

      # We aim to maximize Spearman Correlation and Trustworthiness 
      return spear, trusty

  def do_study(self, n_trials=100):

      self.study = optuna.create_study(directions=["maximize", "maximize"])
      self.study.optimize(self.objective, n_trials=n_trials)

      self.logger.info('Printing out the Pareto front of this study ...')

      self.pareto_front = self.study.best_trials
      for trial in self.pareto_front:
        self.logger.info(f'Trial {trial.number} ...')
        self.logger.info(f'  Parameters: {trial.params}')
        self.logger.info(f'  Spearman Correlation: {trial.values[0]:.4f}')
        self.logger.info(f'  Trustworthiness: {trial.values[1]:.4f}')