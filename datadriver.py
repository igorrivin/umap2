import openml
from metrics import trustworthiness, do_spearman
from myumapper import myUMAP
import plotly.express as px
import pandas as pd
import sys
from loguru import logger

# Cache directory in the source directory
import os
openml.config.cache_directory = os.path.expanduser('.')
#

# Suppress all warnings
import warnings
warnings.filterwarnings('ignore')
#

#
# Create a logger for info messages
#
logger.remove();
logger.add(sys.stdout, level="INFO");

def read_openml(id):
  ds = openml.datasets.get_dataset(id, 
				   download_data=True,
				   download_qualities=True,
				   download_features_meta_data=True)
  X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
  logger.info(f'This is the {ds.name} dataset ...')
  return X, y

def process_data(X, y, do_viz = True, **kwargs):
  keydict = kwargs

  myu= myUMAP(dimension=2,
              n_neighbors=keydict.get('n_neighbors', 25),
              sigma_tol=1e-6,
              max_iter_sigma=200,
              max_iter_layout=keydict.get('max_iter_layout', 200),
              symmetrize = keydict.get('symmetrize', True),
              multiplier = keydict.get('multiplier', 0.5),
	      my_logger=logger,
              )
  myu.fit(X)
  myu.do_diffusion_embedding(alpha=keydict.get('alpha', 0.5),  t=keydict.get('t', 5.0))
  myu.do_layout(cutoff=keydict.get('cutoff', 6), df=keydict.get('df', 1))
  if do_viz:
    fig = px.scatter(pd.DataFrame(myu.init_embedding, columns=['x','y']), x='x', y='y', color = y)
    fig.show()

    fig = myu.visualize(labels = y)
    fig.show()

  reduced_data = myu.layout
  cluster_labels = y

  trusty = trustworthiness(X, reduced_data)
  spear = do_spearman(X, reduced_data, sample_size = 1000)

  return reduced_data, cluster_labels, spear, trusty

def do_workflow(id, **kwargs):
  keydict = kwargs
  data, labels = read_openml(id)
  X, y = data.to_numpy(), labels.to_numpy()
  _, _, spear, trusty = process_data(X, y, **kwargs)
  logger.info(f'Spearman correlation {spear}')
  logger.info(f'Trustworthiness {trusty}')