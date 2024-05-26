import faiss
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh, svds
import numpy as np
from jax import jit
from jax import vmap
from jax import lax
from jax import device_put
import jax.numpy as jnp
import plotly.express as px
import pandas as pd
import gc
import sklearn.utils.extmath as xm
import openml
from loguru import logger
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from tqdm import tqdm
#
# Custom bisection method to compute sigmas for conformal rescaling
#

def bisection_method_factory(tol=1e-5, max_iter=100):

    @jit
    def bisection_method(row, target_c, a, b):
        def f(sigma):
            terms = jnp.exp((-row[1:] + row[1])/ sigma)
            return jnp.sum(terms) - target_c

        def body_fun(loop_vars):
            a, b, fa, fb = loop_vars
            mid = (a + b) / 2.0
            fmid = f(mid)
            return lax.cond(fa * fmid < 0,
                            lambda _: (a, mid, fa, fmid),
                            lambda _: (mid, b, fmid, fb),
                            None)

        def cond_fun(loop_vars):
            a, b, _, _ = loop_vars
            return (b - a) > tol

        fa = f(a)
        fb = f(b)
        a_final, b_final, _, _ = lax.while_loop(cond_fun, body_fun, (a, b, fa, fb))
        return (a_final + b_final) / 2

    return bisection_method

@jit
def t_distribution_kernel( x, df):
    #
    return (1 + x ** 2 / df) ** (-(df + 1) / 2)

@jit
def grad_coeff_rep(x, df, multiplier=0.5):
        y  = t_distribution_kernel(1.0*x, df)
        y /= jnp.max(y)
        grad_coeff = -1.0*y
        return multiplier * grad_coeff

@jit
def grad_coeff_att(x, df, min_dist):
        y  = t_distribution_kernel(1.0/x, df)
        y /= jnp.max(y)
        grad_coeff = 1.0*(y - min_dist)
        return grad_coeff



pipeline = Pipeline([
    ('variance_threshold', VarianceThreshold(threshold=0)),
    ('scaler', StandardScaler())
])

#df_transformed = pipeline.fit_transform(df)

class myUMAP:
  #
  # Class constructor
  #
  def __init__(self,
               dimension=2,
               n_neighbors=10,
               sigma_tol=1e-6,
               max_iter_sigma=200,
               max_iter_layout=100,
               min_dist=1e-1,
               symmetrize = False,
               n_oversamples = None,
               multiplier = 0.5
               ):
    #
    self.dimension = dimension
    self.n_neighbors = n_neighbors
    self.sigma_tol = sigma_tol
    self.max_iter_sigma = max_iter_sigma
    self.max_iter_layout = max_iter_layout
    self.min_dist = min_dist
    self.laplacian_eigs = None
    self.diffusion_eigs = None
    self.layout = None
    self.init_embedding  = None
    self.symmetrize = symmetrize
    self.n_oversamples = n_oversamples
    self.multiplier = multiplier
  #
  # Fit on data
  #
  def fit(self, data):
    #
    self.data = pipeline.fit_transform(data)
    self.n_samples = self.data.shape[0]
    custom_bisection_method = bisection_method_factory(self.sigma_tol, self.max_iter_sigma)
    self.vmap_bisection = vmap(custom_bisection_method, in_axes=(0, None, None, None))
    #
    self.make_knn_adjacency()
    self.do_sigmas()
    self.do_weights()
    #
    return self
  #
  # Transform data
  #
  def transform(self):
    #
    self.do_spectral_embedding()
    self.do_layout()
    return self.layout
  #
  # Fit on data transform it
  #
  def fit_transform(self, data):
    #
    self.data = data
    self.fit(self.data)
    return self.transform()
  #
  # Make kNN adjacency matrix
  #
  def make_knn_adjacency(self):

    logger.info('make_knn_adjacency')
    self.data = np.ascontiguousarray(self.data.astype(np.float32))
    n_neighbors = self.n_neighbors + 1  # Including the point itself
    data_dim = self.data.shape[1]

    res = faiss.StandardGpuResources()

    #config = faiss.GpuIndexIVFFlatConfig()
    #config.device = 0 # GPU
    #config.flatConfig.useFloat32 = False
    #config.flatConfig.useFloat16 = True

    #if self.n_samples < 10000:
    #  n_clusters = 32
    #else:
    #  n_clusters = min(int(np.log(self.n_samples)), 64)

    #quantizer = faiss.IndexFlatL2(data_dim)
    #index_cpu = faiss.IndexIVFFlat(quantizer, data_dim, n_clusters, faiss.METRIC_L2)
    #index = faiss.GpuIndexIVFFlat(res, index_cpu, config)

    index = faiss.GpuIndexFlatL2(res, data_dim)

    #index.train(self.data)
    index.add(self.data)

    distances, indices = index.search(self.data, n_neighbors)

    # Convert distances to the same format you were using
    self.distances = distances
    self.indices = indices

    self.nearest_neighbor_distances = self.distances[:, 1]  # Excluding the point itself
    #self.n_samples = len(self.data)
    self.row_idx = np.repeat(np.arange(self.n_samples), n_neighbors)
    self.col_idx = self.indices.ravel()

    data_values = self.distances.ravel()
    adj_mat = csr_matrix((data_values, (self.row_idx, self.col_idx)), shape=(self.n_samples, self.n_samples))
    self.adjacency = adj_mat
    # release memory from index
    del index; gc.collect()
    logger.info('done ...')
  #
  # Compute sigmas for conformal rescaling of kNN neighborhoods
  #
  def do_sigmas(self):
    #
    logger.info('do_sigmas')
    target_c = jnp.log(self.n_neighbors)
    self.distances_jax = device_put(self.distances)
    self.sigmas_jax    = self.vmap_bisection(self.distances_jax, target_c, self.sigma_tol, target_c)
    logger.info('done ...')
  #
  # Compute probability kernel weights
  #
  def do_weights(self):
    #
    logger.info('do_weights')
    nearest_jax = device_put(self.nearest_neighbor_distances)
    #
    sigmas =   self.sigmas_jax[:, None]
    nndist = - self.distances_jax + nearest_jax[:, None]
    mask   = (nndist > 0) | (sigmas == 0)
    #
    weights_jax = jnp.where(mask,
                            0.0,
                            jnp.exp(nndist / sigmas)
                            )
    #
    # JAX array to NumPy array
    #
    # Ensure JAX computation is done and copy data to host as NumPy array
    weights_np = weights_jax.block_until_ready().copy()
    self.weights = weights_np
    #
    self.W = csr_matrix((weights_np.ravel(), (self.row_idx, self.col_idx)), shape=(self.n_samples, self.n_samples))
    self.W.eliminate_zeros()
    #
    W_transpose = csr_matrix(self.W.transpose())
    W_transpose.eliminate_zeros()
    #
    sums_mat  = self.W + W_transpose
    prod_mat  = self.W.multiply(W_transpose)
    self.similarity = sums_mat - prod_mat
    print('done ...')
  #
  # Compute the normalized Laplacian of a given matrix M
  #
  def make_norm_laplacian(self, A):
    # A = CSR adjacency matrix
    #
    # Compute the (diagonal) degree matrix D
    #
    degrees  = np.array(A.sum(axis=1)).flatten()
    #
    # Compute D^(-1/2)
    #
    D_inv_sqrt = diags(np.power(degrees, -0.5), format="csr")
    #
    # Compute the normalized Laplacian matrix
    #
    # A_norm = D^(-1/2) @ A @ D^(-1/2)
    A_norm = D_inv_sqrt.dot(A)
    A_norm = A_norm.dot(D_inv_sqrt)
    # L_norm = Id - A_norm
    L_norm = eye(degrees.size, format="csr") - A_norm
    #
    return A_norm, L_norm, degrees
    #
  # Gaussian kernel
  #
  def gaussian_kernel(self, x):
    #
    return np.exp(-0.5 * x**2)
  #
  # Compute the diffusion operator
  #
  def make_diffusion_operator(self, A, alpha):
    # A = CSR adjacency matrix
    # alpha = diffusion parameter
    #
    degrees  = np.array(A.sum(axis=1)).flatten()
    #
    D_alpha = diags(np.power(degrees, -alpha), format="csr")
    # L_alpha = D^(-\alpha) @ L @ D^(-\alpha)
    L_alpha = D_alpha.dot(A)
    L_alpha = L_alpha.dot(D_alpha)
    #
    # Compute the diffusion operator matrix
    #
    deg_alpha = np.array(L_alpha.sum(axis=1)).flatten()
    if not self.symmetrize:
      D_inv = diags(np.power(deg_alpha, -1.0), format="csr")
      M = D_inv.dot(L_alpha)
    else:
      D_inv = diags(np.power(deg_alpha, -0.5), format="csr")
      M = D_inv.dot(L_alpha)
      M = M.dot(D_inv)
      #
    return L_alpha, M, degrees
  #
  # Spectral embedding of the kNN graph
  #
  def do_spectral_embedding(self):
    #
    logger.info('do_spectral_embedding')
    self.similarity.data = self.gaussian_kernel(self.similarity.data)
    _, lap, degrees = self.make_norm_laplacian(self.similarity)
    self.degrees = degrees
    s, u = eigsh(lap, k=self.dimension, which='SA', tol=1e-6)
    self.init_embedding = u
    self.laplacian_eigs = s
    logger.info('done ...')
  #
  # Diffusion embedding of the kNN graph ### NEEDS CHECK
  #
  def do_diffusion_embedding(self, alpha, t):
    #
    logger.info('do_diffusion_embedding')
    self.similarity.data = self.gaussian_kernel(self.similarity.data)
    _, difop, degrees = self.make_diffusion_operator(self.similarity, alpha)
    self.degrees = degrees
    #s, u = eigsh(difop, k=self.dimension, which='LA')
    if self.n_oversamples is not None:
      n_oversamples = self.n_oversamples
    else:
      n_oversamples = min(max(int(2*np.log(self.n_samples)), 200), int(self.n_samples/2))
    s, u = xm._randomized_eigsh(difop,
                                n_components=self.dimension,
                                n_oversamples=n_oversamples,
                                n_iter=1,
                                power_iteration_normalizer='QR',
                                random_state=42)
    s /= np.max(s)
    self.init_embedding = u * np.power(s, t)
    self.diffusion_eigs = s
    logger.info('done ...')
  #
  # Student's t-distribution kernel
  #

  #
  # Graph layout
  #
  def do_layout(self, cutoff=42, df=1, num_iterations = None, batch_size= None):
    logger.info('do_layout')
    #
    if num_iterations is None:
      num_iterations = self.max_iter_layout
    #
    if batch_size is None:
      batch_size   = min(int(self.n_samples/10), max(128, int(np.log(self.n_samples))))
    #
    cutoff  = jnp.array([cutoff])     # Cutoff to avoid drift
    #
    init_pos_jax = device_put(self.init_embedding)
    #
    # Normalize initial positions
    #
    init_pos_jax -= init_pos_jax.mean(axis=0)
    init_pos_jax /= init_pos_jax.std(axis=0)
    #
    weights_jax  = device_put(self.weights)
    indices_jax  = device_put(self.indices)

    #
    # Initialize the index of embedding points for repulsive forces
    #

    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, self.dimension)

    #

    for iter_id in tqdm(range(num_iterations)):

      #
      # Reset and populate the index
      #

      logger.debug(f'Iteration {iter_id + 1}')

      init_pos_np = np.ascontiguousarray(init_pos_jax).astype(np.float32)
      index.reset()
      index.add(init_pos_np)
      _, indices_emb  = index.search(init_pos_np, batch_size)
      indices_emb_jax = device_put(indices_emb)

      #
      # Compute attraction forces first
      #

      #
      # Get all 'v_pos' replicated for each neighbor
      #
      v_pos = init_pos_jax[:, None, :]
      #
      # Get all 'u_pos' using advanced indexing
      #
      u_pos = init_pos_jax[indices_jax]
      #
      # Compute the vector differences (broadcast subtraction)
      #
      position_diff = u_pos - v_pos
      #
      # Compute the geometric distances (low-dimensional embedding)
      # with safety against division by zero
      #
      distance_geom = jnp.linalg.norm(position_diff, axis=2, keepdims=True)
      #
      # Calculate directional vectors
      #
      mask      = (distance_geom > 0)
      direction = jnp.where(mask,
                            position_diff / distance_geom,
                            0.0
                            )
      #
      # Calculate gradient coefficients
      #

      #
      coeff_vmap = vmap(grad_coeff_att, in_axes=(0,None, None))
      #
      grad_coeff = jnp.where(mask,
                             coeff_vmap(distance_geom, df, self.min_dist),
                             0.0
                             )
      #
      attraction_force = jnp.sum(grad_coeff * direction, axis=1)

      #
      # Compute repulsion forces next
      #

      # Get all 'v_pos' replicated for each neighbor
      v_pos = init_pos_jax[:, None, :]

      # Get all 'u_pos' using advanced indexing
      u_pos = init_pos_jax[indices_emb_jax]

      # Compute the vector differences
      position_diff = u_pos - v_pos  # Broadcasting subtraction

      # Calculate distances with safety against division by zero
      distance_geom = jnp.linalg.norm(position_diff, axis=2, keepdims=True)
      #
      # Calculate directional vectors
      #
      mask      = (distance_geom > 0)
      direction = jnp.where(mask,
                            position_diff / distance_geom,
                            0.0
                            )
      #
      # Calculate gradient coefficients
      #

      #
      coeff_vmap = vmap(grad_coeff_rep, in_axes=(0, None, None))
      #
      grad_coeff = jnp.where(mask,
                             coeff_vmap(distance_geom, df, self.multiplier),
                             0.0
                             )
      #
      repulsion_force = jnp.sum(grad_coeff * direction, axis=1)
      #
      alpha = 1.0 - iter_id / num_iterations
      net_force = alpha * (attraction_force + repulsion_force)
      net_force = jnp.clip(net_force, -cutoff, cutoff)
      init_pos_jax += net_force
      #

    # Normalize positions
    init_pos_jax -= init_pos_jax.mean(axis=0)
    init_pos_jax /= init_pos_jax.std(axis=0)
    self.layout = np.asarray(init_pos_jax)
    print('done ...')
  #
  # Visualize the layout
  #
  def visualize(self,labels = None):
    #
    if self.layout is None:
      return None
    #
    if self.dimension==2:
      datadf = pd.DataFrame(self.layout, columns=['x','y'])
      if labels is not None:
        datadf['label'] = labels
      else:
        datadf['label'] = 0
      fig = px.scatter(datadf, x='x', y='y', color = 'label')
      return fig
    #
    if self.dimension==3:
      datadf = pd.DataFrame(self.layout, columns=['x','y','z'])
      if labels is not None:
        datadf['label'] = labels
      else:
        datadf['label'] = 0
      fig = px.scatter_3d( datadf, x='x', y='y', z='z', color = 'label')
      return fig
    #
    return None