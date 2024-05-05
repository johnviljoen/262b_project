import jax.numpy as jnp
import numpy as np

def project_vector(vec_to_project, base_vec, components):
    """
    Project a vector onto the hyperplane defined by a base vector and components.

    Parameters:
    - vec_to_project: The vector to be projected.
    - base_vec: A point on the hyperplane (e.g., vec_best_pol_s).
    - components: A matrix whose columns define directions spanning the hyperplane.

    Returns:
    - projected_vec: The projection of vec_to_project onto the hyperplane.
    """
    B = components
    projector = B @ jnp.linalg.inv(B.T @ B) @ B.T
    projection = projector @ (vec_to_project - base_vec)
    projected_vec = base_vec + projection
    return projected_vec

# https://medium.com/@nahmed3536/a-python-implementation-of-pca-with-numpy-1bbd3b21de2e
def pca(data, num_components=2):
    standardized_data = (data - data.mean(axis = 0)) / data.std(axis = 0)
    covariance_matrix = np.cov(standardized_data, ddof = 1, rowvar = False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # np.argsort can only provide lowest to highest; use [::-1] to reverse the list
    order_of_importance = np.argsort(eigenvalues)[::-1] 
    # utilize the sort order to sort eigenvalues and eigenvectors
    # sorted_eigenvalues = eigenvalues[order_of_importance]
    sorted_eigenvectors = eigenvectors[:,order_of_importance] # sort the columns
    # use sorted_eigenvalues to ensure the explained variances correspond to the eigenvectors
    # explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    reduced_data = standardized_data @ sorted_eigenvectors[:,:num_components] # transform the original data
    # total_explained_variance = sum(explained_variance[:k])
    return reduced_data

def find_coefficients(base_vec, components, projection):
    A = np.column_stack([base_vec, components[:,0], components[:,1]])
    coefficients, residuals, rank, s = np.linalg.lstsq(A, projection, rcond=None)
    return coefficients