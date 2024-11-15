import numpy as np
import pandas as pd
from multiprocessing import Pool
from multiprocessing import cpu_count
from math import isclose
from math import ceil

from copy import deepcopy

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

import warnings

from optimizers import *

warnings.filterwarnings("ignore")

class aew():
    def __init__(self, similarity_matrix, data, labels, gamma_init=None):

        #### DATA HOLDING OBJECTS
        self.data = data
        self.labels = labels
        self.eigenvectors = None
        self.gamma = self.gamma_initializer(gamma_init)
        self.similarity_matrix = self.correct_similarity_matrix_diag(similarity_matrix)
        self.final_error = 0

    def correct_similarity_matrix_diag(self, similarity_matrix):
        identity = np.zeros((self.data.shape[0], self.data.shape[0]))
        identity_diag = np.diag(identity)
        identity_diag_res = identity_diag + 1
        np.fill_diagonal(similarity_matrix, identity_diag_res)
        return similarity_matrix

    def gamma_initializer(self, gamma_init=None):
        if gamma_init == None:
            gamma = np.ones(self.data.loc[[0]].shape[1])
        elif gamma_init == 'var':
            gamma = np.var(self.data, axis=0).values
        elif gamma_init == 'random_int':
            gamma = np.random.randint(0, 1000, (1, 41))
        elif gamma_init == 'random_float':
            rng = np.random.default_rng()
            gamma = rng.random(size=(1, 41)) 
        return gamma

    def similarity_function(self, pt1_idx, pt2_idx, gamma): # -> Computation accuracy verified
        point1 = np.asarray(self.data.loc[[pt1_idx]])[0]
        point2 = np.asarray(self.data.loc[[pt2_idx]])[0]

        temp_res = 0

        deg_pt1 = np.sum(self.similarity_matrix[pt1_idx])
        deg_pt2 = np.sum(self.similarity_matrix[pt2_idx])

        #####  Gaussian Kernel
        similarity_measure = np.sum(np.where(np.abs(gamma) > .1e-5, (((point1 - point2)**2)/(gamma**2)), 0))

        ##### Exponential Kernel
        #similarity_measure = np.sum(np.where(point1 - point2  > 0, ((np.sqrt((point1 - point2)**2))*self.gamma), 0))
        similarity_measure = np.exp(-similarity_measure, dtype=np.longdouble)

        degree_normalization_term = np.sqrt(np.abs(deg_pt1 * deg_pt2))

        ##May need to relax this bound
        if degree_normalization_term != 0 and not isclose(degree_normalization_term, 0, abs_tol=1e-100):
            return similarity_measure / degree_normalization_term
        else:
            return 0

    def objective_computation(self, section, adj_matrix, gamma):
        approx_error = 0
        for idx in section:
            degree_idx = np.sum(adj_matrix[idx])
            xi_reconstruction = np.sum([adj_matrix[idx][y]*np.asarray(self.data.loc[[y]])[0] for y in range(len(adj_matrix[idx])) if idx != y], 0)            

            if degree_idx != 0 and not isclose(degree_idx, 0, abs_tol=1e-100):
                xi_reconstruction /= degree_idx
                xi_reconstruction = xi_reconstruction[0]
            else:
                xi_reconstruction = np.zeros(len(self.gamma))

        return np.sum((np.asarray(self.data.loc[[idx]])[0] - xi_reconstruction)**2)

    def objective_function(self, adj_matr, gamma):
        split_data = self.split(range(self.data.shape[0]), cpu_count())
        with Pool(processes=cpu_count()) as pool:
            errors = [pool.apply_async(self.objective_computation, (section, adj_matr, gamma)) \
                                                                 for section in split_data]

            error = [error.get() for error in errors]
        return np.sum(error)

    def gradient_computation(self, section, similarity_matrix, gamma):
        gradient = np.zeros(len(gamma))
        for idx in section:
            dii = np.sum(similarity_matrix[idx])
            xi_reconstruction = np.sum([similarity_matrix[idx][y]*np.asarray(self.data.loc[[y]])[0] for y in range(len(similarity_matrix[idx])) if idx != y], 0)
            if dii != 0 and not isclose(dii, 0, abs_tol=1e-100):
                xi_reconstruction = np.divide(xi_reconstruction, dii, casting='unsafe', dtype=np.longdouble)
                first_term = np.divide((np.asarray(self.data.loc[[idx]])[0] - xi_reconstruction), dii, casting='unsafe', dtype=np.longdouble)
            else:
                first_term  = np.zeros_like(xi_reconstruction)
                xi_reconstruction  = np.zeros_like(xi_reconstruction)
            cubed_gamma = np.where( np.abs(gamma) > .1e-7 ,  gamma**(-3), 0)
            dw_dgamma = np.sum([(2*similarity_matrix[idx][y]* (((np.asarray(self.data.loc[[idx]])[0] - np.asarray(self.data.loc[[y]])[0])**2)*cubed_gamma)*np.asarray(self.data.loc[[y]])[0]) for y in range(self.data.shape[0]) if idx != y])
            dD_dgamma = np.sum([(2*similarity_matrix[idx][y]* (((np.asarray(self.data.loc[[idx]])[0] - np.asarray(self.data.loc[[y]])[0])**2)*cubed_gamma)*xi_reconstruction) for y in range(self.data.shape[0]) if idx != y])


            ##### Exponential kernel derivative
            #dw_dgamma = np.sum([(2*self.similarity_matrix[idx][y]* (((np.asarray(self.data.loc[[idx]])[0] - np.asarray(self.data.loc[[y]])[0])**2))*np.asarray(self.data.loc[[y]])[0]) for y in range(self.data.shape[0]) if idx != y])
            #dD_dgamma = np.sum([(2*self.similarity_matrix[idx][y]* (((np.asarray(self.data.loc[[idx]])[0] - np.asarray(self.data.loc[[y]])[0])**2))*xi_reconstruction) for y in range(self.data.shape[0]) if idx != y])


            #print(gradient, " ", first_term, " ", dw_dgamma, " ", dD_dgamma)
            gradient = gradient + (first_term * (dw_dgamma - dD_dgamma))
            
            gradient = np.nan_to_num(gradient, nan=0)
        return gradient

    def split(self, a, n):
        k, m = divmod(len(a), n)
        return [a[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]

    def gradient_function(self, similarity_matrix, gamma):
        gradient = []
    
        split_data = self.split(range(self.data.shape[0]), cpu_count())

        with Pool(processes=cpu_count()) as pool:
            gradients = [pool.apply_async(self.gradient_computation, (section, similarity_matrix, gamma)) \
                                                                 for section in split_data]

            gradients = [gradient.get() for gradient in gradients]

        return np.sum(gradients, axis=0)

    def generate_optimal_edge_weights(self, num_iterations):
        print("Generating Optimal Edge Weights")

        #self.gradient_descent(.000002, num_iterations, .01)

        #self.adam_computation(num_iterations)

        self.similarity_matrix = self.generate_edge_weights(self.gamma)

        #self.simulated_annealing(num_iterations)

        #pso = ParticleSwarmOptimizer(self.similarity_matrix, self.gamma, self.objective_function, self.generate_edge_weights, 30, len(self.gamma), 10)

        #self.gamma = pso.optimize()

        sba = SwarmBasedAnnealingOptimizer(self.similarity_matrix, self.gamma, self.objective_function, self.gradient_function, self.generate_edge_weights, 30, len(self.gamma), 10)

        self.gamma = sba.optimize()

        self.similarity_matrix = self.generate_edge_weights(self.gamma)
    
    def edge_weight_computation(self, section, gamma):

        res = []

        for idx in section:
            for vertex in range(self.data.shape[0]):
                if vertex != idx:
                    res.append((idx, vertex, self.similarity_function(idx, vertex, gamma)))
        
        return res

    def generate_edge_weights(self, gamma):
        print("Generating Edge Weights")

        curr_sim_matr = self.correct_similarity_matrix_diag(np.zeros_like(self.similarity_matrix))

        split_data = self.split(range(self.data.shape[0]), cpu_count())

        with Pool(processes=cpu_count()) as pool:
            edge_weight_res = [pool.apply_async(self.edge_weight_computation, (section, gamma)) for section in split_data]

            edge_weights = [edge_weight.get() for edge_weight in edge_weight_res]

        for section in edge_weights:
            for weight in section:
                if weight[0] != weight[1]:
                    #self.similarity_matrix[weight[0]][weight[1]] = weight[2]
                    #self.similarity_matrix[weight[1]][weight[0]] = weight[2]
                    curr_sim_matr[weight[0]][weight[1]] = weight[2]
                    curr_sim_matr[weight[1]][weight[0]] = weight[2]

        curr_sim_matr = self.subtract_identity(curr_sim_matr)

        #self.remove_disconnections()

        print("Edge Weight Generation Complete")

        return curr_sim_matr

    def subtract_identity(self, adj_matrix):
        identity = np.zeros((len(adj_matrix[0]), len(adj_matrix[0]))) 
        identity_diag = np.diag(identity)
        identity_diag_res = identity_diag + 2 
        np.fill_diagonal(identity, identity_diag_res) 
        adj_matrix = identity - adj_matrix
        return adj_matrix

    def remove_disconnections(self):
        empty_rows = np.all(self.similarity_matrix == 0, axis = 1)
        empty_cols = np.all(self.similarity_matrix == 0, axis = 0)
        self.similarity_matrix = self.similarity_matrix[~empty_rows, :][:, ~empty_cols]
        self.labels = self.labels.loc[~empty_rows]

    def unit_normalization(self, matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix/norms

    def get_eigenvectors(self, num_components, min_variance):
        pca = PCA()

        if num_components == 'lowest_var':

            pca.fit(self.similarity_matrix)

            expl_var = pca.explained_variance_ratio_
    
            print("Variance Distribution First Five: ", expl_var)

            cum_variance = expl_var.cumsum()

            num_components = ( cum_variance <= min_variance).sum() + 1

        pca = PCA(n_components=num_components)

        pca = pca.fit_transform(self.similarity_matrix)

        self.eigenvectors = self.unit_normalization(pca.real)

