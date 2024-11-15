import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import isclose
import math
import itertools
from multiprocessing import Pool
from multiprocessing import cpu_count

import plotly.express as px
import plotly.graph_objects as go

# Define the class that contains the objective computation function
class OptimizationFunction:
    def __init__(self, data=None, similarity_matrix=None, gamma=None):
        self.data = data
        self.similarity_matrix = similarity_matrix
        self.gamma = gamma
        
    def objective_computation(self, section):
        approx_error = 0
        for idx in section:
            degree_idx = np.sum(self.similarity_matrix[idx])
            xi_reconstruction = np.sum([self.similarity_matrix[idx][y] * np.asarray(self.data.loc[[y]])[0] 
                                        for y in range(len(self.similarity_matrix[idx])) if idx != y], 0)            

            if degree_idx != 0 and not isclose(degree_idx, 0, abs_tol=1e-100):
                xi_reconstruction /= degree_idx
                xi_reconstruction = xi_reconstruction[0]
            else:
                xi_reconstruction = np.zeros(2)

            # Calculate the error (difference between reconstruction and original data)
            approx_error += np.sum((np.asarray(self.data.loc[[idx]])[0] - xi_reconstruction)**2)
        return approx_error

    def objective_function(self, adj_matr):
        self.similarity_matrix = adj_matr
        split_data = self.split(range(len(adj_matr[0])), cpu_count())
        with Pool(processes=cpu_count()) as pool:
            errors = [pool.apply_async(self.objective_computation, (section, )) \
                                                                 for section in split_data]

            error = [error.get() for error in errors]
        return np.sum(error)

    def split(self, a, n):
        k, m = divmod(len(a), n)
        return [a[i*k+min(i,m):(i+1)*k+min(i+1,m)] for i in range(n)]


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def plot_error_surface(aew_obj):

    # Create an instance of the optimization function
    opt_function = OptimizationFunction(data = aew_obj.data, similarity_matrix = aew_obj.similarity_matrix)
    
    values = np.arange(-10, 10.1, .10)

    sim_gammas = np.asarray([list(pair) for pair in itertools.product(values, repeat=2)])

    objective_values = np.zeros(sim_gammas.shape[0])

    for idx, gamma in enumerate(sim_gammas):
        print("Gamma: ", gamma)
        curr_adj_matr = aew_obj.generate_edge_weights(gamma)

        #objective_values.append(opt_function.objective_function(curr_adj_matr))
        objective_values[idx] = opt_function.objective_function(curr_adj_matr)

    # Plotting the surface
    X = np.unique(sim_gammas[:,0]) #np.linspace(0, 10, 10)  # X-axis points (random for illustration)
    Y = np.unique(sim_gammas[:,1])  #np.linspace(0, 10, 10)  # Y-axis points (random for illustration)

    Z = np.zeros((len(X), len(Y)))  

    for gamma, obj_val in zip(sim_gammas, objective_values):
        x_idx = np.where(X == gamma[0])[0][0]
        y_idx = np.where(Y == gamma[1])[0][0]
        Z[x_idx, y_idx] = obj_val

    X, Y = np.meshgrid(X, Y)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor='limegreen', project_z=True))

    fig.update_layout(
            title = 'Error Surface',
            scene=dict(
            xaxis=dict(range=[X.min(), X.max()]),
            yaxis=dict(range=[Y.min(), Y.max()]),
            zaxis=dict(title="Error", range=[Z.min(), Z.max()])
            )
    )

    fig.show()

    #print(len(x))
    #print(len(y))

    #print(len(Z))

    # Create a surface plot
    #fig = plt.figure(figsize=(10, 7))
    #ax = fig.add_subplot(111, projection='3d')

    # Plot a surface
    #ax.plot_surface(X, Y, Z, cmap='viridis')

    # Add labels and title
    #ax.set_xlabel('X Label')
    #ax.set_ylabel('Y Label')
    #ax.set_zlabel('Objective Value')
    #ax.set_title('Surface of the Objective Function')

    #plt.show()

