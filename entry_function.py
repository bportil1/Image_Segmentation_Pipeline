from aew import *
from preprocessing_utils import *
from clustering import *
from aew_surface_plotter import *

from sklearn.metrics import accuracy_score


import warnings

warnings.filterwarnings("ignore")

def aew_test_driver():
    '''
    Function to run AEW clustering test
    '''

    cm1_file = 'sq_ds/cm1.csv'

    data_obj = data(cm1_file, graph_type='stratified')

    data_obj.load_data()

    data_obj.load_labels()

    data_obj.encode_categorical('defects', 'labels')

    data_obj.scale_data('min_max')

    for rep in range(5):
        
        diag_base = str(rep) + "," 

        dir_name = 'results_' + str(rep)  

        os.makedirs(str('./'+dir_name+'/plain_data/'), exist_ok=True)
        
        for strat_idx in range(25):

            print("Stratified Section: ", strat_idx)

            data_obj.generate_graphs(100, strat_idx, data_type='stratified')

            aew_obj = aew(data_obj.graph, data_obj.stratified_data[strat_idx], data_obj.data, data_obj.stratified_labels[strat_idx], strat_idx, 'var')

            if strat_idx > 0:
                aew_obj.gamma = curr_gamma

            aew_obj.generate_optimal_edge_weights(1000)
            
            curr_gamma = aew_obj.gamma
        
        aew_obj.data = data_obj.data

        data_obj.generate_graphs(100, data_type="whole")

        aew_obj.similarity_matrix = aew_obj.correct_similarity_matrix_diag(data_obj.graph)

        aew_obj.similarity_matrix = aew_obj.generate_edge_weights(aew_obj.gamma)
        
        #plot_error_surface(aew_obj)

        aew_obj.get_eigenvectors('lowest_var', .90)

        ###### Original Data Test

        visualizer_obj = visualizer(data_obj.labels, 3)

        visualizer_obj.lower_dimensional_embedding(data_obj.data.to_numpy(), "orig_data_3.html", str("./"+dir_name+"/plain_data/"))

        ###### Eigenvector Data Test

        clustering_obj = clustering(base_data = data_obj.data.to_numpy(), data=aew_obj.eigenvectors, labels=data_obj.labels, path_name = str("./"+dir_name+"/"), name_append="eigenvector_3d_data", workers=-1)

        clustering_obj.generate_spectral()    

        clustering_obj.generate_kmeans()

        clustering_obj.generate_gaussianmixture()

if __name__ == '__main__':
    aew_test_driver()
        
