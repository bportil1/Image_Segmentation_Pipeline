from spread_opt import *
from preprocessing_utils import *
from clustering import *
from aew_surface_plotter import *


from sklearn.metrics import accuracy_score


import warnings

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    #ids_train_file = '/home/bryan_portillo/Desktop/network_intrusion_detection_dataset/Train_data.csv'

    ids_train_file = '/media/mint/NethermostHallV2/py_env/venv/network_intrusion_detection_dataset/Train_data.csv'

    #ids_train_file = '/home/bryanportillo_lt/Documents/py_env/venv/network_intrusion_dataset/Train_data.csv'
   
    #ids_train_file = 'e:/py_env/venv/network_intrusion_detection_dataset/Train_data.csv'
    
    opt_cycles = [30, 35, 40, 45,50]

    opt_cycles = [10]
    '''
    opt_cycles = [2, 5, 10, 25, 30, 35, 40, 45,50]

    opt_cycles = [5]
    
    for rep in range(5):

        for cycle in opt_cycles:
    
            synthetic_data_tester(rep, cycle)
    '''     
    
        #synthetic_data_tester(rep)


    data_obj = data(datapath = ids_train_file)

    data_obj.load_data(400)

    data_obj.load_labels()

    data_obj.encode_categorical('protocol_type', 'data')

    data_obj.encode_categorical('service', 'data')

    data_obj.encode_categorical('flag', 'data')

    data_obj.encode_categorical('class', 'labels')

    data_obj.scale_data('min_max')

    data_obj.generate_graphs(100)

    aew_obj = aew(data_obj.graph.toarray(), data_obj.data, data_obj.labels, 'var')

    aew_obj.similarity_matrix = aew_obj.generate_edge_weights(aew_obj.gamma)

    #aew_obj.get_eigenvectors(2, .90)

    pca = PCA(n_components=2)

    aew_obj.data = pd.DataFrame(pca.fit_transform(data_obj.data))

    plot_error_surface(aew_obj)

    #aew_obj.similarity_matrix = aew_obj.

    #opt_cycles = [10000]

    #test_diag_file = open("errorvopt.txt", "a")
    '''
    for opt_steps in opt_cycles:
        for rep in range(1):
            data_obj = data(datapath = ids_train_file)

            data_obj.load_data(500)

            data_obj.load_labels()

            data_obj.encode_categorical('protocol_type', 'data')

            data_obj.encode_categorical('service', 'data')

            data_obj.encode_categorical('flag', 'data')

            data_obj.encode_categorical('class', 'labels')

            data_obj.scale_data('min_max')

            data_obj.generate_graphs(150)

            data_obj.generate_graphs(150)

            diag_base = str(rep) + "," + str(opt_steps) + ","

            dir_name = 'results_' + str(rep) + '_' + str(opt_steps) 

            aew_obj = aew(data_obj.graph.toarray(), data_obj.data, data_obj.labels)

            aew_obj.generate_optimal_edge_weights(opt_steps)

            error_str = diag_base + str(aew_obj.final_error) + "\n"

            test_diag_file.write(error_str)

            os.makedirs(str('./'+dir_name+'/plain_data/twod/'), exist_ok=True)
    
            os.makedirs(str('./'+dir_name+'/plain_data/threed/'), exist_ok=True)

            os.makedirs(str('./'+dir_name+'/eigen_data/twod/'), exist_ok=True)

            os.makedirs(str('./'+dir_name+'/eigen_data/threed/'), exist_ok=True)

            ###### Test 2d Data

            aew_obj.get_eigenvectors(2, .90)

            ###### Original Data Test

            ##base Daataa

            visualizer_obj = visualizer(data_obj.labels, 2)

            visualizer_obj.lower_dimensional_embedding(data_obj.data.to_numpy(), "orig_data_2d.html", str("./"+dir_name+"/plain_data/"), downsize=True)

            ####clustering the whole regulaar data

            clustering_obj = clustering(base_data=data_obj.data.to_numpy(), data=aew_obj.data, labels = aew_obj.labels, path_name = str("./"+dir_name+"/plain_data/twod/"),
name_append='whole_regular_2d_data', workers=-1)

            clustering_obj.generate_spectral()

            visualizer_obj = visualizer(clustering_obj.pred_labels, 2)

            visualizer_obj.lower_dimensional_embedding(data_obj.data.to_numpy(), "plain_data_90_perc_var_2d.html", str("./"+dir_name+"/plain_data/"))

            ###### Eigenvector Data Test

            clustering_obj = clustering(base_data = data_obj.data.to_numpy(), data=aew_obj.eigenvectors, labels=aew_obj.labels, path_name = str("./"+dir_name+"/"), name_append="eigenvector_2d_data", workers=-1)
    
            clustering_obj.generate_spectral()

            visualize_obj = visualizer(clustering_obj.pred_labels, 2)

            visualizer_obj.lower_dimensional_embedding(data_obj.data.to_numpy(),  "eigen_data_90_perc_var_2d.html", str("./"+dir_name+"/eigen_data/"))

            ###### Test 3d Data

            aew_obj.get_eigenvectors(3, .90)

            ###### Original Data Test

            visualizer_obj = visualizer(data_obj.labels, 3)
    
            visualizer_obj.lower_dimensional_embedding(data_obj.data.to_numpy(), "orig_data_3.html", str("./"+dir_name+"/plain_data/"))

            clustering_obj = clustering(base_data = data_obj.data.to_numpy(), data=aew_obj.data, labels = aew_obj.labels, path_name = str("./"+dir_name+"/plain_data/threed/"), name_append='whole_regular_3d_data', workers=-1)

            clustering_obj.generate_spectral()

            visualizer_obj = visualizer(clustering_obj.pred_labels, 3)

            visualizer_obj.lower_dimensional_embedding(data_obj.data.to_numpy(), "plain_data_90_perc_var_3d", str("./"+dir_name+"/plain_data/"))

            ###### Eigenvector Data Test

            clustering_obj = clustering(base_data = data_obj.data.to_numpy(), data=aew_obj.eigenvectors, labels=aew_obj.labels, path_name = str("./"+dir_name+"/"), name_append="eigenvector_3d_data", workers=-1)

            clustering_obj.generate_spectral()

            visualizer_obj = visualizer(clustering_obj.pred_labels, 3)

            visualizer_obj.lower_dimensional_embedding(aew_obj.data.to_numpy(),  "eigen_data_90_perc_var_3d.html", str("./"+dir_name+"/eigen_data/"))
    
    '''
'''

    clustering_with_adj_matr_prec_kmeans = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans', n_jobs=-1)

    print("Kmeans Train: ", accuracy_score(clustering_with_adj_matr_prec_kmeans.fit_predict(aew_train.eigenvectors), aew_train.labels))

    clustering_with_adj_matr_prec_disc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='discretize', n_jobs=-1)

    print("Discretize Train: ", accuracy_score(clustering_with_adj_matr_prec_disc.fit_predict(aew_train.eigenvectors), aew_train.labels))

    clustering_with_adj_matr_prec_clust = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='cluster_qr', n_jobs=-1)

    print("Cluster_qr Train: ", accuracy_score(clustering_with_adj_matr_prec_clust.fit_predict(aew_train.eigenvectors), aew_train.labels))

    clustering_with_adj_matr_prec_kmeans1 = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans', n_jobs=-1)

    print("Kmeans Test: ", accuracy_score(clustering_with_adj_matr_prec_kmeans1.fit_predict(aew_test.eigenvectors), aew_test.labels))

    clustering_with_adj_matr_prec_disc1 = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='discretize', n_jobs=-1)

    print("Discretize Test: ", accuracy_score(clustering_with_adj_matr_prec_disc1.fit_predict(aew_test.eigenvectors), aew_test.labels))

    clustering_with_adj_matr_prec_clust1 = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='cluster_qr', n_jobs=-1)

    print("Cluster_qr Test: ", accuracy_score(clustering_with_adj_matr_prec_clust1.fit_predict(aew_test.eigenvectors), aew_test.labels))
    
    
    plain_graph_clustering = clustering(aew_train.eigenvectors, aew_train.labels, aew_test.eigenvectors, aew_test.labels, "full", "40_dim_no_proj_graph_data", clustering_methods=clustering_meths,  workers = -1)

    plain_graph_clustering.generate_clustering()
'''
'''
    num_components = [3, 8, 12, 15, 20, 40]

    for num_comp in num_components:

        print("Current number of components: ", num_comp)

        data_obj.train_projection, _ = data_obj.downsize_data(aew_train.eigenvectors, 'train', num_comp)

        data_obj.test_projection, _ = data_obj.downsize_data(aew_test.eigenvectors, 'test', num_comp)
        init_path = './results/orig_data_visualization/num_comp_' + str(num_comp) + '/'

        os.makedirs(init_path, exist_ok=True)

        for projection in data_obj.train_projection.keys():

            #print("Train NaNs: ", np.count_nonzero(np.isnan(data_obj.train_projection[projection])))    

            #print("Test NaNs: ", np.count_nonzero(np.isnan(data_obj.test_projection[projection])))

            data_obj.lower_dimensional_embedding(data_obj.train_projection[projection], 'train', 'Train Mappings Base: 3-Dimensions', init_path)

            data_obj.lower_dimensional_embedding(data_obj.test_projection[projection], 'test', 'Test Mappings Base: 3-Dimensions', init_path)

            clustering_graph_data = clustering(data_obj.train_projection[projection], data_obj.train_labels, data_obj.test_projection[projection], data_obj.test_labels, num_comp, projection, clustering_methods=clustering_meths, workers = -1)

            clustering_graph_data.generate_clustering()
    '''
