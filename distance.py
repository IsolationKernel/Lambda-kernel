from   fermat        import Fermat
from   scipy.spatial import  distance_matrix
from   ripser        import Rips
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
from Lambda_feature import *
from DTM_filtrations import *
from sklearn.manifold import MDS
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
import os

def get_mds(D):
    embedding = MDS(n_components=2,dissimilarity='precomputed')
    X_mds = embedding.fit_transform(D)
    return X_mds

def IK_app(dm1,dm2):
    neigh = NearestNeighbors(n_neighbors=1,metric='precomputed')
    neigh.fit(dm1)
    NN_dist,NN = neigh.kneighbors(dm2,return_distance=True)
    return NN_dist*4

def CKNN_approximate(X,query_pts,m,dm=None):
    print("CKNN")
    N_tot = X.shape[0]     
    k = math.floor(m*N_tot)+1   # number of neighbors
    print('k = ',k)
    kdt = KDTree(X, leaf_size=30, metric='euclidean')
    NN_Dist, NN = kdt.query(X, k, return_distance=True)  
    NN_Dist = NN_Dist.reshape((N_tot,k))
    knn_dist = NN_Dist[:,-1]
    knn_dist = knn_dist.reshape((N_tot,1))
    dm = distance_matrix(X,query_pts,p=2)
    dm_cknn = dm/knn_dist
    cknn_fv = np.min(dm_cknn,axis=0)
    return cknn_fv*2

def deal_inf(X):
    # set infity to be twice of the maximum distance
    re = np.max(X[np.where(np.isfinite(X))])
    X[np.where(np.isinf(X))] = re*4
    return X

def compute_Riemannian_distance(X,k):
    # Appximate Riemannian distance with k-NN graph
    mat = kneighbors_graph(X, k, mode='distance', include_self=False)
    W = mat.todense()
    A = W>0
    A = A.astype(int)
    mA = np.minimum(A,A.T) # mutual k-NN graph
    mW = A*W
    graph = csr_matrix(mW)
    dist_matrix = shortest_path(csgraph=graph, directed=False)
    return deal_inf(dist_matrix)

def compute_cknn_distance(data,m,dm=None,k=None):
    # Compute CkNN distance
    if k is None:
        k = int(data.shape[0]*m)+1

    if dm is None:
        dm = distance_matrix(data,data,p=2)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
        distances, indices = nbrs.kneighbors(data)
    else:
        print("Using input distance")
        nbrs = NearestNeighbors(n_neighbors=k,metric='precomputed').fit(dm)
        distances, indices = nbrs.kneighbors(dm)

        
    n,d = distances.shape
    col = distances[:,-1].reshape((n,1))
    normalize_term = np.sqrt(np.dot(col,col.T))
    assert np.count_nonzero(np.abs(normalize_term-0)<=np.finfo(float).eps)==0
    assert dm.shape == normalize_term.shape
    dm_cknn = np.divide(dm,normalize_term)
    return dm_cknn

def compute_lambda_distance(dm,eta,psi,t=100):
    # Compute Lambda-distance
    if eta == -1:
        _,dm_lambda = lambda_feature_infty_dm(dm,psi,t)
    elif eta > 0:
        _,dm_lambda = lambda_feature_continous_dm(dm,eta,psi,t)
    return dm_lambda
    
def compute_fermat_distance_D(data, p, k):
    
    #Compute euclidean distances
    distances = distance_matrix(data, data)
    
    # Initialize the model
    fermat = Fermat(alpha = p, path_method='D',k=k) #method Dijkstra

    # Fit
    fermat.fit(distances)
    
    ##Compute Fermat distances
    fermat_dist = fermat.get_distances()
    
    return  fermat_dist

def compute_fermat_distance(data, p, dm = None):    
    '''
    Computes the sample Fermat distance.
    '''
    if dm is None:
        #Compute euclidean distances
        distances = distance_matrix(data,data)
    else:
        distances = dm
    
    # Initialize the model
    fermat = Fermat(alpha = p, path_method='FW')  # method Floyd-Warshall

    # Fit
    fermat.fit(distances)
    
    ##Compute Fermat distances
    fermat_dist = fermat.get_distances()
    
    return  fermat_dist


def compute_kNN_distance(data, k):
    '''
    Computes the  estimator of geodesic distance using kNN graph.
    '''
    
    distances = distance_matrix(data,data)

    # Initialize the model
    f_aprox_D = Fermat(1, path_method='D', k=k) 

    # Fit
    f_aprox_D.fit(distances)
    adj_dist = f_aprox_D.get_distances() 
    
    return adj_dist

def Fermat_dgm(data, p, dm = None, rescaled=False, d=None, mu=None, title=None):
    '''
    Computes the persistence diagram using Fermat distance.
    '''
    if dm is not None:
        print('Using Input Distance to Compute Fermat Distance')
        distance_matrix = compute_fermat_distance(data, p, dm=dm)
    else:
        distance_matrix = compute_fermat_distance(data, p)

    X_mds = get_mds(distance_matrix)
    f = plt.figure()
    plt.scatter(X_mds[:,0],X_mds[:,1],s=1)
    plt.axis('equal')
    plt.title('Fermat')

    if rescaled:
        distance_matrix = (distance_matrix*len(data)**((p-1)/d))/mu
    rips = Rips()
    dgms = rips.fit_transform(distance_matrix, distance_matrix=True)
    # fig = plt.figure()
    # rips.plot(dgms, lifetime=True)
    # if title==None:
    #     plt.title('Fermat distance with p = %s'%(p))
    # else:
    #     plt.title(title)
    return dgms

def get_pd_dtm(X,m,dm=None):
    # get persistence diagram from DTM
    p = 1
    dimension_max = 2
    # creating a simplex tree
    if dm is not None:
        print('Using Input distance')
        st_DTM = DTMFiltration_DM(dm, m, p, dimension_max)
    else:
        st_DTM = DTMFiltration(X, m, p, dimension_max)  

    diag= st_DTM.persistence()
    diag_1 = [(1,list(x[1])) for x in diag if x[0]==1]
    diag_0 = [(0,list(x[1])) for x in diag if x[0]==0]
    return diag,diag_0,diag_1

def get_pd_dm(dm):
    # compute dim-0,dim-1 PD from distance matrix (Rips)
    eps = 0.001
    rps = gd.RipsComplex(distance_matrix=dm,max_edge_length=np.max(dm)+eps)
    rps_tree = rps.create_simplex_tree(max_dimension=2)
    diag = rps_tree.persistence()
    diag_1 = np.array([(1,list(x[1])) for x in diag if x[0]==1])
    diag_0 = np.array([(0,list(x[1])) for x in diag if x[0]==0])
    return diag,diag_0,diag_1

def plot_PD(dgm,dgm_0,dgm_1,para = '',sf = True):

    folder = 'Appendix'
    if not os.path.exists(folder):
        os.mkdir(folder)

    gd.plot_persistence_diagram(dgm,legend=True)
    if sf: 
        plt.savefig(folder +os.sep+ 'pd_01_'+para+'.png')
    gd.plot_persistence_diagram(dgm_0,legend=True)
    if sf:
        plt.savefig(folder +os.sep+ 'pd_0_'+para+'.png')
    gd.plot_persistence_diagram(dgm_1,legend=True)
    if sf:
        plt.savefig(folder +os.sep+ 'pd_1_'+para+'.png')
    gd.plot_persistence_barcode(dgm_1,legend=True)
    if sf:
        plt.savefig(folder +os.sep+ 'barcode_1_'+para+'.png')

def gd_format(dgms_ripser):
    # change PD format from ripser to pd
    dgm0,dgm1 = dgms_ripser[0],dgms_ripser[1]
    test_1 = [(1,x) for x in dgm1]
    test_0 = [(0,x) for x in dgm0]
    test = test_0 + test_1
    return test,test_0,test_1

def get_mds(D):
    embedding = MDS(n_components=2,dissimilarity='precomputed')
    X_mds = embedding.fit_transform(D)
    return X_mds


