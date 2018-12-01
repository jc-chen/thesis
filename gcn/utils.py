import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys, os, Cython
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, rdmolops, rdPartialCharges
import tensorflow as tf
from random import shuffle

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Creates a mask. 
       Masks are used to separate training, validation, and test sets from the full set of data.
       e.g. a set of 10 samples [1,2,3,..,10] could have samples 1,3,5,6,7 as training data, 2,4,8 for validation, and 
       9, 10 for testing. Then the mask for the training set would be [1,0,1,0,1,1,1,0,0,0], the validation mask would be
       [0,1,0,1,0,0,0,1,0,0], and the test mask would be [0,0,0,0,0,0,0,0,1,1]"""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def pickle_file(path, *args):
    """Saves parsed data under gcn/loaded_data/ as a text file.
       Usage: pickle_file(path, ('name of molecular data',data), ('another piece of data',data2), ('more',data3), ...)           
       e.g.:  pickle_file(path, ('adj', adj), ('features', features), ...) """
    for arg in args:
        file_name = path + str(arg[0]) + '.txt'
        print(file_name)
        with open(file_name,'wb') as file:
            pkl.dump(arg[1],file,protocol=2)
    return 

def load_pickled(path, *args):
    """Loads parsed data. 
       Usage:   load_pickled(path,'adj','features',...) """
    data = {}
    for arg in args:
        print("Loading "+str(arg)+".txt now")
        file_name = path + str(arg) + '.txt'
        with open(file_name,'rb') as file:
            data[str(arg)] = pkl.load(file)

    return data

def add_sample(url,features,target,As,sizes,num_molecules):
    """Extracts information from .xyz file and returns the features array, 
       target array, array of adjacency matrices, array of molecule sizes, and total
       number of molecules."""
    try:
        properties = [];
        with open(url,'r') as file:
            for row in file:
                properties += row.split()
        
        SMILES = properties[-4]
        #INCHI = properties[-2]
        m = Chem.MolFromSmiles(SMILES)
        #m = Chem.AddHs(m)    #uncomment to include hydrogen atoms
        
        vertices = m.GetAtoms()
        d = len(vertices)
        partial_charges=properties[22:(23+5*(d-1)):5]

        rdPartialCharges.ComputeGasteigerCharges(m)
        
        # Targets
        dipole_moment = float(properties[6])
        polarizability = float(properties[7])
        homo = float(properties[8])
        lumo = float(properties[9])
        gap = float(properties[10])
        r2 = float(properties[11])
        zpve = float(properties[12])
        U0 = float(properties[13])
        internal_energy = float(properties[14])
        enthalpy = float(properties[15])
        free_nrg = float(properties[16])
        heat_capacity = float(properties[17])
        
        # This is where you decide which properties you would like to predict
        target.append([heat_capacity])
        #target.append([dipole_moment,polarizability,homo,lumo,gap,
        #    r2,zpve,U0,internal_energy,enthalpy,free_nrg,heat_capacity])

        #populate the adjacency matrix and write features
        tempA = rdmolops.GetDistanceMatrix(m) #np.zeros((d,d)); #Adjacency matrix
        tempBO = np.zeros((d,d))
        tempAR = np.zeros((d,d))
        num_features = 11
        tempfeatures = [[0]*num_features for _ in range(d)]; # d=#nodes,  f=#features available

        for atom in vertices:
            # Get features of the atom
            v_i = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()
            degree = atom.GetDegree()
            valence = atom.GetTotalValence()
            hybrid = int(atom.GetHybridization())
            atom_aromatic = atom.GetIsAromatic()
            ring = atom.IsInRing()
            rad = atom.GetNumRadicalElectrons()
            pc = atom.GetDoubleProp("_GasteigerCharge")
            
            # Get bonds
            linked_atoms =[x.GetIdx() for x in atom.GetNeighbors()]

            # Populate adjacency matrix
            tempBO[v_i][v_i] = 10 #arbitrarily large number to indicate connectivity to self
            for v_j in linked_atoms:
                bond_order = m.GetBondBetweenAtoms(v_i,v_j).GetBondTypeAsDouble()
                #bond_length = m.GetBondBetweenAtoms(v_i,v_j).GetBondLength()
                bond_aromatic = m.GetBondBetweenAtoms(v_i,v_j).GetIsAromatic()
                tempBO[v_i][v_j] = bond_order
                tempAR[v_i][v_j] = bond_aromatic
            
            # Write features
            tempfeatures[v_i][0] = atomic_num
            tempfeatures[v_i][1] = int(atomic_num==1) #H
            tempfeatures[v_i][2] = int(atomic_num==6) #C
            tempfeatures[v_i][3] = int(atomic_num==7) #N
            tempfeatures[v_i][4] = int(atomic_num==8) #O
            tempfeatures[v_i][5] = int(atomic_num==9) #F
            tempfeatures[v_i][6] = int(atom_aromatic)
            tempfeatures[v_i][7] = hybrid
            tempfeatures[v_i][8] = degree
            tempfeatures[v_i][9] = valence
            tempfeatures[v_i][10] = atom.GetDoubleProp("_GasteigerCharge")

        As[0].append(sp.coo_matrix(tempA))
        As[1].append(sp.coo_matrix(tempBO))
        As[2].append(sp.coo_matrix(tempAR))
        
        if (num_molecules == 0):
            sizes = sizes + [d-1]
        else:
            sizes = sizes + [d]
        num_molecules=num_molecules+1

        if (num_molecules % 5000 == 0):
            print("On the "+str(num_molecules)+"th molecule")
        
        features+=tempfeatures
        return features, target, As, sizes, num_molecules
    except Exception as e:
        # Write exception to file
        print(str(e))
        # with open("analysis/problem_files.txt","a+") as file:
        #     file.write(str(num_molecules)+" :  "+str(url)+ "   " + str(e) + "\n")
        return features, target, A, sizes, num_molecules


def load_data(input_path, output_path=None, loadflag=False):
    """Loads molecular data. Returns an array of information required for training the neural network. """

    # load pickled data if available
    if (loadflag):
        return load_pickled(input_path,'adj','features','y_train',
            'y_val', 'y_test','train_mask','val_mask','test_mask','molecule_partitions','num_molecules')

    # initialize variables
    features = [] #features of each node
    A=[] #bond "graph distance" for now  #list of graph adjacency matrices; each entry is the adjacency matrix for one molecule
    BO = [] #adjacency matrix of bond order
    AR = [] #adjacency matrix - is_aromatic
    sizes = [] #list of sizes of molecules; each entry is the size (number of constituent atoms) of a molecule
    num_molecules = 0
    target = [] #list of "y's" - each entry is an "answer" for a molecule

    # load data from file
    for file in os.listdir(input_path):
        features, target, [A,BO,AR], sizes, num_molecules = add_sample(input_path+file,features,target,[A,BO,AR],sizes,num_molecules)

    print("Total molecules",num_molecules)
    
    molecule_partitions=np.cumsum(sizes) #to get partition positions

    # Divide into train, validation, test sets
    # train_size and val_size determine what fractions of the total dataset are used for training and validation
    randomized_order = list(range(num_molecules))
    shuffle(randomized_order)
    train_size = int(num_molecules*3/5)
    val_size = train_size + int(num_molecules/5)
    idx_train = randomized_order[0:train_size]
    idx_val = randomized_order[train_size:val_size]
    idx_test = randomized_order[val_size:]
    
    adj = [sp.csr_matrix(sp.block_diag(BO))]#,sp.csr_matrix(sp.block_diag(AR))]
    labels = np.array(target)

    #Create masks to separate the data set
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_test[test_mask] = labels[test_mask]
    y_val[val_mask] = labels[val_mask]

    features = np.array(features)

    #Normalize features
    elements = np.array([1,6,7,8,9])
    features[:,0] = features[:,0]/9.0
    # features[:,1] = (features[:,1] - np.mean(features[:,1]))/np.std(features[:,1])
    # # #features[:,0] = (features[:,0] - np.mean(features[:,0]))/np.std(features[:,0])
    # # #features[:,0] = (features[:,0]-elements_mean)/elements_stdev
    # features[:,2] = (features[:,2] - np.mean(features[:,2]))/np.std(features[:,2])
    # features[:,3] = (features[:,3] - np.mean(features[:,3]))/np.std(features[:,3])
    # features[:,4] = (features[:,4] - np.mean(features[:,4]))/np.std(features[:,4])

    features[:,7] = (features[:,7] - np.mean(features[:,7]))/np.std(features[:,7])
    features[:,8] = (features[:,8] - np.mean(features[:,8]))/np.std(features[:,8])
    features[:,9] = (features[:,9] - np.mean(features[:,9]))/np.std(features[:,9])
    features[:,10] = (features[:,10] - np.mean(features[:,10]))/np.std(features[:,10])
    
    features = sp.coo_matrix(features).tolil()

    print("About to write to file")

    if output_path is not None:
        pickle_file(output_path,('adj', adj), ('features', features), ('y_train', y_train), ('y_val', y_val), ('y_test', y_test), 
            ('train_mask', train_mask), ('val_mask', val_mask), ('test_mask', test_mask),
            ('molecule_partitions',molecule_partitions),('num_molecules',num_molecules))

        print("finished writing to file")

    data = {'adj': adj,
            'features': features,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask,
            'molecule_partitions': molecule_partitions,
            'num_molecules': num_molecules}

    return data


def load_test_data(data_path, output_path=None, loadflag=0):
    """Loads molecular data required to test a neural network."""
    features = [] #features of each node
    A=[] #list of graph adjacency matrices; each entry is the adjacency matrix for one molecule
    sizes = [] #list of sizes of molecules; each entry is the size of a molecule
    num_molecules = 0
    target = [] #list of "y's" - each entry is an "answer" for a molecule
    if (loadflag):
        return load_pickled(input_path,'adj','features','y','molecule_partitions','num_molecules')


    # Info for standardizing data
    elements = np.array([1,6,7,8,9])
    elements_mean = np.mean(elements)
    elements_stdev = np.std(elements)
    elements_all = [elements, elements_mean, elements_stdev]

    for file in os.listdir(data_path):
        features, target, A, sizes, num_molecules = add_sample(data_path+file,features,target,A,sizes,num_molecules,elements_all)

    print("Total molecules",num_molecules)

    molecule_partitions=np.cumsum(sizes) #to get partition positions

    adj = sp.csr_matrix(sp.block_diag(A))
    labels = np.array(target)

    features = sp.coo_matrix(np.array(features)).tolil()
    
    if output_path is not None:
        pickle_file(output_path,('adj', adj), ('features', features), ('y', labels),('molecule_partitions',molecule_partitions),('num_molecules',num_molecules))
        print("finished writing to file")

    data = {'adj': adj,
            'features': features,
            'y': labels,
            'molcule_partitions': molecule_partitions,
            'num_molecules': num_molecules}
    return data



def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Convert feature matrix to tuple representation."""
    """Uncomment to row-normalize here. Row-normalization has already been done under load_data().
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1.).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    """
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of a list of adjacency matrices for simple GCN model and conversion to tuple representation."""
    factor = 0. # Positive real number that adjusts the value of all diagonal entries in the adjacency matrices. 
    for i in range(len(adj)):
        adj_normalized = normalize_adj(adj[i] + factor*10.0*sp.eye(adj[i].shape[0]))
        adj[i] = sparse_to_tuple(adj_normalized)
    return adj


# features, support, labels, labels_mask, molecule_partitions, num_molecules 
def construct_feed_dict(data, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: data['labels']})
    feed_dict.update({placeholders['labels_mask']: data['labels_mask']})
    feed_dict.update({placeholders['features']: data['features']})
    feed_dict.update({placeholders['adjacency_matrices'][i]: data['adjacency_matrices'][i] for i in range(len(data['adjacency_matrices']))})
    feed_dict.update({placeholders['num_features_nonzero']: data['features'][1].shape})
    feed_dict.update({placeholders['molecule_partitions']: data['molecule_partitions']})
    feed_dict.update({placeholders['num_molecules']: data['num_molecules']})

    # optional ones
    if 'dropout' in data:
        feed_dict.update({placeholders['dropout']: data['dropout']})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def tensor_diff(self, input_tensor):
    # builds an m by m matrix with 1s in the diagonal and -1s on the column left of the diagonal
    # for example for 3 molecules:
    #  [1   0  0]
    #  [-1  1  0]
    #  [0  -1  1]
    # and multiplies it by the input vector to get a difference vector
    # for example, if V = [v1 v2 v3] then the function returns [v1 v2-v1 v3-v2]

    A = tf.eye(self.num_molecules)
    B = tf.pad(tf.negative(tf.eye(tf.subtract(self.num_molecules,tf.constant(1)))), tf.constant([[1, 0,], [0, 1]]), "CONSTANT")
    d_tensor = tf.add(A,B)
    out = tf.matmul(d_tensor,input_tensor, name="output_after_tensorDiff")
    return out


def sparse_matrix_to_sparse_tensor(coo):
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def print_learn_rate(rate):
    print("......\n......\n ......Changing learning rate to: ", rate, "\n......\n......")
    return


### The following functions add_sample_distribution() and load_data-distribution()
### were used to plot data distribution graphs in Fig. 3.2 of section 3.3.1
def add_sample_distribution(url,features,target,As,sizes,num_molecules):
    '''Same function as add_sample() but used when extracting distribution information'''

    try:
        properties = [];
        with open(url,'r') as file:
            for row in file:
                properties += row.split()
        
        SMILES = properties[-4]
        m = Chem.MolFromSmiles(SMILES)
        m = Chem.AddHs(m)
        vertices = m.GetAtoms()
        d = len(vertices)
        partial_charges=properties[22:(23+5*(d-1)):5]

        # Targets
        dipole_moment = float(properties[6])
        polarizability = float(properties[7])
        homo = float(properties[8])
        lumo = float(properties[9])
        gap = float(properties[10])
        r2 = float(properties[11])
        zpve = float(properties[12])
        U0 = float(properties[13])
        internal_energy = float(properties[14])
        enthalpy = float(properties[15])
        free_nrg = float(properties[16])
        heat_capacity = float(properties[17])
  
        target.append([dipole_moment, polarizability, homo, lumo, gap, r2,zpve,U0, internal_energy,
            enthalpy, free_nrg, heat_capacity])
        #populate the adjacency matrix and write features
        tempA = rdmolops.GetDistanceMatrix(m) #np.zeros((d,d)); #Adjacency matrix
        tempBO = np.zeros((d,d))
        tempAR = np.zeros((d,d))
        f = 12
        tempfeatures = [[0]*f for _ in range(d)]; # d=#nodes,  f=#features available
        for atom in vertices:
            # Get features of the atom
            v_i = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()
            degree = atom.GetDegree()
            valence = atom.GetTotalValence()
            hybrid = int(atom.GetHybridization())
            atom_aromatic = atom.GetIsAromatic()
            ring = atom.IsInRing()
            rad = atom.GetNumRadicalElectrons()

            # Get bonds
            linked_atoms =[x.GetIdx() for x in atom.GetNeighbors()]

            # Populate adjacency matrix
            tempBO[v_i][v_i] = 10 #arbitrarily large number to indicate connectivity to self
            for v_j in linked_atoms:
                bond_order = m.GetBondBetweenAtoms(v_i,v_j).GetBondTypeAsDouble()
                #bond_length = m.GetBondBetweenAtoms(v_i,v_j).GetBondLength()
                bond_aromatic = m.GetBondBetweenAtoms(v_i,v_j).GetIsAromatic()
                tempBO[v_i][v_j] = bond_order
                tempAR[v_i][v_j] = bond_aromatic
            
            tempfeatures[v_i][0] = atomic_num
            tempfeatures[v_i][1] = int(atomic_num==1) #H
            tempfeatures[v_i][2] = int(atomic_num==6) #C
            tempfeatures[v_i][3] = int(atomic_num==7) #N
            tempfeatures[v_i][4] = int(atomic_num==8) #O
            tempfeatures[v_i][5] = int(atomic_num==9) #F
            tempfeatures[v_i][6] = int(atom_aromatic)
            tempfeatures[v_i][7] = hybrid
            tempfeatures[v_i][8] = atom.GetDoubleProp("_GasteigerCharge")
            tempfeatures[v_i][9] = valence
            tempfeatures[v_i][10] = degree
            tempfeatures[v_i][11] = ring

        As[0].append(sp.coo_matrix(tempA))
        As[1].append(sp.coo_matrix(tempBO))
        As[2].append(sp.coo_matrix(tempAR))
        
        if (num_molecules == 0):
            sizes = sizes + [d-1]
        else:
            sizes = sizes + [d]
        num_molecules=num_molecules+1
        if (num_molecules % 5000 == 0):
            print("On the "+str(num_molecules)+"th molecule")
        features+=tempfeatures
        return features, target, As, sizes, num_molecules
        
    except Exception as e:
        # Write exception to file
        print(str(e))
        # TODO decide how to handle writing to file
        # with open("analysis/problem_files.txt","a+") as file:
        #     file.write(str(num_molecules)+" :  "+str(url)+ "   " + str(e) + "\n")
        return features, target, A, sizes, num_molecules

def load_data_distribution(data_path,pickle_flag=0):
    '''Same function as load_data() but used for acquiring distribution information'''
    features = [] #features of each node
    A=[] #bond "graph distance" for now  #list of graph adjacency matrices; each entry is the adjacency matrix for one molecule
    BO = [] #adjacency matrix of bond order
    AR = [] #adjacency matrix - isaromatic

    sizes = [] #list of sizes of molecules; each entry is the size of a molecule
    num_molecules = 0
    target = [] #list of "y's" - each entry is an "answer" for a molecule

    for file in os.listdir(data_path):
        features, target, [A,BO,AR], sizes, num_molecules = add_sample_distribution(data_path+file,features,target,[A,BO,AR],sizes,num_molecules)

    #features, target, adj, sizes, num_molecules =load_pickled('distribution_set/plots/full/data/','features','target','adj','sizes','num_molecules')
    #pickle_file('distribution_set/plots/full/data/',('features',features),('target',target),
    #    ('adj',[A,BO,AR]),('sizes',sizes),('num_molecules',num_molecules))
    target = np.array(target)
    target_mean = np.mean(target,axis=0)
    target_std = np.std(target,axis=0)

    labels = target
    nbins=100
    # names = ["Dipole moment ($\mu$)", "Isotropic polarizability ($\\alpha$)", "HOMO", "LUMO", "HOMO/LUMO energy gap ($\Delta$E)", 
    #          "Electronic spatial extent ($r^2$)", "Zero point vibrational energy ($zpve$)", 
    #          "Internal energy at 0K  (U$_0$)", "Internal energy at 298.15K (U)", "Enthalpy at 298.15K (H)",
    #          "Free energy at 298.15K (G)","Specific heat at 298.15K (C$_v$)"]
    # namesaxis = ["$\mu$", "$\\alpha$", "HOMO", "LUMO", "$\Delta$E", 
    #          "$r^2$", "$zpve$", 
    #          "U$_0$", "U", "H",
    #          "G","C$_v$"]
    # names2 = ["dipolemoment", "polarizability", "HOMO", "LUMO", "Gap", 
    #         "r2", "zpve", "U0", "U", "H", "G","Cv"]
    
    # units = ["Debye","Bohr$^3$","Hartree","Hartree","Hartree","Bohr$^2$","Hartree","Hartree","Hartree","Hartree","Hartree", "cal/(mol$\cdot$K)"]
    # for i in range(len(labels[0])):
    #     histy,histx = np.histogram(labels[:,i],bins=nbins)
    #     fig,ax = plt.subplots()
    #     ax.plot(histx[:-1],histy)
    #     plt.title("Distribution of " + names[i] + " (" + str(nbins) + " bins)")
    #     plt.xlabel(namesaxis[i] + " (" + units[i] + ")")
    #     plt.ylabel("Count")
    #     plt.savefig("./loaded_data/distribution_set/plots/full/pics/" + str(nbins) + names2[i] + ".jpg")
    #     #plt.show()
    #     pickle_file('distribution_set/plots/full/hists/',(str(nbins) + names2[i]+'y', histy),(str(nbins) + names2[i]+'x',histx))

    names2 = []
    features = np.array(features)
    for i in range(len(features[0])):
        histy,histx = np.histogram(features[:,i],bins=nbins)
        fig,ax = plt.subplots()
        ax.plot(histx[:-1],histy)
        plt.title("Distribution of feature " + str(i))
        plt.xlabel("Feature")
        plt.ylabel("Count")
        pickle_file('distribution_set/features/1k/hists/',(str(nbins) + names2[i]+'y', histy),(str(nbins) + names2[i]+'x',histx))

    return

