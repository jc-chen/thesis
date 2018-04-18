import numpy as np;
import Cython;
import molmod as mm;

import sys, os
import tensorflow as tf
import scipy.sparse as sp


d = 5; #no. features (size of largest molecule)
n = 6; #no. samples
X = np.zeros((n,d));
y = np.zeros(n);
A = np.zeros((n*d,n*d));
path = "../../tem/";

def add_sample(url,inputMatrix,outputArray,i,n,d): 
	
	properties = [];
	with open(url,'r') as file:
		for row in file:
			properties += row.split();

	#print(url)
	#extract information from xyz file
	mol = mm.Molecule.from_file(url);
	#mol.write_to_file("new.xyz");
	mol.graph = mm.MolecularGraph.from_geometry(mol);
	vertices = mol.graph.numbers;
	edges = mol.graph.edges;
	#print(vertices)
	inputMatrix[i][0:len(vertices)] = vertices #[vertices[v] for v in vertices,0 for j in range(len(inputMatrix[i])-len(vertices)-1)];
	outputArray[i] = float(properties[15]);

	tempA = np.zeros((d,d)); #Adjacency matrix

	#populate the adjacency matrix
	for tupl in edges:
		tuple_list = list(tupl);
		v_i = tuple_list[0];
		v_j = tuple_list[1];
		tempA[v_i][v_j] = 1;
		tempA[v_j][v_i] = 1;
	A[i*d:(i+1)*d,i*d:(i+1)*d] = tempA;
	return;



i=0;
#maxs=1.0;
for file in os.listdir(path):
	add_sample(path+file,X,y,i,n,d);
	i += 1;
	#with open("../usb_mount/qm9-gdb9-133885/"+file,'r') as f:
		#line = f.readline();
		#print(line);
		#maxs=max(float(line),maxs)
#print(maxs)

sA = sp.csr_matrix(A);

