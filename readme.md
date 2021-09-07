# FixLocator: Fault Localization to Detect Co-Change Fixing Locations

<p aligh="center"> This repository contains the code for <b>Fault Localization to Detect Co-Change Fixing Locations</b> and the Page (https://github.com/fixlocatorresearch/fixlocator-public) that has some visualizd data. </p>

## Contents
1. [Website](#Website)
2. [Introduction](#Introduction)
3. [Dataset](#Dataset)
4. [Requirement](#Requirement)
5. [Instruction](#Instruction)


## Website

We published the visualizd experimental results reported in the paper in Page(https://github.com/fixlocatorresearch/fixlocator-public).


## Introduction

We present FixLocator, a DL-based fault localization (FL) approach
supporting the detection of faulty statements in one or multiple
methods that need to be modified accordingly in the same fix. Let 
us call them co-change (CC) fixing locations for a fault. We treat
this FL problem as a dual learning task with two models. First,
the method-level FL model, MethFL, learns the methods to be fixed
together. Second, the statement-level FL model, StmtFL, learns the
statements to be co-fixed. Correct learning in a model can benefit
the other and vice versa. Exploring this duality provides useful
constraints for FixLocator to learn derive CC fixing statements.
Thus, we simultaneously train them with soft-sharing the models’
parameters via cross-stitch units to exploit this duality. In a crossstitch
unit, the sharing of representations between MethFL and StmtFL
is modeled by the learning a linear combination of the input features
from two models. The cross-stitch units enable the propagation
of the impact of method-level FL on statement-level FL and vice
versa. In addition to the new dual learning model solution, we also
explore a novel feature, which is the co-change information among
statements.We use Graph-based Convolution Network to integrate
different types of program dependencies among the statements. Our
empirical evaluation on real-world datasets shows that FixLocator
relatively improves over the state-of-the-art statement-level FL
baselines by locating more CC fixing statements from 26.5% to
155.6%, and reduces the statements to be examined by 22%–30%.

## Dataset

### Preprocessed Dataset

We published our processed dataset at 

Please download the dataset, unzip it and put all files in ```./processed``` folder.

### Use your own dataset

If you want to use your own dataset, please prepare the data as follow:

1. There are two types of data files including ```data_n.pt``` and ```index.npy```

2. Each ```data_n.pt``` include a ```Data``` object from ```torch_geometric```. Each ```Data``` represent a pair of method-level data and relevant statement-level data. Each bug may have multiple pairs to cover all methods in the method-level graph. With in each ```Data``` object, you should have:
	
	1> ```Data.x = [feature_1, ..., feature_6]```
	
	2> ```Data.y = [edge_list_s, label_m, label_s]```
	
	3> ```Data.edge_index = edge_list_m```
	
Where ```feature_1, feature_6``` are ```N*L*E, N'*L'*E``` sized torch tensors that represent the method sub-token embedding and the variables in the statement-level. ```N``` is the number of nodes in method-level, ```N'``` is the number of nodes in statement-level, ```L``` is the sub-token sequence length (padded), ```L'``` is the variable sequence length, and ```E``` is the sub-token embedding length.

```feature_2, feature_3, feature_5``` are lists that represent the method AST, most similar method AST, and the subtree of AST for the statement. And each of them contains the ```Tree``` object structure mentioned in ```main.py``` and the relevant value dictionary. The id in the dictionary matches the id in the ```Tree``` object.

```feature_4``` is a ```N'*T``` sized torch tensor that represent the code coverage information. ```T``` is the two times of the total number of test case that represent the coverage and the pass and fail condition.

```edge_list_s, edge_list_m``` are the matrixs to represent the graph edges for method-level and the statement-level. Please refer to ```torch_geometric``` package for more details.

```label_m, label_s``` are the true labels for the method-level and statement-level. Each of them are ```N*1``` and ```N'*1``` sized torch tensors. Value ```1``` means the node is buggy while value ```0``` represent non-buggy.
	
3. ```index.npy``` file contains a ```M*x``` numpy array that shows which data file belongs to which bug. ```M``` is the total number of bugs and ```x``` is the data file id. Each bug may have different ```x```.

## Requirement

Install ```Torch``` by following the instuction from [PyTorch](https://pytorch.org/get-started/locally).

Install ```torch_sparse``` by following the instuction from [pytorch_sparse](https://github.com/rusty1s/pytorch_sparse).

Install ```torch_geometric``` by following the instuction from [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).


## Instruction

Run ```main.py``` to see the result for our experiment. 
