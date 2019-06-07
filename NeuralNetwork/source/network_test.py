import neural_network as nk
import numpy
import matplotlib.pyplot as pl
import scipy.special
import scipy.ndimage.interpolation
import json
import time
import progressbar

# 测试神经网络
if __name__ == "__main__":
	input_nodes = nk.input_nodes
	hidden_nodes = nk.hidden_nodes
	hidden_nodes_2 = nk.hidden_nodes_2
	#hidden_nodes_3 = nk.hidden_nodes_3
	output_nodes = nk.output_nodes
	learningrate = nk.learningrate
	Network = nk.neuralNetwork(input_nodes,hidden_nodes,hidden_nodes_2,output_nodes,learningrate)
	nk.test(Network,"mnist_test.csv")
	

# hidden_nodes = 200 lr = 0.01 performance = 97.34