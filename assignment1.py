class NN(object):
    
    def __init__(self,hidden_dims=(1024,2048),n_hidden=2,mode='train',datapath=None,model_path=None):
	
	
	
	def initialize_weights(self,n_hidden,dims):
	# either ZERO init / NORMAL DISTRIBUTION init / GLOROT init
	
	
	def forward(self,input,labels,..):
	# forward propagation of the NN (use activation functions)
	
	
	def activation(self,input):
	# activation function (sigmoid / softmax / or whatever)
	
	
	def loss(self,prediction,..):
	
	
	def softmax(self,input,..):
	# softmax activation function
	
	
	def backward(self,cache,labels,...):
	# backward propagation
	
	
	def update(self,grads,..):
	# Upgrade the weights after backward propagation
	
	
	def train(self):
	# RUN	- initialization
	# THEN DO
	#		- forward
	#		- loss
	#		- backward
	#		- update
	# UNTIL satisfied with the training / loss
	
	
	def test(self):
	# test the non-training dataset and output the accuracy rate...
	
	
	
	
	