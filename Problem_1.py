# Authors:
# p1220459 Mathieu Champagne
# pXXXXXXX Masoud Karami
# pXXXXXXX Narges Salehi

# UdeM, IFT6135, Assignment 1, H2019

import datetime
import numpy as np
import pickle
import gzip

class BadInit(Exception):
    """Something in the initialization is wrong"""
    pass
class BadInput(Exception):
    """Something is wrong with the inputs"""
    pass

class NN(object):
    
    def __init__(self,dims=(784,1024,2048,10),n_hidden=2,init_mode='GLOROT',mode='train',datapath=None,model_path=None):
        
        if model_path is None:
            if n_hidden+2 != len(dims):
                raise BadInit('The dimentions and number of hidden layers do not add up!')
            self.dims = dims
            self.n_hidden = n_hidden
            self.layers = []
            self.initialize_weights(n_hidden,dims,init_mode)
        else:
            self.load(model_path)
    
    def initialize_weights(self,n_hidden,dims,init_mode):
	# either ZERO init / NORMAL DISTRIBUTION init / GLOROT init
        for l in range(n_hidden+1):
            bias = 0.0
            self.layers.append(Layer(bias,dims[l:l+2],init_mode))
    
    def forward(self,input):
        # forward propagation of the NN
        
        # Input verifications
        if np.ndim(input) == 2:
            if np.shape(input)[1] != self.dims[0]:
                raise BadInput('The number of inputs do not match the dimentions!')
        else:
            if len(input) != self.dims[0]:
                raise BadInput('The number of inputs do not match the dimentions!')
        
        # propagate forward until output layer
        for l in range(self.n_hidden+1):
            output = self.layers[l].forward(input)
            if l < self.n_hidden:
                input = self.activation(output)
            else:
                return self.softmax(output)
        
    def activation(self,input):
    # activation function (sigmoid / ReLU / Maxout / linear / or whatever)
	# input : vector with results of pre-activation of one layer
	# output : vector with results of activation of the layer 
        output = np.maximum(input,np.zeros(np.shape(input)))
        return output
    
    def cross_entropy(self,prediction,target):
        # make sure the inputs are in valid range
        if np.any(prediction < 0) or np.any(target < 0):
            raise ValueError('Negative prediction or target values')
        np.clip(prediction,1e-12,1)
        try:
            out_neuron = prediction[np.arange(prediction.shape[0]),target]
        except:
            # for when there is only one sample
            out_neuron = prediction[target]
        return -(np.log(out_neuron)).mean()
    
    def softmax(self,input):
	# softmax activation function (slide #17 of Artificial Neuron presentation)
	# the sum of the output vector will be = 1.
        a = np.exp(input)
        try:
            return a/a.dot(np.ones([np.shape(input)[1], np.shape(input)[1]]))
        except:
            return a/a.sum()
	
    def backward(self,cache,labels): #...
	# backward propagation
        print("")
    
    def update(self,grads): #...
	# Upgrade the weights after backward propagation
        print("")
	
    def train(self,training_set,target_set,batch_size,epochs=10):
        # split up the training set in batch size
        n_batch = int(len(training_set) / batch_size)
        
        for epoch in epochs:
            for b in range(n_batch):
                tr_x = training_set[(b*batch_size):((b+1)*batch_size)]
                tr_y = target_set[(b*batch_size):((b+1)*batch_size)]
                prediction = self.forward(tr_x)
                # delta ?
                
    def shuffle_set(self,training_set,target_set):
        seed = np.random.randint(2**31)
        np.random.seed(seed)
        np.random.shuffle(training_set)
        np.random.seed(seed)
        np.random.shuffle(target_set)
        np.random.seed()
        
    def test(self,epoch=10):
	# test the non-training dataset and output the accuracy rate...
    	print("")
        
    def save(self,filename=None):
        # saves the weights and structure of the current NN
        path = "./Saved_models/"
        if filename is None:
            now = datetime.datetime.now()
            filename = 'NN_' + str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + 'h' + str(now.minute) + 'm' + str(now.second) + 's'
        
        with open(path+filename, 'wb') as f:
            pickle.dump([self.dims, self.n_hidden, self.layers], f)
            
    def load(self, filename):
        # load the weights and structure of the saved NN
        path = "./Saved_models/"
        with open(path+filename, 'rb') as f:
            self.dims, self.n_hidden, self.layers = pickle.load(f)
    
    def display(self, display_weights=False):
        print("")
        strings = ["#inputs"]
        for i in range(1,self.n_hidden+1):
            strings.append("hid.layer." + str(i))
        strings.append("#outputs")
        strings.append("#param.")
        out_str = []
        for i in self.dims:
            out_str.append(i)
        num_param = 0
        for i in range(len(self.dims)-1):
            num_param += self.dims[i]*self.dims[i+1] + 1
        out_str.append(num_param)
        for i in range(len(out_str)):
            print('%-13s%-12i' % (strings[i], out_str[i]))
        
         #to visualize the weight of each neurons
        if display_weights:
            for l in range(self.n_hidden+1):
                self.layers[l].display(l+1)
        
class Layer:
    def __init__(self, bias, dims, init_mode):
        d = np.sqrt(6/(dims[0]+dims[1])) #uniform limits for GLOROT init
        if init_mode.upper() == 'GLOROT':
            self.weights = np.random.uniform(-d,d,[dims[0],dims[1]])
        elif init_mode.upper() == 'NORMAL':
            self.weights = np.random.randn(dims[0],dims[1]) #here instead of random, put the normal random function or whatever else
        elif init_mode.upper() == 'ZERO':
            self.weights = np.zeros([dims[0],dims[1]])
        else:
            raise BadInit('Initializing function not valid, choose between GLOROT, NORMAL and ZERO')
        self.bias = bias
    
    def forward(self,inputs):
        return inputs.dot(self.weights) + self.bias
    
    def display(self, layer_num):
        print('\nLayer %i parameters :' % layer_num)
        print('Bias : %1.2f' % self.weights[0][0])
        print('Weights :')
        print(self.weights[1::][::])

def import_MNIST():
    with gzip.open('./data/mnist.pkl.gz', 'rb') as f:
        tr,va,te = pickle.load(f, encoding='latin-1')
    tr_x, tr_y = tr
    va_x, va_y = va
    te_x, te_y = te
    return (tr_x,tr_y,va_x,va_y,te_x,te_y)

def format_target_set(target,):
    out = np.zeros(np.shape(dataset))
      
def main():
    # testing dataset :
    dataset = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])
    
    # import MNIST dataset :
    #tr_x,tr_y,va_x,va_y,te_x,te_y = import_MNIST()
    
    classifier = NN((2,4,2), 1, 'GLOROT')
    #classifier.save()
    #classifier = NN((1,4,1,1),2,'GLOROT','train',None,'NN_2019_1_31_13h10m16s')
    
    display_weights = False
    classifier.display(display_weights)
	
    out = classifier.forward(dataset)
    cross_entropy_loss = classifier.cross_entropy(out,y)
    
    cool = [1,2,3,4,5,6]
    cool2 = [1,2,3,4,5,6]
    classifier.shuffle_set(dataset,y)
    
    print('\nOutputs :')
    for o in out:
        print(o)
	
    print('\nCross-entropy : %1.3f' % cross_entropy_loss)

if __name__ == '__main__':
    main()
