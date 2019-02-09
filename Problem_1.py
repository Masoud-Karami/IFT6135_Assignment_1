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
        bias = 0.0
        for l in range(n_hidden):
            self.layers.append(Layer(bias,dims[l:l+2],init_mode))
        self.layers.append(OutLayer(bias,dims[-2::],init_mode))
    
    def forward(self,inputs):
        # forward propagation of the NN
        
        # Input verifications
        if np.ndim(inputs) == 2:
            if np.shape(inputs)[1] != self.dims[0]:
                raise BadInput('The number of inputs do not match the dimentions!')
        elif len(inputs) != self.dims[0]:
            raise BadInput('The number of inputs do not match the dimentions!')
        
        # propagate forward and activate each layer
        for layer in self.layers:
            outputs = layer.forward(inputs)
            inputs = layer.activation(outputs)
        return inputs
    
    def cross_entropy(self,prediction,target):
        # make sure the inputs are in valid range
        if np.any(prediction < 0) or np.any(target < 0):
            raise ValueError('Negative prediction or target values')
        np.clip(prediction,1e-12,1)
        try:
            out_neuron = prediction[np.arange(prediction.shape[0]),target]
        except:
            # for the case when there is only one sample
            out_neuron = prediction[target]
        return -(np.log(out_neuron)).mean()
	
    def backward(self,prediction,target,learning_rate): 
        # backward propagation
        gradient = prediction
        for l in self.layers[::-1]:
            gradient = l.backward(gradient,target,learning_rate)
            
    def train(self,training,validation,batch_size,learning_rate=0.001,epochs=10):
        # split up the training and validation sets
        training_set, target_set = training
        validation, validation_target = validation
        # split up the training set in batch size
        n_batch = int(len(training_set) / batch_size)
        
        for epoch in range(epochs):
            # validation set to see the progression
            pred = self.forward(validation)
            cross_entropy = self.cross_entropy(pred,validation_target)
            likelihood = (pred.argmax(axis=1) == validation_target).mean() # mean of predictions gotten right
            print('\nLikelihood : %1.3f' % likelihood)
            print('Validation CE : %1.3f' % cross_entropy)
            self.shuffle_set(training_set,target_set)
            for b in range(n_batch):
                tr_x = training_set[(b*batch_size):((b+1)*batch_size)]
                tr_y = target_set[(b*batch_size):((b+1)*batch_size)]
                prediction = self.forward(tr_x)
                self.backward(prediction,tr_y,learning_rate)
                
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
            filename = 'NN_%i_%i_%i_%ih%im%is' % (now.year,now.month,now.day,now.hour,now.minute,now.second)
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
        self.bias = np.ones([1,dims[1]])*bias
    
    def forward(self,inputs):
        self.inputs = []
        self.outputs = []
        self.inputs = inputs
        return inputs.dot(self.weights) + self.bias
    
    def activation(self,inputs):
        # activation function (sigmoid / ReLU / Maxout / linear / or whatever)
        # ReLu activation function :
        self.outputs = np.maximum(inputs,np.zeros(np.shape(inputs)))
        return self.outputs
    
    def backward(self,gradients,target,learning_rate):
        # gradient before activation :
        # h = ReLU(a) | ReLU'(x) = sgn(ReLU(x)) | ReLU'(a) = sgn(h)
        signs = np.sign(self.outputs)
        gradients = gradients*signs
        
        grad_w = self.inputs.T.dot(gradients)
        grad_b = gradients.sum(axis=0)
        self.update(grad_w,grad_b,learning_rate)
        return gradients.dot(self.weights.T)
    
    def update(self,grad_w,grad_b,learning_rate):
        self.weights -= learning_rate*grad_w
        self.bias -= learning_rate*grad_b
    
    def display(self, layer_num):
        print('\nLayer %i parameters :' % layer_num)
        print('Bias :')
        print(self.bias)
        print('Weights :')
        print(self.weights)

class OutLayer(Layer): #subclass of Layer
    def activation(self,inputs):
        # softmax activation function (slide #17 of Artificial Neuron presentation)
        # the sum of the output vector(s) will be = 1.
        # NEED to substract the largest number to avoid INF number (causes problems in the algorithm)
        a = np.exp(inputs - inputs.max(axis=1, keepdims=True))
        try:
            size = np.shape(inputs)[1]
            outputs = a/a.dot(np.ones([size,size]))
        except:
            outputs = a/a.sum()
        return outputs
    
    def backward(self,gradients,target,learning_rate):
        # Output gradient :
        e = np.zeros_like(gradients)
        e[np.arange(e.shape[0]),target] = 1
        gradients = -(e - gradients)
        # Parameters gradient :
        grad_w = self.inputs.T.dot(gradients)
        grad_b = gradients.sum(axis=0)
        self.update(grad_w,grad_b,learning_rate)
        return gradients.dot(self.weights.T)

def import_MNIST():
    with gzip.open('./data/mnist.pkl.gz', 'rb') as f:
        tr,va,te = pickle.load(f, encoding='latin-1')
    return (tr,va,te)
      
def main():
    # testing dataset :
    #dataset = np.array([[0,0],[0,1],[1,0],[1,1]])
    #y = np.array([0,1,1,0])
    
    # import MNIST dataset :
    tr,va,te = import_MNIST()
    
    classifier = NN((784,550,300,10), 2, 'GLOROT')
    #classifier.save()
    #classifier = NN((1,4,1,1),2,'GLOROT','train',None,'NN_2019_1_31_13h10m16s')
    
    display_weights = False
    classifier.display(display_weights)
	
    classifier.train(tr,va,100,0.0003,10)
    
    classifier.save()

if __name__ == '__main__':
    main()
