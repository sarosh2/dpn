import cupy as cp
import math

class Neuron:
    
    alpha = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    def __init__ (self, activation, input_size):
        self.activation = activation
        self.input_size = input_size
        self.output_size = 0
        self.bias = (float)(cp.random.uniform(0.0, 1.0))
        self.children = []
        self.parents = []
        self.output = 0.0
        self.inputs = None
        self.gradients = None

    def initialize_weights(self):
        
        if self.input_size != 0:
            #He initialization for weights
            std_dev = math.sqrt(2 / self.input_size)
            self.weights = cp.random.randn(1, self.input_size) * std_dev

            #parameters for adam optimizer updates
            self.m_w = cp.zeros_like(self.weights)  # First moment vector
            self.m_b = 0.0
            self.v_w = cp.zeros_like(self.weights)  # Second moment vector
            self.v_b = 0.0

    def add_parent(self, parent):
        self.parents.append(parent)
        self.input_size += 1

    def add_child(self, child):
        self.children.append(child)
        self.output_size += 1
        child.add_parent(self)

    def remove_parent(self, parent):
        if parent in self.parents:
            self.parents.remove(parent)
            self.input_size -= 1

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)
            self.output_size -= 1

    def forward_prop(self, input, order):

        if self.inputs is None:
            self.inputs = input
            
        else:
            #append the given input to the list of inputs
            #print(self.inputs.shape, input.shape)
            if order:
                self.inputs = cp.vstack([self.inputs, input])
            else:
                self.inputs = cp.vstack([input, self.inputs])
        
        #if the inputs are complete (all parents have provided an output) propagate the node to its children
        if self.inputs.shape[0] == self.input_size:
            
            #calculate output
            self.output = cp.dot(self.weights, cp.array(self.inputs)) + self.bias
            #APPLY ACTIVATION
            if self.activation == 'Relu':
                self.output = cp.maximum(0, self.output)


            self.temp_inputs = self.inputs
            self.inputs = None
            for child in self.children:
                child.forward_prop(self.output, True)

    def back_prop(self, gradient, t):

        if self.gradients is None:
            self.gradients = gradient
            
        else:
            self.gradients = cp.vstack([gradient, self.gradients]) #append the given input to the list of inputs

        if self.gradients.shape[0] == self.output_size:

            self.gradients = cp.sum(self.gradients, axis=0, keepdims=True)
   
            num_samples = self.gradients.shape[1]
            
            d_output = None
            
            if self.activation == 'Relu':
                d_output = self.gradients * (self.output > 0)

            else:
                d_output = self.gradients


            dW = cp.dot(d_output, self.temp_inputs.T) / num_samples
            db = cp.sum(d_output, axis=1, keepdims=True) / num_samples

            #print(self.weights)
            #print(len(self.parents))
            d_output = cp.dot(self.weights.T, d_output)

            self.m_w = Neuron.beta_1 * self.m_w + (1 - Neuron.beta_1) * dW
            self.m_b = Neuron.beta_1 * self.m_b + (1 - Neuron.beta_1) * db

            # Update biased second moment estimate for weights and biases
            self.v_w = Neuron.beta_2 * self.v_w + (1 - Neuron.beta_2) * cp.square(dW)
            self.v_b = Neuron.beta_2 * self.v_b + (1 - Neuron.beta_2) * cp.square(db)

            # Bias correction for first and second moment estimates
            m_w_hat = self.m_w / (1 - Neuron.beta_1**t)
            m_b_hat = self.m_b / (1 - Neuron.beta_1**t)
            v_w_hat = self.v_w / (1 - Neuron.beta_2**t)
            v_b_hat = self.v_b / (1 - Neuron.beta_2**t)

            # Update the weights and biases
            self.weights -= Neuron.alpha * m_w_hat / (cp.sqrt(v_w_hat) + Neuron.epsilon)
            self.bias -= Neuron.alpha * m_b_hat / (cp.sqrt(v_b_hat) + Neuron.epsilon)

            self.temp_inputs = None
            self.gradients = None

            for i in range(len(self.parents) - 1, -1, -1):
                self.parents[i].back_prop(d_output[-(len(self.parents) - i), :].reshape(1, num_samples), t)