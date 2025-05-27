from neuron import Neuron
import networkx as nx
import matplotlib.pyplot as plt
import cupy as cp
import time


class SPN:

    def __init__ (self):
        self.input_nodes = []
        self.hidden_nodes = []
        self.output_nodes = []

        self.graph = nx.DiGraph()
        self.max_id = 1
        self.vertices = {}
        self.active_node = 0

    def create_node(self, activation, input_size, status):
        neuron = Neuron(activation, input_size)
        
        #add to vertices dict and activate node
        self.vertices[self.max_id] = neuron
        self.active_node = self.max_id
    
        self.graph.add_node(self.active_node)

        if status == 'input':
            self.input_nodes.append(self.active_node)
        elif status == 'output':
            self.output_nodes.append(self.active_node)
            self.vertices[self.active_node].output_size = 1
        else:
            self.hidden_nodes.append(self.active_node)    

        self.max_id += 1

        return self.active_node

    
    def add_connection(self, parent, child):
        self.graph.add_edge(parent, child)
        self.vertices[parent].add_child(self.vertices[child])

    def visualize(self):
        pos = {}

        # Assign positions to input nodes
        for i, node in enumerate(self.input_nodes):
            pos[node] = (1, i)  # Y=3 for input nodes

        # Assign positions to hidden nodes
        for i, node in enumerate(self.hidden_nodes):
            pos[node] = (2, i)  # Y=2 for hidden nodes

        # Assign positions to output nodes
        for i, node in enumerate(self.output_nodes):
            pos[node] = (3, i)  # Y=1 for output nodes

        # Draw the nodes with different colors for input, hidden, and output layers
        nx.draw_networkx_nodes(self.graph, pos, nodelist=self.input_nodes, node_color='lightblue', node_size=100, label='Input Layer')
        nx.draw_networkx_nodes(self.graph, pos, nodelist=self.hidden_nodes, node_color='lightgreen', node_size=100, label='Hidden Layer')
        nx.draw_networkx_nodes(self.graph, pos, nodelist=self.output_nodes, node_color='lightcoral', node_size=100, label='Output Layer')

        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, edgelist= self.graph.edges, width=2)

        # Draw labels for the nodes
        nx.draw_networkx_labels(self.graph, pos, font_size=8)

        # Title and display options
        plt.axis('off')  # Hide the axes
        plt.show()

    def compile(self):
        for vertex in self.vertices.values():
            vertex.initialize_weights()
            #print(vertex.input_size)
            #print(vertex.weights.shape)

    def categorical_crossentropy(output, true_labels):
        epsilon = 1e-12
        Y_pred = cp.clip(output, epsilon, 1. - epsilon)

        # Step 2: Compute the log of predicted probabilities
        log_Y_pred = cp.log(Y_pred)

        # Step 3: Compute element-wise multiplication between Y_true and log_Y_pred
        return -cp.sum(true_labels * log_Y_pred, axis=0)

    def caclulate_accuracy(output, true_labels):
        predicted_classes = cp.argmax(output, axis=0)

        # Step 2: Convert one-hot encoded true labels to class indices (index of 1 for each column)
        true_classes = cp.argmax(true_labels, axis=0)

        # Step 3: Compare predicted classes with true classes
        correct_predictions = cp.sum(predicted_classes == true_classes)

        # Step 4: Compute accuracy
        return correct_predictions / true_labels.shape[1]
    
    def softmax(x):
        e_x = cp.exp(x - cp.max(x, axis=0, keepdims=True))  # Subtract max for numerical stability
        return e_x / cp.sum(e_x, axis=0, keepdims=True)
    
    def get_batches(X, Y, batch_size):
        num_samples = X.shape[1]
        """
        Generator that yields batches from input matrix X and labels Y.
        X: Input data of shape (n_features, n_samples)
        Y: Labels of shape (n_classes, n_samples)
        batch_size: Number of samples per batch
        """
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            yield X[:, start_idx:end_idx], Y[:, start_idx:end_idx]  # Yield batches of X and Y
    
    def train(self, X, Y, t):
    
        start = time.time()

        for node in self.input_nodes:
            self.vertices[node].forward_prop(cp.array(X), False)

        output = self.vertices[self.output_nodes[0]].output
        for j in range(1, 10):
            output = cp.vstack([output, self.vertices[self.output_nodes[j]].output])

        output = SPN.softmax(output)
        gradient = output - Y

        #back prop
        for i in range(len(self.output_nodes) - 1, -1, -1):
            self.vertices[self.output_nodes[i]].back_prop(cp.array(gradient[i, :].reshape(1, Y.shape[1])), t)

        end = time.time()
        
        #calculate loss
        loss = SPN.categorical_crossentropy(output, Y)
        #print("Loss: ", cp.mean(loss))

        #calculate accuracy
        accuracy = SPN.caclulate_accuracy(output, Y)
        #print("Accuracy: ", cp.mean(accuracy))

        return cp.hstack((end - start, cp.mean(loss), cp.mean(accuracy)))
    
    def execute(self, epochs, batch_size, x_train, y_train, x_val, y_val, x_test, y_test):

        t = 1  # Timestep

        for i in range(epochs):

            metrics = []
            for batch_num, (batch_X, batch_Y) in enumerate(SPN.get_batches(x_train, y_train, batch_size)):
                metrics.append(self.train(batch_X, batch_Y, t))
                t += 1

            metrics = cp.vstack(metrics)
            
            #validate output
            for node in self.input_nodes:
                self.vertices[node].forward_prop(cp.array(x_val), False)

            output = self.vertices[self.output_nodes[0]].output

            for j in range(1, 10):
                output = cp.vstack([output, self.vertices[self.output_nodes[j]].output])
            val_loss = cp.mean(SPN.categorical_crossentropy(output, y_val))
            val_accuracy = cp.mean(SPN.caclulate_accuracy(output, y_val))

            print(f"Epoch: {i + 1} Total Time: {cp.sum(metrics[:, 0]):.4f} Average Time per batch: {cp.mean(metrics[:, 0]):.4f} Train Accuracy: {metrics[-1, 2]:.4f} Train Loss: {metrics[-1, 1]:.4f} Val Accuracy: {val_accuracy:.4f} Val Loss: {val_loss:.4f}")


        for node in self.input_nodes:
            self.vertices[node].forward_prop(cp.array(x_test), False)

        output = self.vertices[self.output_nodes[0]].output
        for j in range(1, 10):
            output = cp.vstack([output, self.vertices[self.output_nodes[j]].output])

        #print(output.shape, x_test_flat.shape)
        test_loss = cp.mean(SPN.categorical_crossentropy(output, y_test))
        test_accuracy = cp.mean(SPN.caclulate_accuracy(output, y_test))

        print("Test Accuracy: ", test_accuracy, "Test Loss: ", test_loss)