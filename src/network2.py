
#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        # Ableitung dC/dz -> Error
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        # Cross Entropy in python mit numpy
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        # Ableitung dC/dz
        return (a-y)


#### Main Network class
class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        # Gaussche Verteilung
        # Mittelwert: 0
        # Abweichung: 1/{Anzahl der Neuronen die mit dem neuen Neuron verbunden sind}^.5
        # Kein Problem, da -> Werte durchschnittlich näher bei 0 -> Neuronen saturieren nicht
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        # Auch Gaussche Verteilung
        # Mittelwert: 0
        # Abweichung: 1
        # Problem -> Können extrem große und kleine Werte sein -> grenzen der Wertemenge -> Neuronen saturieren 
        
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        # Output von Netzwerk mit Werten für die Input Schicht a
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            no_improvement_in = 10,
            no_improvement_average = True, # Controls whether the average validation of the last n epochs should be used for early stopping or (False) the max value
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):  
        # Trainingsdaten werden als Liste von Tuplen als parameter übergeben
        # ->[(x, y), (x1, y1), .... (xn, yn)]
        # Stochastic gradient descent -> also wird von backpropagation nicht nur Gradient eines Trainingsbeispiels dC/dw; dC/db berechnet sondern über mehrere Trainingsbeispiele der Durchschnitt.
        # Hauptfunktion: mini_batches formen -> damit dann backpropagation
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            
            
            print(f"Epoch {j} training complete")
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))
            
            if len(evaluation_accuracy) - 1 > no_improvement_in:
                if no_improvement_average:
                    evaluation_length = len(evaluation_accuracy) - 1 # '-1' because the current epoch should be excluded
                    recent_accuracies = evaluation_accuracy[evaluation_length - no_improvement_in: evaluation_length] 
                    average_validation_accuracy = sum(recent_accuracies) / len(recent_accuracies)
                    if evaluation_accuracy[-1] > average_validation_accuracy:
                        print(f"Current validation accuracy ({evaluation_accuracy[-1]}) is greater than average in the last {no_improvement_in} epochs ({average_validation_accuracy}).")
                    else:
                        print(f"Stopping Early: Current validation accuracy ({evaluation_accuracy[-1]}) < than average last {no_improvement_in} epochs ({average_validation_accuracy}).")
                        return
                else:
                    max_classifying_accuracy = max(evaluation_accuracy)
                    if max_classifying_accuracy in evaluation_accuracy[len(evaluation_accuracy) - no_improvement_in : len(evaluation_accuracy)]:
                        print(f"Evaluation data improved in the last {no_improvement_in} epochs.")
                    else:
                        print(f"Stopping Early: No improvement in classification accuracy in the last {no_improvement_in} epochs.")
                        return
            
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        # In die entgegengesetzte Richtung des Gradienten gehen
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
        #                 for w, nw in zip(self.weights, nabla_w)]
        self.weights = [w * (1 - (eta * lmbda) / n) - eta * nw 
                        for w, nw in zip(self.weights, nabla_w)] # L1 Regularisierung -> L2 auskommentiert
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # Ich gebe Tuple nabla_b, nabla_w zurück
        # nabla_b -> dC/db für alle b -> Gradient des Bias
        # nabla_w -> dC/dw für alle w -> Gradient der Weights
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # Error und dC/db und dC/dw
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        
        # Hier gehe ich von Schicht zu Schicht zurück
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        # Gibt die genauigkeit des Netzwerkes in Anzahl an richtigen Ausgaben an
        # Nimmt als Messung für genauigkeit data zum validieren
        
        # 'np.argmax(y) -> weil immer Neuron mit höchstem Wert als output Nummer gewertet wird.
        
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, lmbda, convert=False):
        # Insgesamte Kosten für trainingsDaten oder wenn convert=True für validation- und test-Daten 
        
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        # cost += 0.5*(lmbda/len(data))*sum(
        #     np.linalg.norm(w)**2 for w in self.weights) # L2 Regularisierung
            cost += (lmbda/len(data))*sum(
                abs(w) for w in self.weights
            ) # L1 Regularisierung
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

# Helffunktionen

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
