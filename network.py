# %load network.py

"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

#Programación orientada a objetos
#Representamos la ANN como un objeto
#Definimos las instrucciones de como inicializar la red
class Network(object):

    def __init__(self, sizes):  #Definimos una función
                                #_init_ es el constructo
                                #self se refiere al propio objeto
                                #El constructo tiene como argumento
                                #una lista del num de neuronas
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes) #Número de capas de la red
        self.sizes = sizes           
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #Crea una matriz de tamaño (y,1) con valores aleatorios
        #[1:] desde la segunda capa hasta la última
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        #Creamos los pesos que conectan capa a capa. zip empareja el
        #tamaño de cada capa con el de la siguiente.
        #Crea una matriz de pesos para conectar x neuronas de la capa
        #anterior con y neuronas de la capa actual

    def feedforward(self, a): #función de activación
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights): 
        #Para b y w en la lista de bias y de pesos...
            a = sigmoid(np.dot(w, a)+b) #Calcular activación
        return a

    #Función que divide el conjunto de datos en los batches
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data) #Lista del train_data
        n = len(training_data)              #Número de datos train

        if test_data:                       #En caso de existir test
            test_data = list(test_data)     #Lista del test_data
            n_test = len(test_data)         #Número de datos test

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
            #El elemento que inicia en k sumado el tamaño mini_batch
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
            #Para cada mini_batch se debe actualizar el peso
                self.update_mini_batch(mini_batch, eta)
                #Se llama a la función que actualiza el peso
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
                #Imprime epochs
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #Inicializar ambos en 0
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #Se evalua la suma sobre x de C = SUM C_x
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        #Actualizar pesos, da un paso de tamaño proporcional al eta
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        #Actualizar bias, el signo menos indica la dirección negativa
        #del crecimiento (- grad)

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #Inicialización en 0
        # feedforward
        activation = x #x es el argumento
        activations = [x] # list to store all the activations, layer by layer
        #Lista para guardar todas las activaciones, es un vector
        zs = [] # list to store all the z vectors, layer by layer
        #Lista para guardar todos los vectores z, es una lista de vectores
        for b, w in zip(self.biases, self.weights): #Ciclo sobre cada capa
            z = np.dot(w, activation)+b
            #Mismo código del principio al inicializar la red
            zs.append(z)  #Guardar el valor en una lista
            activation = sigmoid(z) #Calcular activación
            activations.append(activation) #Almacenar activaciones
        # backward pass
        delta = cross_entropy_delta(activations[-1], y)
        #Originalmente delta incluia la derivada de la función de costo. 
        #Ahora aplicamos la derivada de la función cross-entropy.
        nabla_b[-1] = delta
        #Las derivadas respecto a las betas son las mismas deltas
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        #Notar que la variable l en el loop es usada de manera distinta
        #al libro (cap 2). Aquí l=1 significa la ultima capa de neuronas,
        #l=2 la segunda capa, y así sucesivamente. Es un renombramiento
        #de la convención, usada para tomar ventaja de que phyton puede
        #usar indices negativos en las listas.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        #para un verdadero devuelve un 1, para un falso un 0

    def cost_derivative(self, output_activations, y): #derivada de C
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)       #par(C_x)/par(x) = (x-y)

#### Miscellaneous functions
#Definir función sigmoide y derivada
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

#Definimos la función cross-entropy
def cross_entropy_delta(a, y):
    #Con sigmoid en salida, C-E hace que dC/dz = a - y
    return a - y

