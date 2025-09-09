import network
from mnist_loader import load_data_wrapper

training_data, validation_data, test_data = load_data_wrapper()
net = network.Network([784, 30, 10])  # arquitectura clásica del libro
net.SGD(training_data, 30, 10, 0.1, test_data=test_data, mu=0.4) 
#lr y mu optimizados

