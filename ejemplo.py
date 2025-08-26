import network
from mnist_loader import load_data_wrapper

training_data, validation_data, test_data = load_data_wrapper()
net = network.Network([784, 30, 10])  # arquitectura clásica del libro
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
