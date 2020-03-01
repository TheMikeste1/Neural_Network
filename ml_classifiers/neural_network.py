import numpy as np
import matplotlib.pyplot as plt
from ml_classifiers.general import Classifier, FixedSizeList


class Node:
    MODE_BINARY = 0
    MODE_SIGMOID = 1
    MODE_SIGMOID_TANH = 2
    MODE_SOFT_MAX = 3

    def __init__(self, number_inputs, learning_rate, weights=None, mode=MODE_SIGMOID):
        self.learning_rate = learning_rate

        if weights is None:
            # Section 3.3.4 (p. 48) of Machine Leaning: An Algorithmic Perspective:
            # We should initialize weights to small random numbers
            # OR Section 4.2.2 (p. 80):
            # Set them to -1 / sqrt(num_inputs) < weight < 1 / sqrt(num_inputs)
            weights = np.random.uniform(-1 / np.math.sqrt(number_inputs),
                                        1 / np.math.sqrt(number_inputs),
                                        size=number_inputs)
        self.weights = weights
        self.mode = mode

        self.output = None
        self.inputs = None
        self.error = None
    # def __init__

    def add_weight(self, weight):
        self.weights.append(weight)
    # def add_weight

    def change_weight(self, index, new_weight):
        if index > len(self.weights):
            raise IndexError
        self.weights[index] = new_weight
    # def change_weight

    @staticmethod
    def __sigmoid_activation(x, beta=1):
        return 1 / (1 + np.exp(-beta * x))

    @staticmethod
    def __sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def __tanh_sigmoid_activation(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @staticmethod
    def __soft_max_activation(inputs):
        # This might not be correct
        # Section 4.2.3 (p. 81)
        # http://www.adeveloperdiary.com/data-science/deep-learning/neural-network-with-softmax-in-python/
        # According to the above site, I am subtracting by inputs' max
        if type(inputs) is not np.ndarray:
            inputs = np.array(inputs)
        ez = np.exp(inputs - np.max(inputs))
        return ez / ez.sum()

    def __update_activation(self, total, inputs):
        if self.mode == self.MODE_SIGMOID:
            return self.__sigmoid_activation(total)
        elif self.mode == self.MODE_SIGMOID_TANH:
            return self.__tanh_sigmoid_activation(total)
        elif self.mode == self.MODE_SOFT_MAX:
            return self.__soft_max_activation(inputs)
        return 1.0 if total > 0 else 0.0

    def apply_inputs(self, inputs):
        if len(inputs) != len(self.weights):
            raise ArithmeticError(f"Number of inputs and number of weights must match! Received {len(inputs)} inputs,"
                                  f"expected {len(self.weights)}")
        self.inputs = inputs
        total = sum(self.weights * inputs)
        self.output = self.__update_activation(total, inputs)
        return self.output
    # def apply_inputs

    def calculate_error(self, exp_output, sum_k=None):
        if self.MODE_SIGMOID:
            self.error = self.__sigmoid_derivative(self.output) * \
                         (sum_k if sum_k is not None else (self.output - exp_output))
        else:
            raise NotImplementedError("Currently only MODE_SIGMOID is implemented")
        return self.error

    def update_weights(self, exp_output, sum_k):
        pre_weights = self.weights
        for i in range(len(self.weights)):
            change = self.learning_rate * self.calculate_error(exp_output, sum_k) * self.inputs[i]
            # Added random value to add momentum
            self.weights[i] -= change + np.random.uniform(-change / 10, change / 10)
        return pre_weights
# class Node


class NeuralLayer:
    def __init__(self, num_inputs, num_nodes, learning_rate, bias=-1):
        self.num_inputs = num_inputs
        self.num_nodes = num_nodes
        self.bias = bias
        self.nodes = np.empty(0)
        for i in range(num_nodes):
            self.nodes = np.append(self.nodes, [Node(self.get_total_inputs(), learning_rate)])

    def get_total_inputs(self):
        if self.bias is not None:
            return self.num_inputs + 1
        return self.num_inputs

    def get_weights(self):
        weights = []
        for node in self.nodes:
            weights.append(node.weights)
        return np.array(weights)

    def get_errors(self):
        errors = []
        for node in self.nodes:
            errors.append(node.error)
        return np.array(errors)

    def apply_inputs(self, inputs):
        if len(inputs) != self.num_inputs:
            raise ArithmeticError(f"Number of inputs is incorrect! Received {len(inputs)} inputs,"
                                  f"expected {self.num_inputs}")
        biased_inputs = inputs.copy()
        if self.bias is not None:
            biased_inputs = np.append(biased_inputs, self.bias)
        outputs = []
        for node in self.nodes:
            outputs.append(node.apply_inputs(biased_inputs))
        return np.array(outputs)

    def update_weights(self, exp_outputs=None, sum_k=None):
        if exp_outputs is None:
            exp_outputs = [None] * len(self.nodes)
        elif len(self.nodes) is not len(exp_outputs):
            raise AssertionError("Number of nodes should be equal to expected number of outputs")
        if sum_k is None:
            sum_k = [None] * len(self.nodes)
        elif len(sum_k) is not len(self.nodes):
            raise AssertionError("Number out nodes should be equal to length of sum_k")
        pre_weights = []
        for i in range(len(self.nodes)):
            pre_weights.append(self.nodes[i].update_weights(exp_outputs[i], sum_k[i]))
        return np.array(pre_weights).copy()


class NeuralBrain:
    def __init__(self, num_inputs, nodes_per_layer, learning_rate=np.random.uniform(0.1, 0.4)):
        if type(nodes_per_layer) is not np.ndarray and type(nodes_per_layer) is not list:
            raise TypeError(f"nodes_per_layer must be type numpy.ndarray or list. Got {type(nodes_per_layer)}")

        self.num_inputs = num_inputs
        self.num_layers = len(nodes_per_layer)
        self.learning_rate = learning_rate
        if self.num_layers <= 0:
            print("You want a brain with less than 1 layer? Meaning you want an empty brain? Well here you go! You \n"
                  "now have exactly what you had before: nothing. The only difference is that there is now an empty\n"
                  "shell. A shell that could have been so much. A shell that could have done great things. A shell \n"
                  "that could have changed the world! But here we are. With a brain. Filled with nothingness.")
            self.neural_layers = "Nothing"
            return

        if len(nodes_per_layer) != self.num_layers:
            raise AssertionError("nodes_per_layer must account for each layer, but len(nodes_per_layer) does not equal "
                                 "num_layers!")

        self.neural_layers = np.array([NeuralLayer(num_inputs, nodes_per_layer[0], self.learning_rate)])

        for i in range(1, self.num_layers):
            self.neural_layers = np.append(self.neural_layers, [NeuralLayer(nodes_per_layer[i - 1], nodes_per_layer[i],
                                                                            self.learning_rate)])

    def apply_inputs(self, inputs):
        if len(inputs) != self.num_inputs:
            raise ArithmeticError(f"Number of inputs is incorrect! Received {len(inputs)} inputs,"
                                  f"expected {self.num_inputs}")
        if type(self.neural_layers) is str:
            return inputs

        outputs = inputs
        for layer in self.neural_layers:
            outputs = layer.apply_inputs(outputs)

        return outputs

    def get_num_outputs(self):
        return self.neural_layers[-1].num_nodes

    def update_weights(self, target):
        weights = self.neural_layers[-1].update_weights(target)
        total_error = self.neural_layers[-1].get_errors()

        # Get the total sum_k for the layer k
        sum_k = []
        for weight_i in range(len(weights[0]) - 1):
            sum_k.append(sum(weights[:, weight_i] * total_error))

        for layer_i in range(len(self.neural_layers) - 2, 0, -1):
            weights = self.neural_layers[layer_i].update_weights(None, sum_k)
            error = self.neural_layers[layer_i].get_errors()
            sum_k = []
            for weight_i in range(len(weights[0]) - 1):
                sum_k.append(sum(weights[:, weight_i] * error))
        return total_error

# class NeuralBrain


class NeuralNetworkClassifier(Classifier):
    MODE_WEIGHT_UPDATE_SEQUENTIAL = 0
    MODE_WEIGHT_UPDATE_BATCH = 1

    MODE_CLASSIFIER_CLASSIFICATION = 0
    MODE_CLASSIFIER_REGRESSION = 1

    def __init__(self, nodes_per_layer, weight_update_mode=MODE_WEIGHT_UPDATE_SEQUENTIAL, learning_rate=None,
                 verbose=True, desired_accuracy=0.9, max_iterations=5000,
                 classifier_mode=MODE_CLASSIFIER_CLASSIFICATION, enable_graph=False, epochs_per_point=1):
        if type(nodes_per_layer) is not np.ndarray and type(nodes_per_layer) is not list:
            raise TypeError(f"nodes_per_layer must be type numpy.ndarray or list. Got {type(nodes_per_layer)}")
        self.nodes_per_layer = nodes_per_layer
        self.weight_update_mode = weight_update_mode
        self.learning_rate = learning_rate
        self.verbose = verbose
        if desired_accuracy > 1:
            desired_accuracy /= 100
        self.desired_accuracy = desired_accuracy
        self.max_iterations = max_iterations
        if classifier_mode is not self.MODE_CLASSIFIER_CLASSIFICATION:
            raise NotImplementedError("Only MODE_CLASSIFIER_CLASSIFICATION has been implemented")
        self.classifier_mode = classifier_mode
        self.enable_graph = enable_graph
        self.epochs_per_point = epochs_per_point
        self.graph_points = np.empty((0, 2))

        self.accuracy = 0
        self._brain = None
        self._targets = None

    def fit(self, dataset, targets):
        if self.classifier_mode is self.MODE_CLASSIFIER_CLASSIFICATION:
            if type(targets) is np.ndarray:
                self._targets = np.unique(targets).tolist()
            else:
                self._targets = []
                for t in targets:
                    if t not in self._targets:
                        self._targets.append(t)
        if self.learning_rate is not None:
            self._brain = NeuralBrain(len(dataset[0]), self.nodes_per_layer, learning_rate=self.learning_rate)
        else:
            self._brain = NeuralBrain(len(dataset[0]), self.nodes_per_layer)
            self.learning_rate = self._brain.learning_rate

        done = False
        iteration = 0
        last_ten_acc = FixedSizeList(10, 0.0)
        while not done:
            correct = 0

            for data_i in range(len(dataset)):
                data = dataset[data_i]
                target = targets[data_i]

                # Get the target output
                target = self._targets.index(target)
                # We need to translate target into one-hot encoding
                target = ([0] * target)
                target.append(1)
                target.extend([0] * (len(self._targets) - len(target)))
                target = np.array(target)

                # Apply inputs
                prediction = self.find_closest_category(self._brain.apply_inputs(data))
                if prediction == targets[data_i]:
                    correct += 1

                # Update weights
                self._brain.update_weights(target)

            accuracy = correct / len(dataset)

            if self.enable_graph and iteration % self.epochs_per_point == 0:
                point = np.array([iteration, accuracy * 100])
                self.graph_points = np.vstack((self.graph_points, point))

            unique = None
            if iteration % 100 == 0 and self.verbose:
                last_ten_acc.append(accuracy)
                print(f"Epoch {iteration}: {accuracy * 100:.2f}%")

            # This is to avoid loops of accuracy we can't get out of
            if min(last_ten_acc.data) != 0:
                unique = np.unique(last_ten_acc.data)

            if (self.desired_accuracy > 0 and accuracy - self.desired_accuracy >= 0) \
                    or (self.max_iterations > 0 and iteration >= self.max_iterations) \
                    or (unique is not None and len(unique) <= 2):
                print(f"Epoch {iteration}: {accuracy * 100:.2f}%")
                self.accuracy = accuracy
                done = True
                if self.graph_points[-1, 0] != iteration:
                    point = np.array([iteration, accuracy * 100])
                    self.graph_points = np.vstack((self.graph_points, point))
            iteration += 1

    def predict(self, dataset):
        if self._brain is None:
            raise AttributeError("Neural Brain has not been initialized! Please fit your data first.")
        if self._targets is None:
            raise AttributeError("Targets have not been initialized! Please fit your data first.")
        if type(dataset) is np.ndarray:
            dataset = dataset.copy().tolist()

        predictions = []
        for data in dataset:
            predictions.append(self._brain.apply_inputs(data))

        if self.classifier_mode is self.MODE_CLASSIFIER_CLASSIFICATION:
            for i in range(len(predictions)):
                predictions[i] = self.find_closest_category(predictions[i])

        return np.array(predictions)

    def find_closest_category(self, inputs):
        # Of the targets, return the one with the highest input
        return self._targets[inputs.argmax()]

    def display_graph(self, title="Neural Network Accuracy"):
        if not self.enable_graph:
            print("Cannot show graph: Graphing is not enabled")
            return
        width = self.graph_points.shape[0] / 30
        if width < 4:
            width = 4
        figure = plt.figure(figsize=(width, 4), dpi=96)
        graph = figure.add_subplot()
        graph.set_ylim((0, 100))
        graph.set_xlim((0, self.graph_points[-1, 0]))
        graph.plot(self.graph_points[:, 0], self.graph_points[:, 1])
        graph.set_xlabel("Epoch")
        graph.set_ylabel("Accuracy %")
        graph.set_title(title)
        figure.tight_layout()
        plt.show()


