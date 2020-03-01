from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from ml_classifiers.neural_network import NeuralNetworkClassifier
from ml_classifiers.general import display_correct_amount


def main():
    iris = datasets.load_iris()

    ratio = 70 / 100

    print("Classifier 0")
    data_train, data_test, targets_train, targets_test = train_test_split(iris.data, iris.target, train_size=ratio)
    nodes_per_layer = [3, 3]
    classifier = NeuralNetworkClassifier(nodes_per_layer, max_iterations=20000, desired_accuracy=.90, enable_graph=True)
    classifier.fit(data_train, targets_train)
    classifier.display_graph("Classifier 0")
    print()

    max_iterations = 2000
    for i in range(5):
        print()
        print(f"Classifier {i + 1}")
        nodes_per_layer = []
        num_layers = np.random.randint(1, 4)
        for j in range(num_layers):
            nodes_per_layer.append(np.random.randint(1, 4))
        nodes_per_layer.append(3)
        data_train, data_test, targets_train, targets_test = train_test_split(iris.data, iris.target, train_size=ratio)
        c = NeuralNetworkClassifier(nodes_per_layer, desired_accuracy=0, max_iterations=max_iterations,
                                    enable_graph=True)
        c.fit(data_train, targets_train)
        c.display_graph(f"Classifier {i + 1}")
        if c.accuracy > classifier.accuracy:
            print("Replaced!")
            classifier = c
        print()
        if classifier.accuracy >= 0.99:
            print(f"Broke early on {i + 1}")
            break

    print(f"Nodes per layer: {classifier.nodes_per_layer}")
    print(f"Learning rate: {classifier.learning_rate}")
    print(f"Estimated Accuracy: {classifier.accuracy * 100:.2f}%")
    print()
    predictions = classifier.predict(data_test)

    display_correct_amount(predictions, targets_test)


if __name__ == "__main__":
    # execute only if run as a script
    main()
