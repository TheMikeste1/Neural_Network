import numpy as np
from ml_classifiers.general import Classifier
from ml_classifiers.general import ClassifierData
from ml_classifiers.general import FixedSizeList


class kNNClassifier(Classifier):
    def __init__(self, k=3, predict_category=True):
        self._data = []
        if k < 1:
            print("k must be greater than 0! Resetting to 3.")
            k = 3
        self._k = k
        self.predict_category = predict_category

    def fit(self, dataset, targets):
        i = 0
        for data in dataset:
            self._data.append(ClassifierData(data, targets[i]))
            i += 1

    def get_nearest_neighbors(self, point, data: [ClassifierData]):
        k = self._k
        if k > len(self._data):
            k = len(self._data)

        nearest_neighbors = FixedSizeList(k)
        nearest_neighbors_distances = FixedSizeList(k)

        for i in range(len(data)):
            for j in range(k):
                distance = self.calculate_euclidean_radicand(point, data[i].get_data())
                if nearest_neighbors[j] is None or distance < nearest_neighbors_distances[j]:
                    nearest_neighbors.insert(j, data[i])
                    nearest_neighbors_distances.insert(j, distance)
                    break
        return nearest_neighbors

    def calculate_prediction(self, nearest_neighbors):
        if not self.predict_category:
            total = 0
            for neighbor in nearest_neighbors:
                total += neighbor.get_target()
            return total / self._k

        types = []
        occurrences = []

        for neighbor in nearest_neighbors:
            added = False
            for i in range(len(types)):
                if types[i] == neighbor.get_target():
                    occurrences[i] += 1
                    added = True
                    break
            if not added:
                types.append(neighbor.get_target())
                occurrences.append(1)

        i = 0
        largest_number_of_occurrences = occurrences[0]
        for j in range(1, len(occurrences)):
            if occurrences[j] > largest_number_of_occurrences:
                i = j
                largest_number_of_occurrences = occurrences[j]
            elif occurrences[j] == largest_number_of_occurrences:
                amount = 0
                for times in occurrences:
                    amount += times
                return self.calculate_prediction(nearest_neighbors[:-(amount // 3 * 2)])

        return types[i]

    def predict(self, dataset):
        # This can be optimized by only checking data near our data
        predictions = np.array([])
        if self._k > len(self._data):
            print("Warning: k is greater than the amount of data. This will likely cause inaccuracies.")
        for data in dataset:
            nearest_neighbors = self.get_nearest_neighbors(data, self._data)
            predictions = np.append(predictions, self.calculate_prediction(nearest_neighbors))

        return predictions

    def set_k(self, k):
        if k < 1:
            raise AttributeError("k must be greater than 0!")
        self._k = k

    def get_data(self):
        return self._data

    def get_k(self):
        return self._k

    def calculate_euclidean_radicand(self, point_a, point_b):
        if len(point_a) != len(point_b):
            raise NotImplementedError("Point A and Point B must be equal coords!")
        total = 0
        for coord in range(0, len(point_a)):
            total += (int(point_a[coord]) - int(point_b[coord])) ** 2
        return total

    def calculate_euclidean_distance(self, point_a, point_b):
        print("#calculate_euclidean_distance wastes processing power, as it it simply square roots the radicand. It is "
              "recommended to simply use #calculate_euclidean_radicand")
        return np.sqrt(self.calculate_euclidean_radicand(point_a, point_b))
# class kNNClassifier
