import numpy as np
import pandas as pd
from ml_classifiers.general import Classifier


class ID3Tree(Classifier):
    def __init__(self):
        self._tree = None
        self._classes = []
        self._default = None
        self._depth = 0
        self._size = 0

    def get_size(self):
        return self._size

    def get_depth(self):
        return self._depth

    def fit(self, dataset, targets):
        if self._tree is not None:
            print("Warning! Overriding previous tree!")
        self._tree, self._size = self.build_tree(pd.DataFrame.to_numpy(dataset), targets, dataset.columns.values)
        self._default = targets[np.argmax(targets)]
        for target in targets:
            if target not in self._classes:
                self._classes.append(target)

    def predict(self, dataset):
        predictions = np.zeros((0, 0))

        for _, datapoint in dataset.iterrows():
            path = []
            prediction = self._tree
            while prediction not in self._classes:
                feature = list(prediction.keys())[0]
                feature_data = datapoint[feature]
                path += [feature] + [feature_data]
                try:
                    prediction = prediction[path[-2]][path[-1]]
                except KeyError:  # This is in case the data for the feature has not been seen before
                    prediction = self._default

            path += [prediction]
            predictions = np.append(predictions, prediction)
        return predictions

    @staticmethod
    def calculate_entropy(node):
        return -node * np.log2(node) if node != 0 else 0

    def calculate_info_gain(self, data, classes, feature):
        """
        Calculates the total amount of information gained by calculating entropy for each datapoint received.
        :param data: The received data (typically from a CSV file).
        :param classes: The classes, targets, or categories, a datapoint might be.
        :param feature: The index for the attribute we are calculating for.
        :return: The amount of info gained, expressed as a float from 0 - 1, 1 being the best.
        """
        data_len = data.shape[0]
        # Get all possible values
        values = []
        for data_index in range(data_len):
            if data[data_index][feature] not in values:
                values.append(data[data_index][feature])

        feature_counts = np.zeros(len(values))
        entropy_amounts = np.zeros(len(values))

        info_gain = 0
        for value_index in range(len(values)):
            # Get each datapoint's class with this value
            datapoint_classes = []
            for data_index in range(data_len):
                if data[data_index][feature] == values[value_index]:
                    datapoint_classes.append(classes[data_index])
                    feature_counts[value_index] += 1

            # Compress all the classes into a list of relevant classes
            relevant_classes = []
            for aclass in datapoint_classes:
                if aclass not in relevant_classes:
                    relevant_classes.append(aclass)

            # Count how instances of each class there is
            class_count = np.zeros(len(relevant_classes))
            for class_index in range(len(relevant_classes)):
                for aclass in datapoint_classes:
                    if aclass == relevant_classes[class_index]:
                        class_count[class_index] += 1

            # Calculate entropy for each class
            for class_index in range(len(relevant_classes)):
                entropy_amounts[value_index] += self.calculate_entropy(class_count[class_index] / sum(class_count))

            # Add weighted entropy to info_gain
            info_gain += feature_counts[value_index] * entropy_amounts[value_index]  # / data_len
            #                                                                          Not used because it would
            #                                                                          be constant throughout the tree

        return info_gain

    def build_tree(self, data, classes, features, size=1, level=0):
        if level > self._depth:
            self._depth = level
        # Only one class left
        if len(np.unique(classes)) == 1:
            return classes[0], size

        default_class = classes[np.argmax(classes)]

        data_size = len(data)
        feature_size = len(features)

        # Return default if we've reached the end
        if data_size == 0 or feature_size == 0:
            return default_class, size

        # Create tree
        # Figure out which feature will give us the most info
        info_gain = np.zeros(feature_size)
        for feature_index in range(feature_size):
            gain = self.calculate_info_gain(data, classes, feature_index)
            info_gain[feature_index] = gain
        # Normally we subtract gain from 1 to give us the technical amount of info gained
        # but since 1 is a constant we can just take the min instead.
        best_feature = np.argmin(info_gain)
        tree = {features[best_feature]: {}}

        # Get all possible values
        values = []
        for data_index in range(len(data)):
            if data[data_index][best_feature] not in values:
                values.append(data[data_index][best_feature])

        for value in values:
            data_index = 0
            new_data = np.zeros((0, feature_size - 1))
            new_classes = np.zeros((0, 0))
            new_features = np.zeros((0, 0))
            for datapoint in data:
                if datapoint[best_feature] == value:
                    if best_feature == 0:
                        new_datapoint = datapoint[1:]
                        new_features = features[1:]
                    elif best_feature == feature_size:
                        new_datapoint = datapoint[:-1]
                        new_features = features[:-1]
                    else:
                        new_datapoint = datapoint[:best_feature]
                        new_datapoint = np.append(new_datapoint, datapoint[best_feature + 1:])
                        new_features = features[:best_feature]
                        new_features = np.append(new_features, features[best_feature + 1:])
                    new_data = np.vstack([new_data, new_datapoint])
                    new_classes = np.append(new_classes, classes[data_index])
                data_index += 1
            subtree, size = self.build_tree(new_data, new_classes, new_features, size + 1, level + 1)
            tree[features[best_feature]][value] = subtree
        return tree, size
# class ID3Tree
