class Classifier:
    def fit(self, dataset, targets):
        raise NotImplementedError("#fit is not yet implemented for " + str(type(self)))

    def predict(self, dataset):
        raise NotImplementedError("#predict is not yet implemented for " + str(type(self)))
# class Classifier


class ClassifierData:
    def __init__(self, data, target=None, comparator=None):
        self._data = data
        self._target = target
        self.comparator = comparator

    def __gt__(self, other):
        return self.comparator.is_greater(self, other)

    def __lt__(self, other):
        return self.comparator.is_lesser(self, other)

    def __eq__(self, other):
        return self.comparator.is_equal(self, other)

    def get_data(self):
        return self._data

    def get_target(self):
        return self._target

    def set_target(self, target):
        self._target = target
# class ClassifierData


class FixedSizeList:
    def __init__(self, size, data=None):
        self.size = size
        if data is None:
            data = [None] * size
        elif type(data) is not list:
            data = [data]

        if len(data) != size:
            data = data * (size // len(data))

        self.data = data

    def __len__(self):
        return self.get_size()

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.size:
            self.n += 1
            return self.data[self.n - 1]
        raise StopIteration

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return str(self.data)

    def get_size(self):
        return self.size

    def insert(self, location, item):
        self.data.insert(location, item)
        self.data.pop()
        return self


    def remove(self, item):
        self.data.remove(item)
        self.data.append(None)
        return self

    def copy(self):
        return FixedSizeList(self.size, self.data.copy())

    def clear(self):
        self.data = [None] * self.size

    def get_data(self):
        return self.data

    def fill(self, element):
        for i in range(self.size):
            if self.data[i] is None:
                self.data[i] = element
        return self

    def append(self, element):
        self.data.append(element)
        self.data.pop(0)
        return self
# class FixedSizeList


def get_correct(predictions, targets_test):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == targets_test[i]:
            correct += 1
    return correct


def display_correct_amount(predictions, targets_test, display_number=True, display_percent=True):
    correct = get_correct(predictions, targets_test)

    if display_number:
        print(f"Correct: {correct} of {len(targets_test)}")
    if display_percent:
        print(f"Percent: {correct / len(predictions) * 100:.2f}%")

