from ml_classifiers.general import Classifier


class HardCodedClassifier(Classifier):
    def fit(self, dataset, targets):
        print("\n\nReally? You tried to fit? It's called a Hardcoded classifier. It doesn't fit. It doesn't train. It\n"
              "just does one thing. 0. It does 0. That's it. Nothing fancy. Nothing intelligent. It just 0s. Nothing \n"
              "to get excited about. In fact, what you should be doing is writing a classifier that actually does \n"
              "something. Those machines aren't going to learn themselves. Now, if you happen to have a set of data \n"
              "that are all the same type, and that type happens to be 0, this classifier might work for you. \n"
              "Otherwise, I don't recommend it. It's just a waste of time. It's a waste of my time. It's a waste of \n"
              "your time. It's a waste of everyone's time. So how about you get on with that intelligent classifier, \n"
              "eh? Something that actually learns? Something that will do the thing? That'd be good. Have fun!\n")
        return

    def predict(self, dataset):
        return [0] * len(dataset)
# class HardCodedClassifier
