import collections
import math
import sys

feature_arity = []
feature_names = []
feature_data = []
class_counts = collections.defaultdict(lambda: 0)
feature_likelihoods = collections.defaultdict(lambda: 0)


def train(feature_names, feature_arity):
    """
    Count the total occurences of each class.
    Count the occurences of each feature in the training data:
        Stored in dict with key as tuples of form
        (class, feature name, feature value)
        and value as the total count.

    Return the counts for the features.
    """
    for row in feature_data:
        data_class = row[0]
        class_counts[data_class] += 1
        for col in range(1, len(row)):
            feature_likelihoods[(data_class,
                                 feature_names[col], row[col])] += 1

    return feature_likelihoods


def classify(feature_data, feature_names):
    """
    Uses training data to predict the class of new data rows.

    Takes as arguments:
        feature_data - a new data row to be classified.
        feature_names - names we trained the classifier with.

    We calculate the posterior probabilties for the data given each class,
    and return the class with the highest probability.
    """
    class_probabilities = {}

    #  calculate posterior using bayes rule
    denom = 0
    for data_class in class_counts:
        numer = 1
        for col in range(1, len(feature_data)):

            feature_value = feature_data[col]
            feature_name = feature_names[col]
            feature_count = float(feature_likelihoods[(
                data_class, feature_name, feature_value)])

            numer *= feature_count / class_counts[data_class]

        numer *= float(class_counts[data_class]) / sum(class_counts.values())
        class_probabilities[data_class] = numer
        denom += numer

    #  normalise to get probabilities
    for data_class in class_probabilities.keys():
        class_probabilities[data_class] /= denom

    print_class_probabiltiies(feature_data, feature_names, class_probabilities)
    return max(class_probabilities,
               key=lambda classLabel: class_probabilities[classLabel])


def read_training_data(filename):
    """
    Read the training data file into three data structures:

        feature_names - a list for storing the feature names
        feature_arity - a list for storing the feature arity
        feature_data - a list for storing the data rows
    """
    with open(filename) as file:
        for i, line in enumerate(file):
            if i == 0:
                feature_names = line.strip().split(' ')
            if i == 1:
                feature_arity = filter(None, line.strip().split(' '))
            if i > 1:
                feature_data.append(filter(None, line.strip().split(' ')))
    return feature_names, feature_arity


def read_test_file(filename):
    """Read test data file into a 2d list ignoring feature names and arity"""
    with open(filename) as file:
        return [filter(None, line.strip().split(' '))
                for i, line in enumerate(file) if i > 1]


def print_class_probabiltiies(input_data, feature_names, probabilities):
    """
    Prints the probabiltiies in the format specified by the assessment brief.
    e.g. P(C=0|X1=0,X2=0,X3=1) = **. P(C=1|X1=0,X2=0,X3=1) = **.
    """
    string_to_print = ""
    for data_class in range(0, len(probabilities)):
        data_class = str(data_class)
        string_to_print += "P(%s=%s|" % (feature_names[0], data_class)
        for col in range(1, len(input_data)):
            string_to_print += "%s=%s" % (feature_names[col], input_data[col])
            if (col != len(feature_names) - 1):
                string_to_print += ","
        string_to_print += ") = %f. " % (probabilities[data_class])
    print string_to_print


def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    feature_names, feature_arity = read_training_data(train_file)
    train(feature_names, feature_arity)

    test_data = read_test_file(test_file)
    for data_row in test_data:
        classify(data_row, feature_names)


if __name__ == "__main__":
    main()
