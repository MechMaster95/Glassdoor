from dataset import Dataset as ds
from sklearn import preprocessing

class Preprocess:
    """
       Class containing methods for preprocessing data
       @author Ramesh Paudel
       """

    def __init__(self):
        # Get training and test set
        self.data = ds.read_data()

    def check_data_distribution(self):
        print "\n------- Mean ------"
        print(self.data.mean(axis=0))
        print "\n\n------- Standard Deviation -------"
        print(self.data.std(axis=0))

        dt = preprocessing.scale(self.data)
