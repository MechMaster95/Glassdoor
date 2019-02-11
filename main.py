import sys
from classifiers import Classifiers
from preprocess import Preprocess

def main():

    preprocess = Preprocess()
    preprocess.check_data_distribution()

    print "\n\n*********** ANALYSIS PART I *******************"
    partI_classifier = Classifiers(1)
    partI_classifier.draw_auc_curve(1)

    #Open this for Analysis II & close the above one
    #print "*********** ANALYSIS PART II *******************"
    #partII_classifier = Classifiers(2)
    #partII_classifier.draw_auc_curve(2)

if __name__ == "__main__":
    main()

