import pandas as pd
from sklearn import svm, linear_model, neighbors, metrics, preprocessing
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import resample
from dataset import Dataset as ds

class Classifiers:
    """
    Class containing methods for different ML classifiers
    @author Ramesh Paudel
    """
    K_FOLD = 10

    ALGORITHMS = {
        "svm":"svm",
        "random_forest":"random_forest",
        "dt":"dt",
        "ada":"ada",
        "knn":"knn",
        "log":"log",
        "bag":"bag",
        "ann":"ann",
        "nb":"nb"
    }

    def __init__(self, analysis):
        self.process = analysis
        #Get training and test set
        self.train = ds.get_train_data_from_file(analysis)
        self.test = ds.get_test_data_from_file(analysis)

        #Split both training and test set into input (X) and output(Y)
        self.train_Y, tr_X = self.train['apply'], self.train.drop('apply', axis = 1)
        self.test_Y, te_X = self.test['apply'], self.test.drop('apply', axis = 1)

        #scale data to normal distribution (have mean zero and standard deviation 1)
        self.train_X = preprocessing.scale(tr_X)
        self.test_X = preprocessing.scale(te_X)


    def logistic_regression(self):
        """
        Uses Logisitic Regression Classifier,  by downsampling the majority class, and upsampling the minority class and balancing the dataset
        :return: Predicted : Array of Class predicted by the classifier
        """

        print("\n\n---- Initial Class Distribution ---- ")
        print(self.train_Y.value_counts())

        log = linear_model.LogisticRegression()
        log.fit(self.train_X, self.train_Y)
        p_org = log.predict(self.test_X)
        print "\n\n Logistic Regression (Original)"
        print "===========================================\n"
        self.display_report(self.test_Y, p_org, self.ALGORITHMS["log"])

        print("\n---- Predicted Class Distribution ---- ")
        print( pd.value_counts(pd.Series(p_org)))

        #partition data based on class
        apply = self.train[self.train['apply'] == 1]
        noapply = self.train[self.train['apply'] == 0]
        #down sample the majority class (here no-apply class) to match minority class
        majority_downsampled = resample(noapply,
                                           replace=False,  # sample without replacement
                                           n_samples=97417,  # to match minority class
                                           random_state=123)  # reproducible results
        #concat minority class as well downsampled majority class, this will be the balanced dataset
        downsampled = pd.concat([majority_downsampled, apply])

        y = downsampled['apply']
        X = downsampled.drop('apply', axis=1)
        #use Logistic Regression to predict
        log = linear_model.LogisticRegression()
        log.fit(X, y)

        p_down = log.predict(self.test_X)
        print "\n\n Logistic Regression (undersampled)"
        print "===========================================\n"
        self.display_report(self.test_Y, p_down, self.ALGORITHMS["log"])
        print("\n---- Predicted Class Distribution ---- ")
        print(pd.value_counts(pd.Series(p_down)))

        # up sample the minority class (apply class) to match majority class
        minority_upsampled = resample(apply,
                                      replace=True,  # sample with replacement
                                      n_samples=986947,  # to match majority class
                                      random_state=123)  # reproducible results
        # concat majority class as well upsampled minority class, this will be the balanced dataset
        upsampled = pd.concat([minority_upsampled, noapply])
        print("\n---- Upsampled Training Set Class Distribution ---- ")
        print(upsampled['apply'].value_counts())
        y_up = upsampled['apply']
        X_up = upsampled.drop('apply', axis=1)

        log.fit(X, y)
        p_up = log.predict(self.test_X)
        print "\n\n Logistic Regression (oversampled)"
        print "===========================================\n"
        self.display_report(self.test_Y, p_up, self.ALGORITHMS["log"])
        print("\n---- Predicted Class Distribution ---- ")
        print(pd.value_counts(pd.Series(p_up)))

        log = linear_model.LogisticRegression(class_weight='balanced')
        log.fit(self.train_X, self.train_Y)
        predicted = log.predict(self.test_X)
        print "\n\n Logistic Regression (Class Weight)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["log"])

        return predicted

    def naive_bayes(self):
        """
        Uses Multinomial Naive Bayes Classifier
        :return: Predicted : Array of Class predicted by the classifier
         """
        print("\n\n ---- Initial Class Distribution ---- ")
        print(self.train_Y.value_counts())


        # partition data based on class
        apply = self.train[self.train['apply'] == 1]
        noapply = self.train[self.train['apply'] == 0]
        # down sample the majority class (here no-apply class) to match minority class
        majority_downsampled = resample(noapply,
                                        replace=False,  # sample without replacement
                                        n_samples=97417,  # to match minority class
                                        random_state=123)  # reproducible results
        # concat minority class as well downsampled majority class, this will be the balanced dataset
        downsampled = pd.concat([majority_downsampled, apply])

        y = downsampled['apply']
        X = downsampled.drop('apply', axis=1)
        mnb = GaussianNB()
        mnb.fit(X, y)
        predicted = mnb.predict(self.test_X)
        print "\n\n Naive Bayes Report (Downsampled)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["nb"])

        mnb = GaussianNB()
        mnb.fit(self.train_X, self.train_Y)
        predicted = mnb.predict(self.test_X)
        print "\n\n Naive Bayes (Original)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["nb"])

        print("\n---- Predicted Class Distribution ---- ")
        print(pd.value_counts(pd.Series(predicted)))

        return predicted

    def ann_classify(self, max_iteration=2000):
        """
        Uses Artificial Neural Network Classifier
        :return: None
        """

        # partition data based on class
        apply = self.train[self.train['apply'] == 1]
        noapply = self.train[self.train['apply'] == 0]
        # down sample the majority class (here no-apply class) to match minority class
        majority_downsampled = resample(noapply,
                                        replace=False,  # sample without replacement
                                        n_samples=97417,  # to match minority class
                                        random_state=123)  # reproducible results
        # concat minority class as well downsampled majority class, this will be the balanced dataset
        downsampled = pd.concat([majority_downsampled, apply])

        y = downsampled['apply']
        X = downsampled.drop('apply', axis=1)

        mlp = MLPClassifier(hidden_layer_sizes=(200,), solver="sgd", max_iter=max_iteration,
                            learning_rate_init=0.01, learning_rate="adaptive")
        mlp.fit(X, y)
        predicted = mlp.predict(self.test_X)
        print "\n\n Artificial Neural Network Report"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["ann"])

        mlp1 = MLPClassifier(hidden_layer_sizes=(200,), solver="sgd", max_iter=max_iteration,
                             learning_rate_init=0.01, learning_rate="adaptive")
        mlp1.fit(self.train_X, self.train_Y)
        predicted = mlp1.predict(self.test_X)
        print "\n\n Artificial Neural Network Report (Original)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["ann"])

        return predicted

    def decision_tree_classify(self):
        """
        Uses Decision Tree Classifier
        :return: Predicted : Array of Class predicted by the classifier
        """
        decision_tree_classifier = DecisionTreeClassifier()
        decision_tree_classifier.fit(self.train_X, self.train_Y)
        predicted = decision_tree_classifier.predict(self.test_X)
        print "\n\n Decision Tree (Original)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["dt"])


        # partition data based on class
        apply = self.train[self.train['apply'] == 1]
        noapply = self.train[self.train['apply'] == 0]
        # down sample the majority class (here no-apply class) to match minority class
        majority_downsampled = resample(noapply,
                                        replace=False,  # sample without replacement
                                        n_samples=97417,  # to match minority class
                                        random_state=123)  # reproducible results
        # concat minority class as well downsampled majority class, this will be the balanced dataset
        downsampled = pd.concat([majority_downsampled, apply])

        y = downsampled['apply']
        X = downsampled.drop('apply', axis=1)
        decision_tree_classifier = DecisionTreeClassifier()
        decision_tree_classifier.fit(X, y)
        predicted = decision_tree_classifier.predict(self.test_X)
        print "\n\n Decision Tree Report (DownSampled)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["dt"])

        decision_tree_classifier = DecisionTreeClassifier(class_weight='balanced')
        decision_tree_classifier.fit(self.train_X, self.train_Y)
        predicted = decision_tree_classifier.predict(self.test_X)
        print "\n\n Decision Tree (Balanced)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["dt"])

        return predicted

    def bagging_decision_tree(self):
        """
        Uses Decision Tree Classifier
        :return: Predicted : Array of Class predicted by the classifier
        """
        # Bagged Decision Trees for Classification
        # partition data based on class
        apply = self.train[self.train['apply'] == 1]
        noapply = self.train[self.train['apply'] == 0]
        # down sample the majority class (here no-apply class) to match minority class
        majority_downsampled = resample(noapply,
                                        replace=False,  # sample without replacement
                                        n_samples=97417,  # to match minority class
                                        random_state=123)  # reproducible results
        # concat minority class as well downsampled majority class, this will be the balanced dataset
        downsampled = pd.concat([majority_downsampled, apply])

        y = downsampled['apply']
        X = downsampled.drop('apply', axis=1)

        dt = DecisionTreeClassifier(class_weight='balanced')
        num_trees = 25
        seed = 7
        model = BaggingClassifier(base_estimator=dt, n_estimators=num_trees, random_state=seed)
        model.fit(X, y)
        print "\n\n Bagging Report (Balanced)"
        print "===========================================\n"
        predicted = model.predict(self.test_X)
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["bag"])
        return predicted

    def random_forest_classify(self):
        """
        Uses Random Forest Classifier
        :return: Predicted : Array of Class predicted by the classifier
        """
        random_forest_classifier = RandomForestClassifier()
        random_forest_classifier.fit(self.train_X, self.train_Y)
        predicted = random_forest_classifier.predict(self.test_X)
        print "\n\n Random Forest (Original)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["random_forest"])


        # partition data based on class
        apply = self.train[self.train['apply'] == 1]
        noapply = self.train[self.train['apply'] == 0]
        # down sample the majority class (here no-apply class) to match minority class
        majority_downsampled = resample(noapply,
                                        replace=False,  # sample without replacement
                                        n_samples=97417,  # to match minority class
                                        random_state=123)  # reproducible results
        # concat minority class as well downsampled majority class, this will be the balanced dataset
        downsampled = pd.concat([majority_downsampled, apply])

        y = downsampled['apply']
        X = downsampled.drop('apply', axis=1)

        random_forest_classifier = RandomForestClassifier()
        random_forest_classifier.fit(X, y)
        predicted = random_forest_classifier.predict(self.test_X)
        print "\n\n Random Forest Report (Downsampled)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["random_forest"])

        random_forest_classifier = RandomForestClassifier(class_weight='balanced')
        random_forest_classifier.fit(self.train_X, self.train_Y)
        predicted = random_forest_classifier.predict(self.test_X)
        print "\n\n Random Forest (Balanced)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["random_forest"])

        return predicted

    def ada_boost_classify(self):
        """
        Uses Ada-Boost Classifier
        :return: Predicted : Array of Class predicted by the classifier
        """

        # partition data based on class
        apply = self.train[self.train['apply'] == 1]
        noapply = self.train[self.train['apply'] == 0]
        # down sample the majority class (here no-apply class) to match minority class
        majority_downsampled = resample(noapply,
                                        replace=False,  # sample without replacement
                                        n_samples=97417,  # to match minority class
                                        random_state=123)  # reproducible results
        # concat minority class as well downsampled majority class, this will be the balanced dataset
        downsampled = pd.concat([majority_downsampled, apply])

        y = downsampled['apply']
        X = downsampled.drop('apply', axis=1)

        ada_boost_classifier = AdaBoostClassifier()
        ada_boost_classifier.fit(X, y)
        predicted = ada_boost_classifier.predict(self.test_X)
        print "\n\n Ada Boost Report (Downsampled)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["ada"])

        ada_boost_classifier1 = AdaBoostClassifier()
        ada_boost_classifier1.fit(self.train_X, self.train_Y)
        predicted = ada_boost_classifier1.predict(self.test_X)
        print "\n\n Ada Boost Classifier (Original)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["ada"])

        return predicted

    def knn_classify(self):
        """
        Uses K-Nearest Neighbor Classifier
        :return: Predicted : Array of Class predicted by the classifier
        """

        neigh1 = neighbors.KNeighborsClassifier(weights='distance')
        neigh1.fit(self.train_X, self.train_Y)
        predicted = neigh1.predict(self.test_X)
        print "\n\n KNN Classifier Report (Original)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["knn"])

        # partition data based on class
        apply = self.train[self.train['apply'] == 1]
        noapply = self.train[self.train['apply'] == 0]
        # down sample the majority class (here no-apply class) to match minority class
        majority_downsampled = resample(noapply,
                                        replace=False,  # sample without replacement
                                        n_samples=97417,  # to match minority class
                                        random_state=123)  # reproducible results
        # concat minority class as well downsampled majority class, this will be the balanced dataset
        downsampled = pd.concat([majority_downsampled, apply])

        y = downsampled['apply']
        X = downsampled.drop('apply', axis=1)
        neigh = neighbors.KNeighborsClassifier(weights='distance')
        neigh.fit(X, y)
        predicted = neigh.predict(self.test_X)
        print "\n\n KNN Classifier (Downsampled)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["knn"])
        return predicted

    def svm_classify(self):
        """
        Uses Support Vector Machine Classifier
        :return: Predicted : Array of Class predicted by the classifier
        """
        svm_classifier1 = svm.SVC(kernel="linear")
        svm_classifier1.fit(self.train_X, self.train_Y)
        predicted = svm_classifier1.predict(self.test_X)
        print "\n\n SVM Classifier Report (Original)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["svm"])


        # partition data based on class
        apply = self.train[self.train['apply'] == 1]
        noapply = self.train[self.train['apply'] == 0]
        # down sample the majority class (here no-apply class) to match minority class
        majority_downsampled = resample(noapply,
                                        replace=False,  # sample without replacement
                                        n_samples=97417,  # to match minority class
                                        random_state=123)  # reproducible results
        # concat minority class as well downsampled majority class, this will be the balanced dataset
        downsampled = pd.concat([majority_downsampled, apply])

        y = downsampled['apply']
        X = downsampled.drop('apply', axis=1)

        svm_classifier = svm.SVC(kernel="linear")
        svm_classifier.fit(X, y)
        predicted = svm_classifier.predict(self.test_X)
        print "\n\n SVM Classifier Report (Downsampled)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["svm"])

        svm_classifier1 = svm.SVC(kernel="linear", class_weight='balanced')
        svm_classifier1.fit(self.train_X, self.train_Y)
        predicted = svm_classifier1.predict(self.test_X)
        print "\n\n SVM Classifier Report (Balanced)"
        print "===========================================\n"
        self.display_report(self.test_Y, predicted, self.ALGORITHMS["svm"])

        return predicted

    def draw_auc_curve(self, analysis):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        #plt.figure()
        log_pred = self.logistic_regression()
        fpr["log1"], tpr["log1"], _ = roc_curve(self.test_Y.ravel(), log_pred.ravel())
        roc_auc["log1"] = auc(fpr["log1"], tpr["log1"])
        #plt.figure()
        plt.plot(fpr["log1"], tpr["log1"], color='orange', lw=2,
                 label='Logistic Reg. (auc = %0.2f)' % roc_auc["log1"])
        print "log " + str(roc_auc["log1"])

        nb_pred = self.naive_bayes()
        fpr["nb1"], tpr["nb1"], _ = roc_curve(self.test_Y.ravel(), nb_pred.ravel())
        roc_auc["nb1"] = auc(fpr["nb1"], tpr["nb1"])
        #plt.figure()
        plt.plot(fpr["nb1"], tpr["nb1"], color='brown', lw=2,
                 label='Naive Bayes. (auc = %0.2f)' % roc_auc["nb1"])
        print "nb1 " + str(roc_auc["nb1"])

        dt_pred = self.decision_tree_classify()
        fpr["dt"], tpr["dt"], _ = roc_curve(self.test_Y.ravel(), dt_pred.ravel())
        roc_auc["dt"] = auc(fpr["dt"], tpr["dt"])
        plt.plot(fpr["dt"], tpr["dt"], color='green', lw=2,
                 label='Decision Tree (auc = %0.2f)' % roc_auc["dt"])
        print "dt " + str(roc_auc["dt"])

        predicted = self.random_forest_classify()
        fpr["rf"], tpr["rf"], _ = roc_curve(self.test_Y.ravel(), predicted.ravel())
        roc_auc["rf"] = auc(fpr["rf"], tpr["rf"])
        plt.plot(fpr["rf"], tpr["rf"], color='black', lw=2,
                 label='Random Forest (auc = %0.2f)' % roc_auc["rf"])
        print "rf" + str(roc_auc["rf"])

        predicted = self.bagging_decision_tree()
        fpr["bag"], tpr["bag"], _ = roc_curve(self.test_Y.ravel(), predicted.ravel())
        roc_auc["bag"] = auc(fpr["bag"], tpr["bag"])
        plt.plot(fpr["bag"], tpr["bag"], color='skyblue', lw=2,
                 label='Bagging (auc = %0.2f)' % roc_auc["bag"])
        print "bag" + str(roc_auc["bag"])

        predicted = self.ada_boost_classify()
        fpr["ab"], tpr["ab"], _ = roc_curve(self.test_Y.ravel(), predicted.ravel())
        roc_auc["ab"] = auc(fpr["ab"], tpr["ab"])
        plt.plot(fpr["ab"], tpr["ab"], color='navy', lw=2,
                 label='Ada Boost (auc = %0.2f)' % roc_auc["ab"])
        print "ab" + str(roc_auc["ab"])

        nn_pred = self.ann_classify()
        fpr["nn1"], tpr["nn1"], _ = roc_curve(self.test_Y.ravel(), nn_pred.ravel())
        roc_auc["nn1"] = auc(fpr["nn1"], tpr["nn1"])
        #plt.figure()
        plt.plot(fpr["nn1"], tpr["nn1"], color='gray', lw=2,
                 label='Neural Network (auc = %0.2f)' % roc_auc["nn1"])
        print "nn1 " + str(roc_auc["nn1"])

        predicted = self.knn_classify()
        fpr["knn"], tpr["knn"], _ = roc_curve(self.test_Y.ravel(), predicted.ravel())
        roc_auc["knn"] = auc(fpr["knn"], tpr["knn"])
        plt.plot(fpr["knn"], tpr["knn"], color='red', lw=2,
                 label='KNN (auc = %0.2f)' % roc_auc["knn"])
        print "knn" + str(roc_auc["knn"])

        predicted = self.svm_classify()
        fpr["svm1"], tpr["svm1"], _ = roc_curve(self.test_Y.ravel(), predicted.ravel())
        roc_auc["svm1"] = auc(fpr["svm1"], tpr["svm1"])
        plt.plot(fpr["svm1"], tpr["svm1"], color='deeppink', lw=2,
                 label='SVM (auc = %0.2f)' % roc_auc["svm1"])
        print "SVM" + str(roc_auc["svm1"])

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('AUC Curve for different algorithm')
        plt.legend(loc="lower right")
        plt.savefig("figures/auc_"+str(analysis)+".png")

    def display_report(self, target, prediction, algorithm):
        """
        Displays the classification report
        :param target: target output
        :param prediction: Prediction of the model
        :param algorithm: Algorithm used to generate predictions
        :return: None
        """
        print("area under curve (auc): ", metrics.roc_auc_score(target, prediction))
        print classification_report(target,prediction)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(target, prediction)
