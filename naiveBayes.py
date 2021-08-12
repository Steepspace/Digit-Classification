# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    # print("Testing train tune")
    # print(self.legalLabels)
    # print(type(trainingLabels[0]), type(self.legalLabels[0]))
    # print("Training labels", trainingLabels)
    # print(len(self.features))
    # print('Training Data', trainingData[0])

    # for i in range(100):
    #   highestValue = trainingData[i].sortedKeys()[0]
    #   print(trainingData[i][highestValue])

    # for items in trainingData[0].items():
    #   print(items)
    # for label in trainingLabels:
    #   print(label)


    ######################
    # Prior Distribution #
    ######################
    prior = util.Counter()

    for label in trainingLabels:
      prior[label] += 1

    prior.normalize()

    self.prior = prior

    # self.calculateLogJointProbabilities(trainingData[0])

    # print(prior)

    ####################################################
    # Keep track of feature values per pixel per label #
    ####################################################
    count_dict = {}
    feature_val_dict = {}
    FEATURE_VALS = [0,1,2]

    feature_val_counter = util.Counter()
    for feature in self.features:
      feature_val_counter[feature]

    for label in self.legalLabels:
      feature_val_dict = {}
      for val in FEATURE_VALS:
        feature_val_dict[val] = feature_val_counter.copy()

      count_dict[label] = feature_val_dict

    for i in range(len(trainingLabels)):
      label = trainingLabels[i]
      for items in trainingData[i].items():
        count_dict[label][items[1]][items[0]] += 1

    # for label_item in count_dict.items():
    #   print("#####")
    #   print("Label:", label_item[0])
    #   print("#####")
    #   print()

    #   for val_item in label_item[1].items():
    #     print("#############")
    #     print("Feature Value", val_item[0])
    #     print("#############")
    #     print()

    #     print("#############")
    #     print("Counter", val_item[1])
    #     print("#############")
    #     print()

    ######################################
    # Calculate Conditional Probabilites #
    ######################################

    validation_accuracy = 0
    self.k = kgrid[0]
    self.table = None

    for k in kgrid:
      prob_table = {}

      for label, table in count_dict.items():
        total_sum = util.Counter()
        for counts in table.values():
          total_sum += counts

        total_sum.incrementAll(total_sum.keys(), len(FEATURE_VALS)*k)

        feature_dict = {}
        for feature, counts in table.items():
          temp_counts = counts.copy()
          temp_counts.incrementAll(temp_counts.keys(), k)
          feature_dict[feature] = temp_counts/total_sum

        prob_table[label] = feature_dict

      #######################
      # Validation Accuracy #
      #######################

      # print("K Value: ", self.k)
      # print("Validation Accuracy: ", validation_accuracy)

      current_accuracy = 0
      old_table = self.table
      self.table = prob_table

      for datum, label in zip(validationData, validationLabels):
        logJoint = self.calculateLogJointProbabilities(datum)
        validation_label = logJoint.argMax()

        if(validation_label == label):
          current_accuracy += 1

      # print("Current K Value: ", k)
      # print("Current Accuracy: ", current_accuracy)
      # print "--------------------------------------"
      if(current_accuracy > validation_accuracy):
        self.k = k
        validation_accuracy = current_accuracy

      else:
        self.table = old_table

    # for label_item in self.table.items():
    #   print("#####")
    #   print("Label:", label_item[0])
    #   print("#####")
    #   print()

    #   for val_item in label_item[1].items():
    #     print("#############")
    #     print("Feature Value", val_item[0])
    #     print("#############")
    #     print()

    #     print("#############")
    #     print("Counter", val_item[1])
    #     print("#############")
    #     print()

  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()

    for label in self.legalLabels:
      logJoint[label] += math.log(self.prior[label])

      for datum_key, datum_value in datum.items():
        logJoint[label] += math.log(self.table[label][datum_value][datum_key])

    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
