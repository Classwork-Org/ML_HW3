import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal
import math
import random
from functools import partial
from multiprocessing import Pool
import pickle

class GuassianClassConditionalPdf:
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma
    def generate(self, count):
        return np.random.multivariate_normal(self.mean, self.sigma, count)
    def evaluate(self, data):
        return multivariate_normal(mean=self.mean, cov=self.sigma).pdf(data);

class Data:
    def __init__(self, features = None, labels = None, count = 0, classCount = 0, gaussianConditionals=None):
        self.features = features
        self.labels = labels
        self.count = count
        self.classCount = classCount
        self.gaussianConditionals = gaussianConditionals
    def generateData(self, gaussianConditionals, count, classPriors = None):
        self.count = count
        self.gaussianConditionals = gaussianConditionals
        self.classCount = len(gaussianConditionals)
        if classPriors is None:
            self.labels = np.array([math.floor(num) for num in np.random.uniform(0,self.classCount,count)])
            self.labels.sort()
            self.labelCount = [np.count_nonzero(self.labels == l, axis=0) for l in range(0, self.classCount)]
            self.features = []
            for eachClass in range(0, self.classCount):   
                self.features.extend(gaussianConditionals[eachClass].generate(self.labelCount[eachClass]).tolist())
            self.features = np.array(self.features)
            data_aggregate = list(zip(self.labels, self.features))
            random.shuffle(data_aggregate)
            self.labels = np.array([label for label, feature in data_aggregate])
            self.features = np.array([np.array(feature) for label, feature in data_aggregate])    
            self.labels = self.labels.astype('float32') 
            self.features = self.features.astype('float32') 

    def plotData3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for l in range(0, self.classCount):
            labelIndexes = np.where(self.labels==l)[0]
            x1 = self.features[labelIndexes,0]
            x2 = self.features[labelIndexes,1]
            x3 = self.features[labelIndexes,2]
            ax.scatter(x1,x2,x3)
        plt.show()
    def mapClassifyWithTrueData(self):
        pxGivenL = []
        for eachClass in range(0, self.classCount):
            pxGivenL.append(self.gaussianConditionals[eachClass].evaluate(self.features))
        classifyResult = np.array(np.argmax(np.array(pxGivenL), axis=0))
        labels = np.array(self.labels)
        accuracy = np.count_nonzero(np.equal(classifyResult, labels))
        return accuracy/len(labels)
    def partition(self, partitionCount):
        elementsPerParition = math.floor(self.count/10)
        paritionedDataFeatures = np.array([self.features[partition:partition+elementsPerParition] for partition in range(0, self.count, elementsPerParition)])
        paritionedDataLabels = np.array([self.labels[partition:partition+elementsPerParition] for partition in range(0, self.count, elementsPerParition)])
        dataList = []
        for eachParitionLabels, eachParitionFeatures in zip(paritionedDataLabels, paritionedDataFeatures):
            dataList.append(Data(eachParitionFeatures, eachParitionLabels, elementsPerParition, self.classCount, self.gaussianConditionals))
        return dataList
    @classmethod
    def fuse(cls, partitions):
        features = []
        labels = []
        for eachPartition in partitions:
            features.extend(eachPartition.features)
            labels.extend(eachPartition.labels)
        gaussians = partitions[0].gaussianConditionals
        classCount = len(gaussians)
        return Data(np.array(features), np.array(labels), len(features), classCount, gaussians)

class ModelHelper:
    @classmethod
    def createModel(cls, output_size, perceptrons):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(perceptrons, activation='softplus'),
            tf.keras.layers.Dense(output_size, activation='softmax'),
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
        return model
    @classmethod
    def fitModel(cls, batch_size, model, data, monitor='accuracy', min_delta=0.001, patience=5, epochs=100):
        callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience)
        model.fit(np.array(data.features, dtype=np.float32),np.array(data.labels, dtype=np.float32), epochs=epochs, batch_size=batch_size, verbose = 0 , callbacks=[callback])
        return model
    @classmethod
    def evalAccuracy(cls, model, data):
        predictions = model(data.features).numpy()
        classifyResult = np.array(np.argmax(predictions, axis=1))
        return (np.count_nonzero(classifyResult==data.labels))/data.count

def kfoldCrossValidation(kfolds, dataset, modelCreator, modelFitter, modelEvaluater, hyperparamater_range):
    hyperparameter_accuracy = {}
    for hyperparameter in hyperparamater_range:
            model = modelCreator(hyperparameter)
            partitionedDataset = dataset.partition()
            for k in range(0, kfolds):



if __name__ == "__main__":
    print(tf.test.is_gpu_available())
    d1 = Data()
    d2 = Data()
    d3 = Data()
    d4 = Data()
    dvalid = Data()
    c1 = GuassianClassConditionalPdf([1,0,0], [[1,0,0],[0,1,0],[0,0,1]])
    c2 = GuassianClassConditionalPdf([0,0,2.5], [[1,0,0],[0,1,0],[0,0,1]])
    c3 = GuassianClassConditionalPdf([0,0.5,0], [[1,0,0],[0,1,0],[0,0,1]])
    c4 = GuassianClassConditionalPdf([1,0,10], [[1,0,0],[0,1,0],[0,0,1]])
    d1.generateData([c1, c2, c3, c4], 100)
    d2.generateData([c1, c2, c3, c4], 500)
    d3.generateData([c1, c2, c3, c4], 1000)
    d4.generateData([c1, c2, c3, c4], 5000)
    datasets = [d1, d2, d3, d4]
    dvalid.generateData([c1, c2, c3, c4], 100000)

    kfolds = 10
    results = {}
    for eachDataset in datasets:
        print('Dataset: ' + str(eachDataset.count))
        
        results[str(eachDataset.count)] = {}
        results[str(eachDataset.count)]['map_accuracy'] = eachDataset.mapClassifyWithTrueData()
        results[str(eachDataset.count)]['dataset'] = {}
        results[str(eachDataset.count)]['optimal_hyperparameter'] = {}
        results[str(eachDataset.count)]['hyperParameter_accuracy'] = {}
        results[str(eachDataset.count)]['fittedModel_validation_accuracy'] = {}

        modelCreator = partial(ModelHelper.createModel, 4)
        batch_size = 2**int(math.log((eachDataset.count/kfolds)*0.50,2))
        print('Dataset batch size: ' + str(batch_size))

        modelFitter = partial(ModelHelper.fitModel, batch_size)

        start = timeit.default_timer()
        (fittedModel, accuracy, optimal_hyperparameter, hyperParameter_accuracy) = kfoldCrossValidation(kfolds, eachDataset, modelCreator, modelFitter, ModelHelper.evalAccuracy, hyperparamater_range = range(1,20))
        fittedModel_validation_accuracy = ModelHelper.evalAccuracy(fittedModel, dvalid)
        stop = timeit.default_timer()

        print('Dataset ' + str(str(eachDataset.count)) + ' Time (20 perceptrons): ', stop - start)  

        results[str(eachDataset.count)]['dataset'] = eachDataset
        results[str(eachDataset.count)]['optimal_hyperparameter'] = optimal_hyperparameter
        results[str(eachDataset.count)]['hyperParameter_accuracy'] = hyperParameter_accuracy
        results[str(eachDataset.count)]['fittedModel_validation_accuracy'] = fittedModel_validation_accuracy
        with open('final_results.pkl', 'wb') as output:
            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)

    print('Done')

    # model = ModelHelper.createModel(perceptrons = 10, output_size=4)
    # fitted_model = ModelHelper.fitModel(model, d1, batch_size=500)
    # accuracy = ModelHelper.evalAccuracy(fitted_model, dvalid)
    # print('accuracy = ' + str(accuracy))
