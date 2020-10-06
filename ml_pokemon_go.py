import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Initialization for Homework 2

class DataPoint(object): # DataPoint class helps to group data and methods
    def __init__(self, attr):
        self.stamina = attr['stamina']
        self.attVal = attr['attack_value']
        self.defVal = attr['defense_value']
        self.capRate = attr['capture_rate']
        self.fleeRate = attr['flee_rate']
        self.spChan = attr['spawn_chance']
        self.primStr = attr['primary_strength']
        self.ptOut = attr['combat_point_outcome']

    def attributes_vector(self):
        return np.array([self.stamina, self.attVal, self.defVal, self.capRate, self.fleeRate, self.spChan, self.primStr, self.ptOut])

    def __str__(self):
        return "Stamina: {}, Attack Value: {}, Defense Value: {}, Capture Rate: {}, Flee Rate: {}, Spawn Chance: {}, Primary Strength: {}, Combat Point Outcome: {}".format(self.stamina, self.attVal, self.defVal, self.capRate, self.fleeRate, self.spChan, self.primStr, self.ptOut)
        
# the following code is for Question (i)

def parse_dataset(filename):
    data_file = open(filename, 'r')  # Open File "to read"
    dataset = []  # List to hold Datapoint objects

    for index, line in enumerate(data_file):
        if index == 0:  # First line describes the datapoint, it's not an actual datapoint
            continue  # do nothing, it will skip all the following code
        name, stamina, attVal, defVal, capRate, fleeRate, spChan, primStr, ptOut = line.strip().split(',')  # strip() removes '\n', and split(',') splits the line at tabs
        dataset.append(DataPoint({'stamina':float(stamina), 'attack_value':float(attVal), 'defense_value':float(defVal), 'capture_rate':float(capRate), 'flee_rate':float(fleeRate), 'spawn_chance':float(spChan), 'primary_strength':primStr, 'combat_point_outcome':float(ptOut)}))  # Create DataPoint object for the given data

    return dataset

poke_set = parse_dataset('hw2_data.csv')

# the following code is for Question (ii)

# create scatter plots to examine relationship between each numerical feature and outcome of interest (combat point)
def plot_scatters(dataset):
    attr1 = [data.stamina for data in dataset]
    attr2 = [data.attVal for data in dataset]
    attr3 = [data.defVal for data in dataset]
    attr4 = [data.capRate for data in dataset]
    attr5 = [data.fleeRate for data in dataset]
    attr6 = [data.spChan for data in dataset]
    pt_outcome = [data.ptOut for data in dataset]

    plt.figure(figsize = (9, 6)) 
    
    s1 = plt.subplot(231)
    s1.scatter(attr1, pt_outcome, c='b', marker='o')
    plt.xlabel('Stamina')
    plt.ylabel('Combat Points')

    s2 = plt.subplot(232)
    s2.scatter(attr2, pt_outcome, c='b', marker='o')
    plt.xlabel('Attack Value')
    plt.ylabel('Combat Points')

    s3 = plt.subplot(233)
    s3.scatter(attr3, pt_outcome, c='b', marker='o')
    plt.xlabel('Defense Value')
    plt.ylabel('Combat Points')

    s4 = plt.subplot(234)
    s4.scatter(attr4, pt_outcome, c='b', marker='o')
    plt.xlabel('Capture Rate')
    plt.ylabel('Combat Points')

    s5 = plt.subplot(235)
    s5.scatter(attr5, pt_outcome, c='b', marker='o')
    plt.xlabel('Flee Rate')
    plt.ylabel('Combat Points')

    s6 = plt.subplot(236)
    s6.scatter(attr6, pt_outcome, c='b', marker='o')
    plt.xlabel('Spawn Chance')
    plt.ylabel('Combat Points')

    plt.suptitle('Correlation between Numerical Attributes and Outcome')
    plt.show()
    
    # compute Pearson's Coefficient for each attribute
    pc1 = np.corrcoef(attr1, pt_outcome)[0, 1]
    pc2 = np.corrcoef(attr2, pt_outcome)[0, 1]
    pc3 = np.corrcoef(attr3, pt_outcome)[0, 1]
    pc4 = np.corrcoef(attr4, pt_outcome)[0, 1]
    pc5 = np.corrcoef(attr5, pt_outcome)[0, 1]
    pc6 = np.corrcoef(attr6, pt_outcome)[0, 1]
    print(pc1, pc2, pc3)
    print(pc4, pc5, pc6)

# plot_scatters(poke_set)

# the following code is for Question (iii)

# create scatter plots to examine relationship between the numerical attributes themselves
def plot_scatters_attr(dataset):
    attr1 = [data.stamina for data in dataset]
    attr2 = [data.attVal for data in dataset]
    attr3 = [data.defVal for data in dataset]
    attr4 = [data.capRate for data in dataset]
    attr5 = [data.fleeRate for data in dataset]
    attr6 = [data.spChan for data in dataset]

    plt.figure(figsize = (15, 9)) 
    
    s1 = plt.subplot2grid((3,5), (0,0))
    s1.scatter(attr1, attr2, c='b', marker='o')
    plt.xlabel('Stamina')
    plt.ylabel('Attack Value')

    s2 = plt.subplot2grid((3,5), (0,1))
    s2.scatter(attr1, attr3, c='b', marker='o')
    plt.xlabel('Stamina')
    plt.ylabel('Defense Value')

    s3 = plt.subplot2grid((3,5), (0,2))
    s3.scatter(attr1, attr4, c='b', marker='o')
    plt.xlabel('Stamina')
    plt.ylabel('Capture Rate')

    s4 = plt.subplot2grid((3,5), (0,3))
    s4.scatter(attr1, attr5, c='b', marker='o')
    plt.xlabel('Stamina')
    plt.ylabel('Flee Rate')

    s5 = plt.subplot2grid((3,5), (0,4))
    s5.scatter(attr1, attr6, c='b', marker='o')
    plt.xlabel('Stamina')
    plt.ylabel('Spawn Chance')

    s6 = plt.subplot2grid((3,5), (1,0))
    s6.scatter(attr2, attr3, c='b', marker='o')
    plt.xlabel('Attack Value')
    plt.ylabel('Defense Value')

    s7 = plt.subplot2grid((3,5), (1,1))
    s7.scatter(attr2, attr4, c='b', marker='o')
    plt.xlabel('Attack Value')
    plt.ylabel('Capture Rate')

    s8 = plt.subplot2grid((3,5), (1,2))
    s8.scatter(attr2, attr5, c='b', marker='o')
    plt.xlabel('Attack Value')
    plt.ylabel('Flee Rate')

    s9 = plt.subplot2grid((3,5), (1,3))
    s9.scatter(attr2, attr6, c='b', marker='o')
    plt.xlabel('Attack Value')
    plt.ylabel('Spawn Chance')

    s10 = plt.subplot2grid((3,5), (1,4))
    s10.scatter(attr3, attr4, c='b', marker='o')
    plt.xlabel('Defense Value')
    plt.ylabel('Capture Rate')

    s11 = plt.subplot2grid((3,5), (2,0))
    s11.scatter(attr3, attr5, c='b', marker='o')
    plt.xlabel('Defense Value')
    plt.ylabel('Flee Rate')

    s12 = plt.subplot2grid((3,5), (2,1))
    s12.scatter(attr3, attr6, c='b', marker='o')
    plt.xlabel('Defense Value')
    plt.ylabel('Spawn Chance')

    s13 = plt.subplot2grid((3,5), (2,2))
    s13.scatter(attr4, attr5, c='b', marker='o')
    plt.xlabel('Capture Rate')
    plt.ylabel('Flee Rate')

    s14 = plt.subplot2grid((3,5), (2,3))
    s14.scatter(attr4, attr6, c='b', marker='o')
    plt.xlabel('Capture Rate')
    plt.ylabel('Spawn Chance')

    s15 = plt.subplot2grid((3,5), (2,4))
    s15.scatter(attr5, attr6, c='b', marker='o')
    plt.xlabel('Flee Rate')
    plt.ylabel('Spawn Chance')

    plt.suptitle("Correlation between Numerical Attributes")
    plt.show()
    
    # compute Pearson's Coefficient for each attribute
    pc1 = np.corrcoef(attr1, attr2)[0, 1]
    pc2 = np.corrcoef(attr1, attr3)[0, 1]
    pc3 = np.corrcoef(attr1, attr4)[0, 1]
    pc4 = np.corrcoef(attr1, attr5)[0, 1]
    pc5 = np.corrcoef(attr1, attr6)[0, 1]
    pc6 = np.corrcoef(attr2, attr3)[0, 1]
    pc7 = np.corrcoef(attr2, attr4)[0, 1]
    pc8 = np.corrcoef(attr2, attr5)[0, 1]
    pc9 = np.corrcoef(attr2, attr6)[0, 1]
    pc10 = np.corrcoef(attr3, attr4)[0, 1]
    pc11 = np.corrcoef(attr3, attr5)[0, 1]
    pc12 = np.corrcoef(attr3, attr6)[0, 1]
    pc13 = np.corrcoef(attr4, attr5)[0, 1]
    pc14 = np.corrcoef(attr4, attr6)[0, 1]
    pc15 = np.corrcoef(attr5, attr6)[0, 1]

    print(pc1, pc2, pc3, pc4, pc5)
    print(pc6, pc7, pc8, pc9, pc10)
    print(pc11, pc12, pc13, pc14, pc15)

# plot_scatters_attr(poke_set)

# the following code is for Question (vi)

def one_hot_encoding(dataset):
    strength = ["Grass", "Fire", "Water", "Bug", "Normal", "Poison", "Electric", "Ground", "Fairy", "Fighting", "Psychic", "Rock", "Ghost", "Ice", "Dragon"] # define possible primary strength for a pokemon
    for data in dataset:
        poke = data.primStr
        encoded = [0 for _ in range(len(strength))]
        for i in range(len(strength)):
            if poke == strength[i]:
                encoded[i] = 1
        data.primStr = encoded
    return dataset

encoded_set = one_hot_encoding(poke_set)


# the following code is for Question (v)

def LR_OLS(dataset, X, y):
    XT = X.transpose()
    XTX = np.matmul(XT, X)
    XTXinv = np.linalg.pinv(XTX)
    w = np.matmul(np.matmul(XTXinv, XT), y)
    return w

def computeRSS(validSet, w, X, y):
    Xw = np.matmul(X, w)
    y_Xw = y - Xw
    y_XwT = y_Xw.transpose()
    RSS = np.matmul(y_XwT, y_Xw)
    return math.sqrt(RSS)


# the following code is for both Question (v) and Question (vi)

# def LR_regular(dataset, X, y, reg_term): # linear regression with regularization
    XT = X.transpose()
    XTX = np.matmul(XT, X)

    s = len(XTX)
    iMatrix = np.zeros([s,s]) # initialize D by D identity matrix
    for i in range(len(iMatrix)):
        iMatrix[i][i] = 1
    regI = reg_term * iMatrix
    
    XTX_reg = XTX + regI
    XTXinv = np.linalg.pinv(XTX_reg)
    w = np.matmul(np.matmul(XTXinv, XT), y)
    return w


def cross_validate(dataset, k, reg_term):
    random.shuffle(dataset)
    folds = np.array_split(dataset, k)
    sqrtVal = [] # store the square root of RSS at every fold
    for i in range(k):
        trainSet = list()
        for j in range(len(folds)):
            if j == i:
                validSet = folds[j] # use one fifth of entire dataset as validation set
            else:
                for dp in folds[j]:
                    trainSet.append(dp)
        
        # now create x and y matrix for training set
        nT = len(trainSet)
        X = np.empty([nT, 22]) # N by D matrix. D = 7+15 because 7 numerical attributes, and 15 categories in the categorical attribute
        for poke in range(nT):
            p = trainSet[poke]
            X[poke] = [1.0, p.stamina, p.attVal, p.defVal, p.capRate, p.fleeRate, p.spChan] + p.primStr

        y = np.empty([nT, 1]) # N by 1 matrix
        for poke in range(nT):
            p = trainSet[poke]
            y[poke] = [p.ptOut]

        # now create x and y matrix for validation set
        nV = len(validSet)
        XX = np.empty([nV, 22])
        for poke in range(nV):
            p = validSet[poke]
            XX[poke] = [1.0, p.stamina, p.attVal, p.defVal, p.capRate, p.fleeRate, p.spChan] + p.primStr

        yy = np.empty([nV, 1])
        for poke in range(nV):
            p = validSet[poke]
            yy[poke] = [p.ptOut]

        if reg_term > 0: # this if statement is for Question (vi)
            w = LR_regular(trainSet, X, y, reg_term)
            sqrtVal.append(computeRSS(validSet, w, XX, yy))
        else:
            w = LR_OLS(trainSet, X, y) # parameters obtained from training set
            sqrtVal.append(computeRSS(validSet, w, XX, yy))

    avgRSS = np.mean(sqrtVal) # average square root of RSS across k folds

# for i in [0, 0.001, 0.01, 0.1, 0.5, 0.888,1, 5, 10]: # experiment with different values for the regularization term
#     cross_validate(encoded_set, 5, i)


# the following code is for Question (vii)

def CVOutAttV(dataset, k): # Combat point against attack value
    random.shuffle(dataset)
    folds = np.array_split(dataset, k)
    sqrtVal = [] 
    for i in range(k):
        trainSet = list()
        for j in range(len(folds)):
            if j == i:
                validSet = folds[j] 
            else:
                for dp in folds[j]:
                    trainSet.append(dp)
        
        nT = len(trainSet)
        X = np.empty([nT, 2]) 
        for poke in range(nT):
            p = trainSet[poke]
            X[poke] = [1.0, p.attVal]

        y = np.empty([nT, 1]) 
        for poke in range(nT):
            p = trainSet[poke]
            y[poke] = [p.ptOut]

        nV = len(validSet)
        XX = np.empty([nV, 2])
        for poke in range(nV):
            p = validSet[poke]
            XX[poke] = [1.0, p.attVal]

        yy = np.empty([nV, 1])
        for poke in range(nV):
            p = validSet[poke]
            yy[poke] = [p.ptOut]

        w = LR_OLS(trainSet, X, y) # parameters obtained from training set
        sqrtVal.append(computeRSS(validSet, w, XX, yy))

    avgRSS = np.mean(sqrtVal) # average square root of RSS across k folds
    print(avgRSS)

# CVOutAttV(encoded_set, 5)

def CVDefVAttV(dataset, k): # Combat point against attack value
    # random.shuffle(dataset)
    folds = np.array_split(dataset, k)
    sqrtVal = [] 
    for i in range(k):
        trainSet = list()
        for j in range(len(folds)):
            if j == i:
                validSet = folds[j] 
            else:
                for dp in folds[j]:
                    trainSet.append(dp)
        
        nT = len(trainSet)
        X = np.empty([nT, 2]) 
        for poke in range(nT):
            p = trainSet[poke]
            X[poke] = [1.0, p.attVal]

        y = np.empty([nT, 1]) 
        for poke in range(nT):
            p = trainSet[poke]
            y[poke] = [p.defVal]

        nV = len(validSet)
        XX = np.empty([nV, 2])
        for poke in range(nV):
            p = validSet[poke]
            XX[poke] = [1.0, p.attVal]

        yy = np.empty([nV, 1])
        for poke in range(nV):
            p = validSet[poke]
            yy[poke] = [p.defVal]

        w = LR_OLS(trainSet, X, y) # parameters obtained from training set
        sqrtVal.append(computeRSS(validSet, w, XX, yy))

    avgRSS = np.mean(sqrtVal) # average square root of RSS across k folds
    print(avgRSS)

# CVDefVAttV(encoded_set, 5)


# the following code is for Question (viii)

def BinOutcome(dataset): 
    combatPts = []
    for poke in dataset:
        combatPts.append(poke.ptOut)

    meanPtOut = np.mean(combatPts)
    combatPts = np.array(combatPts)
    combatPts = combatPts.reshape(1, -1)

    binarizerP = Binarizer(threshold=meanPtOut)
    return binarizerP.fit_transform(combatPts)
 
#  BinComPts = BinOutcome(poke_set)

def logisticR(dataset):
    random.shuffle(dataset)
    X = np.empty([len(dataset), 22])
    for poke, p in enumerate(dataset):    
        X[poke] = [1.0, p.stamina, p.attVal, p.defVal, p.capRate, p.fleeRate, p.spChan] + p.primStr

    y = BinOutcome(dataset)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)

    logReg = LogisticRegression(penalty = 'none', max_iter = 1000) # no regularization
    logReg.fit(X_train, y_train)

    y_pred = logReg.predict(X_test)

#     cnf_matrix = metrics.confusion_matrix(y_test, y_pred) # confusion matrix

# logisticR(encoded_set)


# the following code is for Question (viii)

def LRR(Xtrain, Xtest, ytrain, ytest, reg_term): # perform logistic regression with regularization
    logReg = LogisticRegression(C = 1/reg_term, max_iter = 10000) # penalty = 'l2' by default
    logReg.fit(Xtrain, ytrain)

    ypred = logReg.predict(Xtest)

    cnf_matrix = metrics.confusion_matrix(ytest, ypred) # confusion matrix
    acc = (cnf_matrix[0][0] + cnf_matrix[1][1]) / np.sum(cnf_matrix) # accuracy 
    return acc

def cross_validate(trainX, trainY, k, reg_term):
    foldsX = np.array_split(trainX, k)  
    foldsY = np.array_split(trainY, k) 
    Acc = [] # store the accuracy at every fold

    # now construct training and valid set for each fold
    for i in range(k):
        X = list()
        y = list()
        for j in range(k):
            if j == i:
                XX = foldsX[j] # use one fifth of entire dataset as validation set
                yy = foldsY[j] # use one fifth of entire dataset as validation set
            else:
                for dp in foldsX[j]:
                    X.append(dp)
                for dp in foldsY[j]:
                    y.append(dp)
        Acc.append(LRR(X, XX, y, yy, reg_term))

    return np.mean(Acc) # average performance for current hyperparameter

def logisticR_reg(dataset):
    random.shuffle(dataset)
    X = np.empty([len(dataset), 22])
    for poke in range(len(dataset)):
        p = dataset[poke]
        X[poke] = [1.0, p.stamina, p.attVal, p.defVal, p.capRate, p.fleeRate, p.spChan] + p.primStr

    y = BinOutcome(dataset)[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)

    for i in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 5, 10]: 
        print(cross_validate(X_train, y_train, 5, i), "\t", LRR(X_train, X_test, y_train, y_test, i))
        # printing accuracy obtained from cross-validation for each tested lambda value and final accuracy on test data. 

logisticR_reg(encoded_set) 