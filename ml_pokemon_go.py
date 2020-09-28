import numpy as np
import matplotlib.pyplot as plt
import math

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
        return "Stamina: {}, Attack Value: {}, Defense Value: {}, Capture Rate: {}, Flee Rate: {}, Spawn Chance: {}, Primary Strength: {}, Combat Point Outcome".format(self.stamina, self.attVal, self.defVal, self.capRate, self.fleeRate,self.spChan, self.primStr, self.ptOut)
        
# the following code is for Question (i)

def parse_dataset(filename):
    data_file = open(filename, 'r')  # Open File "to read"
    dataset = []  # List to hold Datapoint objects

    for index, line in enumerate(data_file):
        if index == 0:  # First line describes the datapoint, it's not an actual datapoint
            continue  # do nothing, it will skip all the following code
        name, stamina, attVal, defVal, capRate, fleeRate, spChan, primStr, ptOut = line.strip().split(',')  # strip() removes '\n', and split(',') splits the line at tabs
        dataset.append(DataPoint({'stamina':float(stamina), 'attack_value':float(attVal), 'defense_value':float(defVal), 'capture_rate':float(capRate), 'flee_rate':float(fleeRate), 'spawn_chance':float(spChan), 'primary_strength':primStr, 'combat_point_outcome':float(ptOut)}))  # Create DataPoint object for the given data

    print("Total Number of Data Points: {0}".format(len(dataset)))
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
print(encoded_set[0])

# the following code is for Question (v)
