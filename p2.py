import loader
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords 
import numpy as np
# nltk.download('stopwords')    # needed to download this one 

def preprocess (dict) : 
    # tokenization
    # lemmatization
    # multi-word phrases --> put underscores instead of spaces
    # remove stop words

    processedDict = {}
    stop_words = set(stopwords.words('english')) 
    punctuations = [',', '.', '?', '!', ':', '--', '``', '\'\'']

    for i in dict : 
        curSentence = dict[i].context
        processedSent = [word.lower() for word in curSentence if ((word.lower() not in stop_words) and (word.lower() not in punctuations))]
        processedDict[i] = processedSent
          
    return processedDict

## 1. The most frequent sense baseline: this is the sense indicated as #1 in the synset according to WordNet

# helper function
def getTopSense (lemma) : 
    synset = wn.synsets(lemma)
    if synset : 
        return synset[0]

    return None

# gets the dict with the most frequent sense for each element in the dictionary
def getMostFrequentSenses (dict) :
    senses = {}

    for i in dict : 
        curLemma = dict[i].lemma
        curTopSense = getTopSense(curLemma)
        senses[i] = curTopSense
    
    return senses


# calculates the accuracy between the results obtained and the key dict
def accuracy (myDict, keyDict) : 
    correctCount = 0
    totalCount = 0

    for i in myDict : 
        totalCount = totalCount + 1
        mySense = myDict[i]
        # print('\n', mySense)
        # print(keyDict[i])
        
        possibleKeys = []

        for j in range(len(keyDict[i])) : 
            curSynsetKey = wn.synset_from_sense_key(keyDict[i][j])
            possibleKeys.append(curSynsetKey)

        # print(possibleKeys)

        for sense in possibleKeys : # scans all possible keys 
            if mySense == sense : 
                correctCount = correctCount + 1 # adds one to the count if there's a definition that works
                # print('match!!!')

    return correctCount / totalCount

# meant to test the accuracy using the test set
def testMostFrequent (dict, keyDict) :
    myDict = getMostFrequentSenses(dict)
    return accuracy(myDict, keyDict)
    

## 2. NLTK’s implementation of Lesk’s algorithm (nltk.wsd.lesk)

def leskAlgorithm (dict) : 
    preprocessedDict = preprocess(dict)
    resultDict = {}

    for i in dict : 
        curLemma = dict[i].lemma
        sentence = preprocessedDict[i]
        curSynset = nltk.wsd.lesk(sentence, curLemma)
        resultDict[i] = curSynset

    return resultDict

def testLesk (dict, keyDict) :
    myDict = leskAlgorithm(dict)
    return accuracy(myDict, keyDict)


# 3. Neural network using pretrained word embeddings such as GloVe

def convertPOS(treebank_pos):
    if treebank_pos.startswith('N'):
        return wn.NOUN
    elif treebank_pos.startswith('V'):
        return wn.VERB
    elif treebank_pos.startswith('J'):
        return wn.ADJ
    elif treebank_pos.startswith('R'):
        return wn.ADV
    return None 


def getGlove (filePath) : 
    embeddings = {}
    stop_words = set(stopwords.words('english')) 
    punctuations = [',', '.', '?', '!', ':', '--', '``', '\'\'']

    with open (filePath, 'r', encoding='utf-8') as file : 
        for line in file : 
            tokens = line.split()
            lemma = tokens[0]

            if (lemma in stop_words) or (lemma in punctuations) : 
                continue

            # print(lemma)
            try:
                vector = np.array(tokens[1:], dtype='float32')
                # print(vector.shape) # debugging
                if vector.shape[0] != 50 : # debugging
                    continue
                embeddings[lemma] = vector
            except ValueError:
                continue 

    return embeddings


def getAvgVector (embeddings, sentence) :
    vectors = []

    for word in sentence : 
        if word in embeddings : 
            vectors.append(embeddings[word])
            if embeddings[word].shape[0] != 50 : # debugging
                print("\n") # debugging
                print(word, " : ", embeddings[word]) # debugging
                print("Shape: ", embeddings[word].shape) # debugging

    if not vectors:  # no words found in embeddings
        return np.zeros(50)
    

    averageVector = np.mean(vectors, axis=0)
    return averageVector


def getCosineSim (v1, v2) :
    if (np.linalg.norm(v1) == 0) or (np.linalg.norm(v2) == 0) :
        return 0

    cosSim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cosSim


def gloveMethod (dict, filePath) :
    embeddings = getGlove(filePath)
    preprocessedDict = preprocess(dict)

    resultsDict = {}

    for i in dict : 
        # getting the average vector of the context sentence
        curContext = preprocessedDict[i]
        curLemma = dict[i].lemma
        curPOS = convertPOS(dict[i].pos)
        avgContextVec = getAvgVector(embeddings, curContext)

        # for all the definition sentences
        synsets = wn.synsets(curLemma, pos=curPOS)
        bestMatch = None
        bestSimilitude = -float('inf')

        # get the average vector
        for curSynset in synsets : 
            #get definition into a vector 
            curDefinition = curSynset.definition().lower().split()
            curExamples = curSynset.examples().lower().split()  
            curDef = curDefinition + curExamples

            avgDef = getAvgVector(embeddings, curDef)
            curSimilitude = getCosineSim(avgContextVec, avgDef)

            # select the vector that's closest to the context sentence 
            if (curSimilitude > bestSimilitude) :
                bestMatch = curSynset
                bestSimilitude = curSimilitude
        
        resultsDict[i] = bestMatch

    return resultsDict


def testGloveMethod (myDict, keyDict) : 
    myDict = gloveMethod(myDict, '../wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt')
    return accuracy(myDict, keyDict)


# 4. xyz
def fourthMethod () : 
    return

 
if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = loader.load_instances(data_f)
    dev_key, test_key = loader.load_key(key_f)
    
    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k:v for (k,v) in dev_instances.items() if k in dev_key}
    test_instances = {k:v for (k,v) in test_instances.items() if k in test_key}
    
    # print out all the dev instances
    # for i in dev_instances:
    #     print (i, " : ", dev_instances[i])
        # print(dev_instances[i].context)

    # print out all the preprocessed sentences
    preprocessedDev = preprocess(dev_instances)
    # for i in preprocessedDev:
    #     print (i, " : ", preprocessedDev[i])

    # 1. The most frequent sense baseline: this is the sense indicated as #1 in the synset according to WordNet
    print("1. The most frequent sense baseline :")
    print("Accuracy (dev_instances and dev_key) : ", testMostFrequent(dev_instances, dev_key))
    print("Accuracy (test_instances and test_key) : ", testMostFrequent(test_instances, test_key))

    # 2. NLTK’s implementation of Lesk’s algorithm (nltk.wsd.lesk)
    print("2. NLTK's implementation of Lesk's algorithm (nltk.wsd.lesk)")
    print("Accuracy (dev_instances and dev_key) : ", testLesk(dev_instances, dev_key))
    print("Accuracy (test_instances and test_key) : ", testLesk(test_instances, test_key))

    # 3. Pre-trained neural network using GloVe
    print("3. Pre-trained neural network using GloVe")
    print("Accuracy (dev_instances and dev_key) : ", testGloveMethod(dev_instances, dev_key))
    print("Accuracy (test_instances and test_key) : ", testGloveMethod(test_instances, test_key))
