import loader
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
import numpy as np
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
# nltk.download('stopwords')    # needed to download this for it to run 

def normalizeLemma(lemma):
    return lemma.lower().replace(" ", "_")

def mergeMultiWords(tokens):
    merged = []
    num = 0
    while num < len(tokens):
        if num < len(tokens) - 1:
            twoWords = tokens[num] + "_" + tokens[num+1]
            if wn.synsets(twoWords):
                merged.append(twoWords)
                num += 2
                continue

        if num < len(tokens) - 2:
            threeWords = tokens[num] + "_" + tokens[num+1] + "_" + tokens[num+2]
            if wn.synsets(threeWords):
                merged.append(threeWords)
                num += 3
                continue

        merged.append(tokens[num])
        num += 1

    return merged


def preprocess (dict) : 
    processedDict = {}
    stop_words = set(stopwords.words('english')) 
    punctuations = [',', '.', '?', '!', ':', '--', '``', '\'\'']

    for i in dict:
        curSentence = dict[i].context
        processedSent = [word.lower() for word in curSentence if ((word.lower() not in stop_words) and (word.lower() not in punctuations))]

        processedSent = mergeMultiWords(processedSent)
        processedDict[i] = processedSent
          
    return processedDict

## 1. The most frequent sense baseline: this is the sense indicated as #1 in the synset according to WordNet

# helper function
def getTopSense (lemma) : 
    lemma = normalizeLemma(lemma)
    synset = wn.synsets(lemma)
    if synset:
        return synset[0]
    return None

# gets the dict with the most frequent sense for each element in the dictionary
def getMostFrequentSenses (dict) :
    senses = {}

    for i in dict : 
        curLemma = normalizeLemma(dict[i].lemma)
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
        curLemma = normalizeLemma(dict[i].lemma)
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
            # if embeddings[word].shape[0] != 50 : # debugging
                # print("\n") # debugging
                # print(word, " : ", embeddings[word]) # debugging
                # print("Shape: ", embeddings[word].shape) # debugging

    if not vectors:  # no words found in embeddings
        return np.zeros(50)
    

    averageVector = np.mean(vectors, axis=0)
    return averageVector


# training the model per lemma
def trainGloveNN (dict, keyDict, filePath) : 
    embeddings = getGlove(filePath)
    preprocessedDict = preprocess(dict)

    xDict = defaultdict(list)
    yDict = defaultdict(list)
    
    for i in dict : 
        curLemma = normalizeLemma(dict[i].lemma)
        curPOS = convertPOS(dict[i].pos)

        # build a context vector by using the GloVe embeddings of the words in the context
        curContext = preprocessedDict[i]
        contextVec = getAvgVector(embeddings, curContext)
        synsetKey = wn.synset_from_sense_key(keyDict[i][0])

        xDict[(curLemma, curPOS)].append(contextVec)
        yDict[(curLemma, curPOS)].append(synsetKey.name())
        
    classifiers = {}
    labelSets = {}

    for lemmaPOS in xDict : 
        x = np.array(xDict[lemmaPOS])
        y = yDict[lemmaPOS]
        
        uniqueLabels = set(y)
        classifier = MLPClassifier(hidden_layer_sizes=(200,), max_iter=700, random_state=42)

        classifier.fit(x, y) 
        classifiers[lemmaPOS] = classifier
        labelSets[lemmaPOS] = list(uniqueLabels)

    return classifiers, labelSets, embeddings 


def predictGloveNN (dict, embeddings, classifiers) : 
    preprocessedDict = preprocess(dict)
    predictionsDict = {}

    for i in dict : 
        curLemma = normalizeLemma(dict[i].lemma)
        curPOS = convertPOS(dict[i].pos)
        lemmaPOS = (curLemma, curPOS)

        curContext = preprocessedDict[i]
        contextVec = getAvgVector(embeddings, curContext)
        contextVec = contextVec.reshape(1, -1)

        synsets = wn.synsets(curLemma, pos=curPOS)

        # if there's a lemma that was not on the training set
        if lemmaPOS not in classifiers:
            predictionsDict[i] = synsets[0]   # use the most used sense (MFS)
            continue

        predictedLabel = classifiers[lemmaPOS].predict(contextVec)[0]
        predictionsDict[i] = wn.synset(predictedLabel)

    return predictionsDict


# 4. Word vector + word definition match

def processSynset (synset) :
    lemmatizer = WordNetLemmatizer()
    definition = synset.definition()
    tokens = definition.lower().split()
    stop_words = set(stopwords.words('english'))
    punctuations = set([',', '.', '?', '!', ':', ';', '--', '``', '\'\''])

    processedDef = []
    for word in tokens:
        if word not in stop_words and word not in punctuations:
            processedDef.append(lemmatizer.lemmatize(word))

    return processedDef


def buildPairFeatures(contextVect, defVect):
    absDiff = np.abs(contextVect - defVect)
    prod = contextVect * defVect
    return np.concatenate([contextVect, defVect, absDiff, prod])


def trainFourthMethod (dict, keyDict, embeddings) : 
    preprocessedDict = preprocess(dict)

    x = []
    y = []

    for i in dict : 
        curLemma = normalizeLemma(dict[i].lemma)
        curPOS = convertPOS(dict[i].pos)
        curContext = preprocessedDict[i]
        contextVect = getAvgVector(embeddings, curContext)

        correctSynsets = set()
        for senseKey in keyDict[i]:
            correctSynsets.add(wn.synset_from_sense_key(senseKey))

        possibleSynsets = wn.synsets(curLemma, pos=curPOS)
        # print(synsets)

        for synset in possibleSynsets:
            processedDef = processSynset(synset)
            defVect = getAvgVector(embeddings, processedDef)

            pairFeatures = buildPairFeatures(contextVect, defVect)
            x.append(pairFeatures)

            if synset in correctSynsets:
                # correct pairings => label = 1
                y.append(1)
            else:
                # incorrect pairings => label = 0
                y.append(0)

    x = np.array(x)
    y = np.array(y)

    classifier = MLPClassifier(hidden_layer_sizes=(300,), max_iter=800, random_state=42)
    classifier.fit(x, y)

    return classifier


def predictFourthMethod (embeddings, classifier, dict) :
    preprocessedDict = preprocess(dict)
    predictions = {}

    for i in dict : 
        curLemma = normalizeLemma(dict[i].lemma)
        curPOS = convertPOS(dict[i].pos)
        curContext = preprocessedDict[i]
        contextVect = getAvgVector(embeddings, curContext)

        possibleSynsets = wn.synsets(curLemma, pos=curPOS)

        bestSynset = None
        bestScore = -float('inf')

        for synset in possibleSynsets:
            processedDef = processSynset(synset)
            defVect = getAvgVector(embeddings, processedDef)

            pairFeatures = buildPairFeatures(contextVect, defVect).reshape(1, -1)
            score = classifier.predict_proba(pairFeatures)[0][1]

            if score > bestScore:
                bestScore = score
                bestSynset = synset

        predictions[i] = bestSynset

    return predictions


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

    # 3. Neural network using context vectors and GloVe
    print("3. Neural network using GloVe")
    classifiers, labelSets, embeddings = trainGloveNN(dev_instances, dev_key, "../wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt")
    predictions = predictGloveNN(test_instances, embeddings, classifiers)
    print("Accuracy (predictions using test_instances):", accuracy(predictions, test_key))

    # 4. Neural network using context and definition matches
    print("4. Neural network using context and definition matches")
    classifier = trainFourthMethod(dev_instances, dev_key, embeddings)
    predictions = predictFourthMethod (embeddings, classifier, test_instances)
    print("Accuracy (predictions using test_instances):", accuracy(predictions, test_key))
