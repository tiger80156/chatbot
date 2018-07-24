from nltk.tokenize import word_tokenize
import pandas as pd
import csv
from gensim.models.word2vec import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from random import randint

# This dataset is the open source dataset, extra the question and anser from the dataset
with open("./WikiQA-train.tsv") as tsvFile:
    reader = csv.DictReader(tsvFile,dialect='excel-tab')

    wordList = []
    previous = ''
    answer = []
    x = []

    for row in reader:
        # The Question column
        questionSentence = row["Question"]

        x.append(row['Sentence'])

        if questionSentence != previous:
            word_cut = word_tokenize(questionSentence)
            wordList.append(word_cut)

            if previous:
                answer.append(x)
                x = []
     
        # Previous thing in the Question Sentence
        previous = questionSentence

# Create the word2vec model
model = Word2Vec(sentences=wordList,size=50,min_count=1,window=4)

# Compute the sentence vector add all word vector in one sentence
sentenceVectorList = []

for sentence in wordList:
    sentenceVector = 0
    for word in sentence:
        sentenceVector += model[word]
        sentenceVector /= len(sentence)
    sentenceVectorList.append(sentenceVector)

    # User input the userUtterance and transfer to vector
userUtterance = input()
userUtterance = word_tokenize(userUtterance)
utteranceVector = 0
mostSimlarity = 0

# Find tha answer
for word in userUtterance:
    utteranceVector += model[word]
    utteranceVector /= len(userUtterance) 

sentenceVectorList.append(utteranceVector)

# Minus the fist principle component
pca = PCA(n_components=1)
reduceDeminsion = pca.fit_transform(sentenceVectorList)
sentenceVectorList -= reduceDeminsion*sentenceVectorList

# Find the Most similar vector
for sentenceVector in sentenceVectorList[:-1]:
    #user's answer is similar to the dataset ==> cosine_similarity method
    similarity = cosine_similarity(utteranceVector.reshape(1,-1), sentenceVector.reshape(1,-1)) 

    if  similarity > mostSimlarity:
        mostSimlarity = similarity
        mostSimlaritySentence = sentenceVector

print(mostSimlarity)# The score of the Simlarity
print(mostSimlaritySentence)# The most Simlarity of the vector of the Sentence 


for i in range(len(sentenceVectorList)):
    if np.array_equal(mostSimlaritySentence, sentenceVectorList[i]):
        # print(mostSimlaritySentence, sentenceVectorList[i])
        rni = randint(0,len(answer[i]))
        print(" ".join(wordList[i]))
        print(answer[i][rni])
        break
