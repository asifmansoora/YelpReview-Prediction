import re
from itertools import ifilterfalse
from stemming.porter2 import stem
from nltk.stem import WordNetLemmatizer
import nltk


####################################################################################################################
###         Clean reviews
###
####################################################################################################################


#tokens: List of words
def getPOSList(tokens,Noun=False,Adj = False):
           
        tagged = nltk.pos_tag(tokens)
        #print tagged
        POSList = []
        NounList = []
        AdjList =[]
        # to get all the words that are 'NN','NNP','NNS'
        for i in tagged:
                
                if Noun and i[1][0]=='N':
                        NounList.append(i[0])
                if Adj and i[1][0]=='J':
                        AdjList.append(i[0])
                
                        
        POSList.append(NounList)
        POSList.append(AdjList)
	return POSList


def clean_review(line):
        line = line.lower()
        
        #remove all the characters in string except number, alpha, space and '
        line = re.sub(r'[^a-zA-Z0-9, ,\']', " ", line)
        tokens = nltk.word_tokenize(line)

        
        return tokens
        
       
#Stemming of the word List by using porter stemmer
def word_stemming(wordList):

        
	for i,word in enumerate(wordList):
		wordList[i] =stem(word)
	
	return wordList

def filterListofWords(wordList):
        # It is observed that all the words of length less than 2 are not useful.
        # So we are removing all the words that are less than 2
        
        wordList[:] = ifilterfalse(lambda i: (len(i)<3 ) , wordList)
        return wordList

def removeStopwords(wordList):
        #remove the stop words from the NLTK Stop words list
        stopwords = nltk.corpus.stopwords.words('english')
        wordList[:] = ifilterfalse(lambda i: (i in stopwords) , wordList)
        return wordList
        
#Lemmantiation of the word List
def word_Lemmantization(wordList,wordNetLemma):

        
	for i,word in enumerate(wordList):
		wordList[i] =wordNetLemma.lemmatize(word)
	
	return wordList

        
