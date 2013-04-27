import logging
import numpy
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os.path
from gensim import corpora, models, similarities
import gensim


class MyCorpus(object):
    def __iter__(self):
        for line in open('reviews.txt'):
            yield dictionary.doc2bow(line.lower().split())


def CreateDictionary(texts):
    #texts: List of List of words 
    print "\t Creating Dictionary of words and saving on disk"
     # remove words that appear only once
    all_tokens = sum(texts, [])
    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
    texts = [[word for word in text if word not in tokens_once]
             for text in texts]

    dictionary = corpora.Dictionary(texts)
    dictionary.save('deerwester.dict')
    return dictionary


#To actually convert tokenized documents to vectors:    
def CreateDocVector(texts,VectorfileName):

    print "\t Creating Document Vectors and saving on disk"
    #if Dictionary and Corpus doesn't exist, we will create one
    #else we will upload from the disk
    if not (os.path.exists('deerwester.dict')):
        dictionary = CreateDictionary(texts)
    else:
        #Load the Dictionary
        
        dictionary = corpora.Dictionary.load('deerwester.dict')
        
    # Get the Text and covert it into Feature
    corpus = [dictionary.doc2bow(text) for text in texts]
    #Store doc feature into memory
    VectorfileName = VectorfileName+'.mm'
    corpora.MmCorpus.serialize(VectorfileName, corpus)
    return corpus



############################################################
#Topics and Transformations
#############################################################
def TransformFeatureDoc(texts,VectorfileName,num_topics=100):

    #if Dictionary and Corpus doesn't exist, we will create one
    #else we will upload from the disk
    #if not (os.path.exists('C:\Users\Rajnish Kumar Garg\Documents\GitHub\YelpReview-Prediction\Code\deerwester.dict')):
    
    if not (os.path.exists('deerwester.dict')):
        dictionary = CreateDictionary(texts)
    else:
        #Load the Dictionary
        dictionary = corpora.Dictionary.load('deerwester.dict')

    #Load the Document Vector
    VectorfileName = VectorfileName+'.mm'
    
    if not (os.path.exists(VectorfileName)):
        corpus = CreateDocVector(texts,VectorfileName)
        
    else:
        #Load the Corpus
        corpus = corpora.MmCorpus(VectorfileName)

    #print corpus
    #converting to TF-IDF
    tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
    
    corpus_tfidf = tfidf[corpus]
    print "\t Number of topic are: ",num_topics
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=100)
    #lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
    corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    return corpus_lda
    

def getDocumentFeatures(TopicModel_data, num_topics=100):
    
    #To convert the Masked LDA or LSA Transformed featured to Numpy format
    
    #First save all values into local variable
    DocData = []
    for doc in TopicModel_data:
        DocData.append(doc)

    #Convert from gensim format to numpy (To add data into features)
    numpy_matrix = gensim.matutils.corpus2dense(DocData,num_topics)
    return numpy_matrix.T
        







