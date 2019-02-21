#Please use python 3.5 or above
import keras
from keras.engine.topology import Layer
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, GRU, Flatten, BatchNormalization
from keras import optimizers
from keras.models import load_model
import json, argparse, os
import re
import io
import sys
from keras.layers.convolutional import Conv1D
from keras.layers import GRU, LSTM
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Dense ,LSTM,concatenate,Input,Flatten
from keras.models import Model
import pandas as pd
from keras import initializers, regularizers, constraints
from keras import backend as K
from keras.engine.topology import Layer
# Path to training and testing data file. This data can be downloaded from a link, details of which will be provided.
trainDataPath = ""
testDataPath = ""
# Output file that will be generated. This file can be directly submitted.
solutionPath = ""
# Path to directory where GloVe file is saved.
gloveDir = ""

NUM_FOLDS = None                   # Value of K in K-fold Cross Validation
NUM_CLASSES = None                 # Number of classes - Happy, Sad, Angry, Others
MAX_NB_WORDS = None                # To set the upper limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer 
MAX_SEQUENCE_LENGTH = None         # All sentences having lesser number of words than this will be padded
EMBEDDING_DIM = None               # The dimension of the word embeddings
BATCH_SIZE = None                  # The batch size to be chosen for training the model.
LSTM_DIM = None                    # The dimension of the representations learnt by the LSTM model
DROPOUT = None                     # Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
NUM_EPOCHS = None                  # Number of epochs to train a model for


vector_file = '../../vectors/fasttext/fasttext.wiki.en.vec'
vector_file = '../../vectors/lexvec/lexvec.commoncrawl.300d.W.pos.vectors'
vector_file = '../../vectors/deps/deps.words'
vector_file = 'glove.840B.300d.txt'
#vector_file = '/data/glove.twitter.27B/glove.twitter.27B.25d.txt'
#vector_file = 'wiki-news-300d-1M.vec'
#vector_file = 'crawl-300d-2M-subword.vec'
EMBEDDING_DIM = 300
print("Will use vectors from", vector_file)


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}



class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

        
        
        


def preprocessData(dataFilePath, mode):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    indices = []
    conversations = []
    posconversations = []
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '    
                line = cSpace.join(lineSplit)
            
            line = line.strip().split('\t')
            #print(line)
            if mode == "train" or 1==1:
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                labels.append(label)
            
            conv = ' <eos> '.join(line[1:4])
            posconv = ' <eos> '.join(line[-3:])
            # Remove any duplicate spaces
            duplicateSpacePattern = re.compile(r'\ +')    
            conv = re.sub(duplicateSpacePattern, ' ', conv)
            
            indices.append(int(line[0]))
            conversations.append(conv.lower())
            posconversations.append(posconv.upper())
    if mode == "train" or 1==1:
        return indices, conversations, labels,posconversations
    else:
        return indices, conversations


def getMetrics(predictions, ground):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """

    discretePredictions = to_categorical(predictions.argmax(axis=1),num_classes=4)
    print(discretePredictions[0])
    print(discretePredictions.shape)
    print(ground.shape) 
    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
    
    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)
    
    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    
    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))   
    
    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()    
    
    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))
    
    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)
    
    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------
    
    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)
    
    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


def writeNormalisedData(dataFilePath, texts):
    """Write normalised data to a file
    Input:
        dataFilePath : Path to original train/test file that has been processed
        texts : List containing the normalised 3 turn conversations, separated by the <eos> tag.
    """
    normalisedDataFilePath = dataFilePath.replace(".txt", "_normalised.txt")
    with io.open(normalisedDataFilePath, 'w', encoding='utf8') as fout:
        with io.open(dataFilePath, encoding='utf8') as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                line = line.strip().split('\t')
                normalisedLine = texts[lineNum].strip().split('<eos>')
                fout.write(line[0] + '\t')
                # Write the original turn, followed by the normalised version of the same turn
                fout.write(line[1] + '\t' + normalisedLine[0] + '\t')
                fout.write(line[2] + '\t' + normalisedLine[1] + '\t')
                fout.write(line[3] + '\t' + normalisedLine[2] + '\t')
                try:
                    # If label information available (train time)
                    fout.write(line[4] + '\n')    
                except:
                    # If label information not available (test time)
                    fout.write('\n')


def getEmbeddingMatrix(wordIndex):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    #with io.open(os.path.join(gloveDir, 'glove.6B.100d.txt'), encoding="utf8") as f:
    #with io.open(os.path.join(gloveDir, 'glove.840B.300d.txt'), encoding="utf8") as f:
    with io.open(vector_file, encoding="utf8") as f:
        for line in f:
            #values = line.split(' ')  #Works for glove
            values = line.rstrip().rsplit(' ') #Works for fasttext
            word = values[0]
            try:
                embeddingVector = np.asarray(values[1:], dtype='float32')
            except:
                print("Line=", line)
                print("Values=", values)
                sys.exit()
            embeddingsIndex[word] = embeddingVector
    
    print('Found %s word vectors.' % len(embeddingsIndex))
    
    # Minimum word index of any word is 1. 
    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
    
    return embeddingMatrix
            

def getEmotionMatrix(wordIndex):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    #with io.open(os.path.join(gloveDir, 'glove.6B.100d.txt'), encoding="utf8") as f:
    #with io.open(os.path.join(gloveDir, 'glove.840B.300d.txt'), encoding="utf8") as f:
    with io.open('./DepecheMood++/DepecheMood_english_token_full.tsv', encoding="utf8") as f:
        for line in f:
            #values = line.split(' ')  #Works for glove
            values = line.rstrip().rsplit(' ') #Works for fasttext
            word = values[0]
            try:
                embeddingVector = np.asarray(values[1:], dtype='float32')
            except:
                print("Line=", line)
                print("Values=", values)
                sys.exit()
            embeddingsIndex[word] = embeddingVector
    
    print('Found %s word vectors.' % len(embeddingsIndex))
    
    # Minimum word index of any word is 1. 
    embeddingMatrix = np.zeros((len(wordIndex) + 1, 9))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector
    
    return embeddingMatrix

    
def buildModel(embeddingMatrix,posEmbeddingMatrix,emotionEmbeddingMatrix):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    
    posEmbeddingLayer = Embedding(posEmbeddingMatrix.shape[0],
                                  posEmbeddingMatrix.shape[1],
                                  weights =[posEmbeddingMatrix],
                                  input_length=MAX_SEQUENCE_LENGTH,
                                  trainable=False)
                                  
    
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
                                
    emotionEmbeddingLayer = Embedding(emotionEmbeddingMatrix.shape[0],
                                  emotionEmbeddingMatrix.shape[1],
                                  weights =[emotionEmbeddingMatrix],
                                  input_length=MAX_SEQUENCE_LENGTH,
                                  trainable=False)
                                  
    lstm1 = Bidirectional(LSTM(128, dropout=DROPOUT,return_sequences=True))
    lstm2 = Bidirectional(LSTM(128, dropout=DROPOUT,return_sequences=True))
    lstm3 = Bidirectional(LSTM(32, dropout=DROPOUT,return_sequences=True))
    lstm4 = Bidirectional(LSTM(32, dropout=DROPOUT))
    lstm5 = Bidirectional(LSTM(32, dropout=DROPOUT,return_sequences=True))
    lstm6 = Bidirectional(LSTM(32, dropout=DROPOUT))
    seq1_inp = Input(shape=(MAX_SEQUENCE_LENGTH,),dtype='int32')
    emb_seq1 = embeddingLayer(seq1_inp)
    
    seq2_inp = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    emb_seq2 = posEmbeddingLayer(seq2_inp)
    
    seq3_inp = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    emb_seq3 = emotionEmbeddingLayer(seq3_inp)
    #merged = keras.layers.concatenate([emb_seq1,emb_seq2,emb_seq3])
    #merged = keras.layers.concatenate([emb_seq1,emb_seq2])
    #merged = lstm1(merged)
    merged = emb_seq1
    merged = lstm1(merged)
    merged = lstm2(merged)
    merged = Attention(MAX_SEQUENCE_LENGTH)(merged)
    
    preds = Dense(NUM_CLASSES, activation='sigmoid')(merged)
    
    model = Model(inputs=[seq1_inp,seq2_inp,seq3_inp],output=preds)

    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model
    
def getPosEmbeddingMatrix(posIndex):
    """Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingMatrix = {}
    embeddingMatrix = np.zeros((len(posIndex) + 1, len(posIndex)+1))
    for word, i in posIndex.items():
        if word!= '<eos>':
            embeddingMatrix[i][i]=1
        
        
    return embeddingMatrix
        

def main():
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument('-config', help='Config to read details', required=True)
    args = parser.parse_args()
    with open(args.config) as configfile:
        config = json.load(configfile)
        
    global trainDataPath, testDataPath, solutionPath, gloveDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE    

    trainDataPath = config["train_data_path"]
    testDataPath = config["test_data_path"]
    solutionPath = config["solution_path"]
    gloveDir = config["glove_dir"]
    
    NUM_FOLDS = config["num_folds"]
    NUM_CLASSES = config["num_classes"]
    MAX_NB_WORDS = config["max_nb_words"]
    MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
    #EMBEDDING_DIM = config["embedding_dim"]
    BATCH_SIZE = config["batch_size"]
    LSTM_DIM = config["lstm_dim"]
    DROPOUT = config["dropout"]
    LEARNING_RATE = config["learning_rate"]
    NUM_EPOCHS = config["num_epochs"]
        
    print("Processing training data...")
    trainIndices, trainTexts, labels,posTrainTexts = preprocessData(trainDataPath, mode="train")
    # Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable   
    # writeNormalisedData(trainDataPath, trainTexts)
    print("Processing test data...")
    testIndices, testTexts , testlabels, posTestTexts = preprocessData(testDataPath, mode="test")
    # writeNormalisedData(testDataPath, testTexts)

    print("Extracting tokens...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(trainTexts)
    trainSequences = tokenizer.texts_to_sequences(trainTexts)
    testSequences = tokenizer.texts_to_sequences(testTexts)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating embedding matrix...")
    embeddingMatrix = getEmbeddingMatrix(wordIndex)

    data = pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    
    
    
    val=['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB','<eos>']
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(val)
    posIndex = tokenizer.word_index
    posEmbeddingMatrix = getPosEmbeddingMatrix(posIndex)
    #print(posTrainTexts)
    
    posTrainTextsSequences = tokenizer.texts_to_sequences(posTrainTexts)
    posTestTextsSequences = tokenizer.texts_to_sequences(posTestTexts)
    
    #print(posTrainTextsSequences[0],posTrainTextsSequences[1])
    #print(posTestTextsSequences[0],posTestTextsSequences[1])
    
    
    posTrainData = pad_sequences(posTrainTextsSequences,maxlen = MAX_SEQUENCE_LENGTH)
    posTestData = pad_sequences(posTestTextsSequences,maxlen = MAX_SEQUENCE_LENGTH)
    
    
    
    emotionEmbeddingMatrix = getEmotionMatrix(wordIndex)
    #print(posTrainTexts)
    
    
    #print(posTrainTextsSequences[0],posTrainTextsSequences[1])
    #print(posTestTextsSequences[0],posTestTextsSequences[1])
    
    labels = to_categorical(np.asarray(labels))
    testlabels = to_categorical(np.asarray(testlabels))
    print("Shape of training data tensor: ", data.shape)
    print("Shape of label tensor: ", labels.shape)
    
    
    # Randomize data
    np.random.shuffle(trainIndices)
    data = data[trainIndices]
    labels = labels[trainIndices]
      
    # Perform k-fold cross validation
    metrics = {"accuracy" : [],
               "microPrecision" : [],
               "microRecall" : [],
               "microF1" : []}
    
    print("Starting k-fold cross validation...")
    for k in range(NUM_FOLDS):
        print('-'*40)
        print("Fold %d/%d" % (k+1, NUM_FOLDS))
        validationSize = int(len(data)/NUM_FOLDS)
        index1 = validationSize * k
        index2 = validationSize * (k+1)
            
        xTrain = np.vstack((data[:index1],data[index2:]))
        yTrain = np.vstack((labels[:index1],labels[index2:]))
        xTrainPos = np.vstack((posTrainData[:index1],posTrainData[index2:]))
        xVal = data[index1:index2]
        xValPos = posTrainData[index1:index2]
        yVal = labels[index1:index2]
        print("Building model...")
        model = buildModel(embeddingMatrix,posEmbeddingMatrix,emotionEmbeddingMatrix)
        model.fit([xTrain,xTrainPos,xTrain], yTrain, 
                  validation_data=([xVal,xValPos,xVal], yVal),
                  epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

        predictions = model.predict([xVal,xValPos,xVal], batch_size=BATCH_SIZE)
        accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, yVal)
        metrics["accuracy"].append(accuracy)
        metrics["microPrecision"].append(microPrecision)
        metrics["microRecall"].append(microRecall)
        metrics["microF1"].append(microF1)
        
    print("\n============= Metrics =================")
    print("Average Cross-Validation Accuracy : %.4f" % (sum(metrics["accuracy"])/len(metrics["accuracy"])))
    print("Average Cross-Validation Micro Precision : %.4f" % (sum(metrics["microPrecision"])/len(metrics["microPrecision"])))
    print("Average Cross-Validation Micro Recall : %.4f" % (sum(metrics["microRecall"])/len(metrics["microRecall"])))
    print("Average Cross-Validation Micro F1 : %.4f" % (sum(metrics["microF1"])/len(metrics["microF1"])))
    
    print("\n======================================")
    print("Retraining model on entire data to create solution file")
    model = buildModel(embeddingMatrix,posEmbeddingMatrix,emotionEmbeddingMatrix)
    model.fit([data,posTrainData,data], labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    
    model.save('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))
    #model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))

    print("Creating solution file...")
    testData = pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict([testData,posTestData,testData], batch_size=BATCH_SIZE)
    accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions,testlabels)
    print("Accuracy " + accuracy + "microPrecision "+ microPrecision + "microRecall" + microRecall + "microF1" + microF1)
    predictions = predictions.argmax(axis=1)

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')        
        with io.open('./devwithoutlabels.txt', encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')
    print("Completed. Model parameters: ")
    print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d" 
          % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))
    
               
if __name__ == '__main__':
    main()
