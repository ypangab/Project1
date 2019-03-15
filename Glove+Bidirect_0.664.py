
import numpy as np
import string
import pandas as pd
import nltk
import keras
import pickle 
#save tokenized text



from sklearn import random_projection
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional
from keras.optimizers import SGD, Adam
from keras import metrics

#remove stopwords and punctuation
stop_words = set(stopwords.words('english') + list(string.punctuation))


# -------------- Helper Functions --------------
def tokenize(text):
    '''
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    https://textminingonline.com/dive-into-nltk-part-ii-sentence-tokenize-and-word-tokenize
    e.g.
    Input: 'It is a nice day. I am happy.'
    Output: ['it', 'is', 'a', 'nice', 'day', 'i', 'am', 'happy']
    '''
    tokens = []
    text = text.replace('-', ' ')
    for word in nltk.word_tokenize(text):
        word = word.lower()
        if word not in stop_words and not word.isnumeric() and word.isalpha():
            tokens.append(word)
            '''steemed_word = porter.stem(word)
            tokens.append(steemed_word)
            print (steemed_word)'''
    return tokens



def get_sequence(data, seq_length, vocab_dict):
    '''
    :param data: a list of words, type: list
    :param seq_length: the length of sequences,, type: int
    :param vocab_dict: a dict from words to indices, type: dict
    return a dense sequence matrix whose elements are indices of words,
    '''
    data_matrix = np.zeros((len(data), seq_length), dtype=int)
    for i, doc in enumerate(data):
        for j, word in enumerate(doc):
            # YOUR CODE HERE
            if j == seq_length:
                break
            word_idx = vocab_dict.get(word, 1) # 1 means the unknown word
            data_matrix[i, j] = word_idx
    return data_matrix


def read_data(file_name, input_length, vocab=None):
    """
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    """
    df = pd.read_csv(file_name)
    for col in df:
        print ('no. of unique values in {}: {}'.format(col,df[col].nunique()))

    #get averaage length of review
    f = file_name[5:-4]
    print (f)
    df['words'] = pd.read_pickle(f+"_tokenized_df.pkl")
    print ('average length of review: ', df['words'].apply(len).mean())
    '''df['words'] = df['text'].apply(tokenize)
    df['words'].to_pickle("./tokenized_df1"+fi".pkl")'''
    '''with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(df['words'], handle, protocol=pickle.HIGHEST_PROTOCOL)'''



    print ('finish clean + tokenizine: ', file_name)
    if vocab is None:
        vocab = set()
        for i in range(len(df)):
            for word in df.iloc[i]['words']:
                vocab.add(word)
    vocab_dict = dict()
    vocab_dict['<pad>'] = 0 # 0 means the padding signal
    vocab_dict['<unk>'] = 1 # 1 means the unknown word
    vocab_size = 2
    for v in vocab:
        vocab_dict[v] = vocab_size
        vocab_size += 1

    #print ('var_vocab: ',vocab)
    #print (vocab_dict)
    data_matrix = get_sequence(df['words'], input_length, vocab_dict)
    #print ('var_datya_matrix: ',data_matrix)
    stars = df['stars'].apply(int) - 1
    return df['review_id'], stars, data_matrix, vocab, vocab_dict###
# ----------------- End of Helper Functions-----------------


def load_data(input_length):
    # Load training data and vocab
    train_id_list, train_data_label, train_data_matrix, vocab, vocab_dict = read_data("data/train.csv", input_length)
    K = max(train_data_label)+1  # labels begin with 0

    ###
    sequence_dict = vocab_dict
    
    # Load valid data
    valid_id_list, valid_data_label, valid_data_matrix, vocab, vocab_dict = read_data("data/valid.csv", input_length, vocab=vocab)

    # Load testing data
    test_id_list, _, test_data_matrix, _, _= read_data("data/test.csv", input_length, vocab=vocab)
    
    print("Vocabulary Size:", len(vocab))
    print("Training Set Size:", len(train_id_list))
    print("Validation Set Size:", len(valid_id_list))
    print("Test Set Size:", len(test_id_list))
    print("Training Set Shape:", train_data_matrix.shape)
    print("Validation Set Shape:", valid_data_matrix.shape)
    print("Testing Set Shape:", test_data_matrix.shape)

    # Converts a class vector to binary class matrix.
    # https://keras.io/utils/#to_categorical
    train_data_label = keras.utils.to_categorical(train_data_label, num_classes=K)
    valid_data_label = keras.utils.to_categorical(valid_data_label, num_classes=K)
    return train_id_list, train_data_matrix, train_data_label, \
        valid_id_list, valid_data_matrix, valid_data_label, \
        test_id_list, test_data_matrix, None, vocab, sequence_dict


if __name__ == '__main__':
    #Start
    print ('hihi')
    # Hyperparameters
    input_length = 100
    embedding_size = 100
    hidden_size = 70
    batch_size = 128
    dropout_rate = 0.4
    learning_rate = 0.007
    total_epoch = 10 
    trainable = False

    train_id_list, train_data_matrix, train_data_label, \
        valid_id_list, valid_data_matrix, valid_data_label, \
        test_id_list, test_data_matrix, _, vocab, sequence_dict = load_data(input_length)

    # Data shape
    N = train_data_matrix.shape[0]
    K = train_data_label.shape[1]

    input_size = len(vocab) + 2
    output_size = K

    # New model
    model = Sequential()

    # embedding layer and dropout
    # YOUR CODE HERE
    max_cap = input_length
    embeddings_index = dict();
    with open('data/glove.6B.100d.txt') as f:
        for line in f:
            values = line.split();
            word = values[0];
            coefs = np.asarray(values[1:], dtype='float32');
            embeddings_index[word] = coefs;
            
    vocab_size = len(sequence_dict);
    embeddings_matrix = np.zeros((vocab_size, embedding_size));
    for word, i in sequence_dict.items():
        embedding_vector = embeddings_index.get(word);
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector;


    print ('Using Glove embeddings')
    model.add(Embedding(len(sequence_dict), output_dim=embedding_size, input_length=max_cap, weights=[embeddings_matrix], trainable=trainable));
    #model.add(Embedding(input_dim=input_size, output_dim=embedding_size, input_length=input_length))
    model.add(Dropout(dropout_rate))

    # LSTM layer
    # YOUR CODE HERE
    model.add(Bidirectional(LSTM(units=hidden_size,return_sequences=True, recurrent_dropout=dropout_rate)))
    model.add(LSTM(units=hidden_size, recurrent_dropout=dropout_rate))
    

    # output layer
    # YOUR CODE HERE

    model.add(Dense(K, activation='softmax'))

    # SGD optimizer with momentum
    #optimizer = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)

    optimizer = Adam(lr=learning_rate, decay=0.001);

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # mode summary
    print(model.summary())

   



   # training
    model.fit(train_data_matrix, train_data_label, epochs=total_epoch, batch_size=batch_size)
    # testing
    train_score = model.evaluate(train_data_matrix, train_data_label, batch_size=batch_size)
    print('Training Loss: {}\n Training Accuracy: {}\n'.format(train_score[0], train_score[1]))
    valid_score = model.evaluate(valid_data_matrix, valid_data_label, batch_size=batch_size)
    print('Validation Loss: {}\n Validation Accuracy: {}\n'.format(valid_score[0], valid_score[1]))

    # predicting
    test_pre = model.predict(test_data_matrix, batch_size=batch_size).argmax(axis=-1) + 1
    sub_df = pd.DataFrame()
    sub_df["review_id"] = test_id_list
    sub_df["pre"] = test_pre
    sub_df.to_csv("pre.csv", index=False)
