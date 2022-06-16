import numpy as np
import pandas as pd
import tokenize
from gensim.models import KeyedVectors


def separate(token):
    '''
    Separates string token of the format 'index+word+label' into a list of strings of the format ['index', 'word', 'label'].
    '''

    index = ''
    word = ''

    split_token = token.split('##')
    
    for i in split_token[0]:
        if i.isdigit():
            index += i
        else:
            word += i
    
    word = word.replace('\t','')
    
    label = split_token[-1]

    return [index, word, label]


def read_source(filepath):
    '''
    Reads a 'cdtb.tok' file into an array of tokens split into ['index', 'word', 'label'].
    '''
    
    with tokenize.open(filepath) as f:
        tokens = tokenize.generate_tokens(f.readline)

        string = ''
        for token in tokens:
            string += token.string

        string = string.replace('_______', '##').split('\n')

        array = []
        for token in string:
            split_token = separate(token)
            
            if split_token[1] != '# newdoc id = chtb_':
                array.append(split_token)

    return np.array(array)


def generate_embedding(emb_filepath, raw_tokens):
    '''
    Generates a 200-dimension vector embedding for the Chinese word in each token.

    Embedding obtained from Tencent AI Lab Embedding Corpus for Chinese Words and Phrases.
    https://ai.tencent.com/ailab/nlp/en/embedding.html
    '''

    model = KeyedVectors.load_word2vec_format(emb_filepath, binary=False)

    embedded_tokens = []

    for token in raw_tokens:
        if token[1] in model.index_to_key:
            token = np.append(token, model[token[1]])
        else:
            token = np.append(token, np.zeros(200, dtype=int))
        
        embedded_tokens.append(token)

    # move label column to last
    embedded_tokens = np.array(embedded_tokens)
    embedded_tokens[:, 2:] = np.roll(embedded_tokens[:, 2:], -1, 1)

    return embedded_tokens


def add_padding(embedded_tokens, window):
    '''
    Adds n=window empty rows to the beginning and the end of the table as padding.
    '''

    padding = np.zeros((window, embedded_tokens.shape[1]), dtype=int)
    padded_tokens = np.insert(embedded_tokens, 0, padding, axis=0)
    padded_tokens = np.vstack((padded_tokens, padding))

    # # adds padding at the end of the table
    # if padded_tokens.shape[0] % window != 0:
    #     missing_rows = window - (padded_tokens.shape[0] % window)
    #     padding = np.zeros((missing_rows+window, embedded_tokens.shape[1]), dtype=int)
    #     padded_tokens = np.vstack((padded_tokens, padding))

    return padded_tokens


def preprocess(padded_tokens, window):
    '''
    Prepares token table into a format suitable for machine learning algorithms.

    Input:

    [[Index 1, Word 1, Embedding(Word 1) 1, Embedding(Word 1) 2, ..., Embedding(Word 1) 200, Label 1],
      ...,
      Index N, Word N, Embedding(Word N) 1, Embedding(Word N) 2, ..., Embedding(Word N) 200, Label N]]
    
    Output:

    [ ...,
     [Embedding(Word i-window) 1, Embedding(Word i-window) 2, ... ,Embedding(Word i-window) 200,
      Embedding(Word i) 1, ..., Embedding(Word i) 2, Embedding(Word i) 200,
      Embedding(Word i+window) 1, Embedding(Word i+window) 2, ... ,Embedding(Word i+window) 200,
      Label i],
      ...]
    '''

    feature_columns = []
    embedding_columns = padded_tokens[:, 2:-1]

    for i in range(window, padded_tokens.shape[0]-window):
        embedding_rows = embedding_columns[i-window:i+window+1, :]
        feature_columns.append(embedding_rows.flatten())
    
    feature_columns = np.array(feature_columns)
    label_column = padded_tokens[window:-window, -1:]

    return np.hstack((feature_columns, label_column))


window = 10

ds_list = ['train', 'test', 'dev']

emb_filepath = 'Embedding/tencent_ailab_embedding_zh_d200_v0.2.0_s.txt'

for i in ds_list:
    print("Preparing %s dataset..." % i)

    raw_tokens          = read_source(filepath = 'CDTB-Modified/dzho.pdtb.cdtb_%s.tok' % i)
    embedded_tokens     = generate_embedding(emb_filepath, raw_tokens)
    padded_tokens       = add_padding(embedded_tokens, window)
    preprocessed_tokens = preprocess(padded_tokens)

    df = pd.DataFrame(preprocessed_tokens)

    df.to_csv('Datasets-Modified/dataset_%s.csv' % i, index=False)

    print("Done!")