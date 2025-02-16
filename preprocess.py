import numpy as np
import pandas as pd
import tokenize
import imblearn
import imblearn.over_sampling
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


def generate_embedding(model, raw_tokens, emb_size):
    '''
    Generates a 'emb_size'-dimension vector embedding for the Chinese word in each token.

    Embedding obtained from Tencent AI Lab Embedding Corpus for Chinese Words and Phrases.
    https://ai.tencent.com/ailab/nlp/en/embedding.html
    '''

    embedded_tokens = []

    for token in raw_tokens:
        if token[1] in model.index_to_key:
            token = np.append(token, model[token[1]])
        else:
            token = np.append(token, np.zeros(emb_size, dtype=int))
        
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


def preprocess(embedded_tokens, window):
    '''
    Prepares token table into a format suitable for machine learning algorithms.

    Input:

    [[Index 1, Word 1, Embedding(Word 1) 1, Embedding(Word 1) 2, ..., Embedding(Word 1) 100, Label 1],
      ...,
      Index N, Word N, Embedding(Word N) 1, Embedding(Word N) 2, ..., Embedding(Word N) 100, Label N]]
    
    Output:

    [ ...,
     [Embedding(Word i-window) 1, Embedding(Word i-window) 2, ... ,Embedding(Word i-window) 100,
      Embedding(Word i) 1, ..., Embedding(Word i) 2, Embedding(Word i) 100,
      Embedding(Word i+window) 1, Embedding(Word i+window) 2, ... ,Embedding(Word i+window) 100,
      Label i],
      ...]
    '''

    padded_tokens = add_padding(embedded_tokens, window)

    feature_columns = []
    embedding_columns = padded_tokens[:, 2:-1]

    for i in range(window, padded_tokens.shape[0]-window):
        embedding_rows = embedding_columns[i-window:i+window+1, :]
        feature_columns.append(embedding_rows.flatten())
    
    feature_columns = np.array(feature_columns)
    label_column = padded_tokens[window:-window, -1:]

    return np.hstack((feature_columns, label_column))


print("Loading embedding models...")

model_100s = KeyedVectors.load_word2vec_format('Embeddings/tencent_ailab_embedding_zh_d100_v0.2.0_s.txt', binary=False)
model_200s = KeyedVectors.load_word2vec_format('Embeddings/tencent_ailab_embedding_zh_d200_v0.2.0_s.txt', binary=False)
# model_200l = KeyedVectors.load_word2vec_format('Embeddings/tencent_ailab_embedding_zh_d200_v0.2.0_l.txt', binary=False)

print("Done!")

models = ['100s', '200s'] # ['100s', '200s', '200l']
windows = ['10', '20']
ds_list = ['train', 'test', 'dev']

for i in models:
    for j in windows:

        # select model
        if i == '100s':
            model = model_100s
            emb_size = 100
        elif i == '200s':
            model = model_200s
            emb_size = 200
        else:
            # model = model_200l
            emb_size = 200

        # select window
        if j == '10':
            window = 10 # 63.9% of the paired DCs in the CDTB have a maximum of 10 word distance
        else:
            window = 20 # 89.2% of the paired DCs in the CDTB have a maximum of 20 word distance

        ds_columns = [str(x) for x in list(range(0, emb_size*(2*window+1)))]
        ds_columns.append('label')

        print("\n")
        print("Embedding: %s -- Window: %s" % (i, j))
        print("\n")

        for k in ds_list:
            print("Preparing %s dataset..." % k)

            raw_tokens          = read_source(filepath = 'CDTB/dzho.pdtb.cdtb_%s.tok' % k)
            embedded_tokens     = generate_embedding(model, raw_tokens, emb_size)
            preprocessed_tokens = preprocess(embedded_tokens, window)

            df = pd.DataFrame(preprocessed_tokens, columns=ds_columns)

            # 0 = not dc, 1 = dc, 2 = paired dc
            df['label'].replace({'_': 0, 'Seg=B-Conn': 1, 'Seg=I-Conn': 1, 'Seg=B-D-Conn': 2, 'Seg=I-D-Conn': 2}, inplace=True)
            df.to_csv('Datasets/dataset_3way_%s_%s_%s.csv' % (i, j, k), index=False)

            # # 0 = not dc, 1 = dc, 1 = paired dc
            # df['label'].replace({'_': 0, 'Seg=B-Conn': 1, 'Seg=I-Conn': 1, 'Seg=B-D-Conn': 1, 'Seg=I-D-Conn': 1}, inplace=True)
            # df.to_csv('Datasets/dataset_2way_%s_%s_%s.csv' % (i, j, k), index=False)

            print("Done!")
            print("Class count:")
            print(df['label'].value_counts(dropna=False))
            print("\n")

            # balance train dataset
            if k == 'train':
                print("Preparing balanced %s dataset..." % k)

                # prepare data for resampling
                df['label'].replace('', np.nan, inplace=True)
                df.dropna(inplace=True)
                X_copy = df.drop(['label'], axis=1).to_numpy(dtype=np.float64, copy=True)
                y_copy = df['label'].to_numpy(dtype=np.int64, copy=True)

                # 70% not dc, 20% dc, 10% paired dc
                strategy = {0:50850, 1:14530, 2:7265}
                
                ros = imblearn.over_sampling.RandomOverSampler(sampling_strategy=strategy)
                X_over, y_over = ros.fit_resample(X_copy, y_copy)

                df_over = pd.DataFrame(np.column_stack((X_over, y_over)), columns=ds_columns)
                df_over.to_csv('Datasets/dataset_3way_%s_%s_%s_balanced.csv' % (i, j, k), index=False)

                print("Done!")
                print("Class count:")
                print(df_over['label'].value_counts(dropna=False))
                print("\n")