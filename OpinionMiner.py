import pyprind
import pandas as pd
import numpy as np
import os
import re
from sklearn.linear_model import SGDClassifier
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle
from vectorizer import vect
import sqlite3
stop = stopwords.words('english')


def process_file():
    pbar = pyprind.ProgBar(50000)
    labels = {'pos':1, 'neg':0}
    df = pd.DataFrame()
    for s in ('test','train'):
        for l in ('pos','neg'):
            path = './aclImdb/%s/%s' % (s,l)
            for file in os.listdir(path):
                with open(os.path.join(path,file), 'r') as infile:
                    txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index = True)
                pbar.update()
    df.columns = ['review','sentiment']
    # Shuffle the file
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    print('writing file to csv')
    df.to_csv('./movie_data.csv', index=False)
    return df


def load_df():
    df = None
    try:
        df = pd.read_csv('./movie_data.csv')
    except OSError:
        print('File does not exist.\n Loading files from tar.')
        df = process_file()
    print(df.head(3))
    return df


def preprocessor(text):
    # Remove the HTML markup
    text = re.sub('<[^>]*>','',text)
    # Extract emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove the noses, make everything lowercase and insert the
    # Emoticons at the end
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text


def tokenizer_porter(text):
    # Turns words into their 'base' words
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]


def tokenizer(text):
    text = preprocessor(text)
    return [w for w in tokenizer_porter(text) if w not in stop]


def stream_docs(path):
    with open(path,'r') as csv:
        next(csv) # Skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(doc_stream, size):
    docs, y = [],[]
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


def mine():
    print("Starting")
    clf = SGDClassifier(loss='log',random_state=1,n_iter=1)
    print('Create/Load Classifier')
    doc_stream = stream_docs(path='./movie_data.csv')
    print('Fitting data')
    classes = np.array([0,1])
    for _ in range(45):
        X_train, y_train = get_minibatch(doc_stream, size=1000)
        if not X_train:
            break
        X_train = vect.transform(X_train)
        clf.partial_fit(X_train, y_train, classes=classes)
    print('Finished Fitting')

    X_test, y_test = get_minibatch(doc_stream, size=5000)
    X_test = vect.transform(X_test)
    print('Accuracy: %.3f' % clf.score(X_test,y_test))

    print('create pickle objects')
    dest = os.path.join('','pkl_objects')
    if not os.path.exists(dest):
        os.makedirs(dest)

    pickle.dump(stop, open(os.path.join(dest,'stopwords.pkl'),'wb'), protocol=4)
    pickle.dump(clf, open(os.path.join(dest,'classifier.pkl'),'wb'), protocol=4)

#mine()

clf = pickle.load(open(os.path.join('pkl_objects','classifier.pkl'),'rb'))
label = {0:'neg',1:'pos'}
example = ['i love this movie']
X = vect.transform(example)
print('pred: %s\nProb: %.2f%%' % (label[clf.predict(X)[0]],np.max(clf.predict_proba(X)) * 100))

conn = sqlite3.connect('reviews.sqlite')
c = conn.cursor()
c.execute("CREATE TABLE review_db (review TEXT, sentiment INTEGER, date TEXT)")
example1 = example[0]
c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?,?,DATETIME('now'))", (example1,1))
example2 = 'i disliked this movie'
c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?,?,DATETIME('now'))", (example2,0))
conn.commit()
conn.close()