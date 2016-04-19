from sklearn.feature_extraction.text import HashingVectorizer
import re, os, pickle

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir,'pkl_objects','stopwords.pkl'), 'rb'))

def preprocessor(text):
    # Remove the HTML markup
    text = re.sub('<[^>]*>','',text)
    # Extract emoticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # Remove the noses, make everything lowercase and insert the
    # Emoticons at the end
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text


def tokenizer(text):
    text = preprocessor(text)
    return [w for w in text.split() if w not in stop]

vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)