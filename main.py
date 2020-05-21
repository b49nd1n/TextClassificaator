from os import mkdir
import json
import pickle
import re
from glob import glob
from os import mkdir
from os.path import join
from shutil import move

import joblib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nltk
import numpy as np
import requests
# matplotlib.use('TkAgg')
from pymystem3 import Mystem
from sklearn.decomposition import LatentDirichletAllocation as LDA
# from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from stop_words import get_stop_words

import preprocessing_tools as pt


# Let\`s do simple interface class
class TopicModeler(object):
    '''
    Inteface object for CountVectorizer + LDA simple
    usage.
    '''

    def __init__(self, count_vect, lda):
        '''
        Args:
             count_vect - CountVectorizer object from sklearn.
             lda - LDA object from sklearn.
        '''
        self.lda = lda
        self.count_vect = count_vect
        self.count_vect.input = 'content'

    def __call__(self, text):
        '''
        Gives topics distribution for a given text
        Args:
             text - raw text via python string.
        returns: numpy array - topics distribution for a given text.
        '''
        vectorized = self.count_vect.transform([text])
        lda_topics = self.lda.transform(vectorized)
        return lda_topics

    def get_keywords(self, text, n_topics=3, n_keywords=5):
        '''
        For a given text gives n top keywords for each of m top texts topics.
        Args:
             text - raw text via python string.
             n_topics - int how many top topics to use.
             n_keywords - how many top words of each topic to return.
        returns:
                list - of m*n keywords for a given text.
        '''
        lda_topics = self(text)
        lda_topics = np.squeeze(lda_topics, axis=0)
        n_topics_indices = lda_topics.argsort()[-n_topics:][::-1]

        top_topics_words_dists = []
        for i in n_topics_indices:
            top_topics_words_dists.append(self.lda.components_[i])

        keywords = np.zeros(shape=(n_keywords * n_topics, self.lda.components_.shape[1]))
        for i, topic in enumerate(top_topics_words_dists):
            n_keywords_indices = topic.argsort()[-n_keywords:][::-1]
            for k, j in enumerate(n_keywords_indices):
                keywords[i * n_keywords + k, j] = 1
        keywords = self.count_vect.inverse_transform(keywords)
        keywords = [keyword[0] for keyword in keywords]
        return keywords


def tf_idf():
    docs = list()
    for i in range(1, 11):
        with open("data/" + str(i) + ".txt") as f:
            docs.append("\n".join(s for s in f.readlines()))

    count_vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 1),
                                       stop_words=get_stop_words('russian'), lowercase=True,
                                       binary=False, strip_accents=None)
    count_vectorizer.fit(docs)
    tf_vectors = count_vectorizer.transform(docs)

    tfidf_transformer = TfidfTransformer(use_idf=True, smooth_idf=False)

    tfidf_model = tfidf_transformer.fit(tf_vectors)
    print(tfidf_model.transform(tf_vectors).toarray())
    print(tfidf_model.idf_)

    tfidf_vectors = tfidf_model.transform(tf_vectors)
    print(tfidf_vectors.toarray())


def download_data():
    r = requests.get(
        "http://localhost:8080/gggggg?limit=10000")

    data = r.content
    # print(data)
    j = json.loads(data)
    # print(j)
    for d in j:
        with open("data/" + str(d[0]) + ".txt", 'a') as the_file:
            the_file.write(re.sub(" +", " ", re.sub("</?\\w+>|[^\\w\\d ]", " ", d[1])))


def preprocess():
    # loading data
    data_path = 'data/'

    template = join(data_path, '*.txt')
    filenames = glob(template)
    print(len(filenames))

    # Let`\s tokenize all texts and collect texts lenghts to see texts lengths distribution.
    texts_lens = []
    for name in filenames:
        with open(name, 'r') as f:
            print('file: ' + f.name)
            text = f.read()
            tok_text = nltk.word_tokenize(text)
            tok_text = pt.normalize(tok_text, tokenized=True)
            texts_lens.append(len(tok_text))
    plt.hist(texts_lens, bins=[i * 100 for i in range(int(50000 / 100))])
    plt.show()

    # collecting small texts
    texts_lens = np.array(texts_lens)
    small_texts_mask = texts_lens < 100
    small_texts_names = np.array(filenames)[small_texts_mask]

    # creating separate directory for them.
    small_texts_dir = join(data_path, 'small_texts')
    mkdir(small_texts_dir)

    # moving them down there
    for name in small_texts_names:
        filename = name.split('/')[-1]
        dst = join(small_texts_dir, filename)
        move(name, dst)

    template = join(data_path, '*.txt')
    filenames = glob(template)
    train_names, test_names = train_test_split(filenames, test_size=0.1, random_state=666)
    print(len(train_names), len(test_names))

    # Serializing splits
    with open('data/train_names.pkl', 'wb') as f:
        pickle.dump(train_names, f)

    with open('data/test_names.pkl', 'wb') as f:
        pickle.dump(test_names, f)


def get_stop_words2():
    stop_words = get_stop_words('english')
    for x in get_stop_words('russian'):
        stop_words.append(x)
    return stop_words


def trainig():
    print("TRAINING!!!")

    stop_words = get_stop_words2()
    with open('data/train_names.pkl', 'rb') as f:
        train_names = pickle.load(f)
    print('Got %d stopwords.\nGot %d texts in training corpus.' % (len(stop_words), len(train_names)))

    print("TF_IDF")

    tf_idf = TfidfVectorizer(input='filename',
                             stop_words=stop_words,
                             smooth_idf=False
                             )
    tf_idf.fit(train_names)
    # getting idfs
    idfs = tf_idf.idf_
    # sorting out too rare and too common words
    # original 1.3 and 7
    # 2 6
    lower_thresh = 3.
    upper_thresh = 6.
    not_often = idfs > lower_thresh
    not_rare = idfs < upper_thresh

    mask = not_often * not_rare

    good_words = np.array(tf_idf.get_feature_names())[mask]
    # deleting punctuation as well.
    cleaned = []

    print("CLEANING")

    for word in good_words:
        word = re.sub("^(\d+\w*$|_+)", "", word)

        if len(word) == 0:
            continue
        cleaned.append(word)
    print("Len of original vocabulary: %d\nAfter filtering: %d" % (idfs.shape[0], len(cleaned)))

    m = Mystem()
    stemmed = set()
    voc_len = len(cleaned)
    for i in range(voc_len):
        print("stemming... " + str(i))
        word = cleaned.pop()
        stemmed_word = m.lemmatize(word)[0]
        stemmed.add(stemmed_word)

    stemmed = list(stemmed)
    print('After stemming: %d' % (len(stemmed)))

    # training count vectorizer
    voc = {word: i for i, word in enumerate(stemmed)}

    count_vect = CountVectorizer(input='filename',
                                 stop_words=stop_words,
                                 vocabulary=voc)

    dataset = count_vect.fit_transform(train_names)

    lda = LDA(n_components=60, max_iter=30, n_jobs=6, learning_method='batch', verbose=1)
    lda.fit(dataset)

    joblib.dump(lda, 'data/models/lda.pkl')
    joblib.dump(count_vect, 'data/models/countVect.pkl')
    joblib.dump(tf_idf, 'data/models/tf_idf.pkl')


def testing():
    with open('data/test_names.pkl', 'rb') as f:
        names = pickle.load(f)

    count_vect = joblib.load('data/models/countVect.pkl')
    dataset = count_vect.transform(names)

    lda = joblib.load('data/models/lda.pkl')

    ind = 20
    with open(names[ind], 'r') as f:
        text = f.read()
    print(len(text))
    print(names[ind])

    tm = TopicModeler(count_vect, lda)
    key_words = tm.get_keywords(text, n_topics=1, n_keywords=10)
    print(key_words)
    print(text)


def clasters():
    with open('data/test_names.pkl', 'rb') as f:
        names = pickle.load(f)

    count_vect = joblib.load('data/models/countVect.pkl')
    dataset = count_vect.transform(names)

    lda = joblib.load('data/models/lda.pkl')
    term_doc_matrix = count_vect.transform(names)

    embeddings = lda.transform(term_doc_matrix)

    from sklearn.manifold import TSNE

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=10)

    clust_labels = kmeans.fit_predict(embeddings)

    clust_centers = kmeans.cluster_centers_

    embeddings_to_tsne = np.concatenate((embeddings, clust_centers), axis=0)

    tSNE = TSNE(n_components=2, perplexity=15)

    tsne_embeddings = tSNE.fit_transform(embeddings_to_tsne)

    tsne_embeddings, centroids_embeddings = np.split(tsne_embeddings, [len(clust_labels)], axis=0)

    print(tsne_embeddings.shape, centroids_embeddings.shape)
    clust_indices = np.unique(clust_labels)

    clusters = {clust_ind: [] for clust_ind in clust_indices}
    for emb, label in zip(tsne_embeddings, clust_labels):
        clusters[label].append(emb)

    for key in clusters.keys():
        clusters[key] = np.array(clusters[key])
    colors = cm.rainbow(np.linspace(0, 1, len(clust_indices)))
    plt.figure(figsize=(10, 10))
    for ind, color in zip(clust_indices, colors):
        x = clusters[ind][:, 0]
        y = clusters[ind][:, 1]
        plt.scatter(x, y, color=color)

        centroid = centroids_embeddings[ind]
        plt.scatter(centroid[0], centroid[1], color='black', marker='x', s=100)

    plt.show()


if __name__ == "__main__":
    # download_data()
    # preprocess()
    # trainig()
    # # testing()
    clasters()
