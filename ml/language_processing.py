"""
Module for processing the scientific language.
Main objective is clustering the terms to reduce unique categories of e.g. manufacturing process, etc.
"""
from typing import List, Union
from distance import levenshtein
from collections import defaultdict

import numpy as np
from sklearn.cluster import AffinityPropagation, KMeans
import pandas as pd
from scipy.spatial.distance import pdist
from gensim.models import KeyedVectors

import file_io
import formatting
from file_io import load_embedding, save_embedding_w2v


def preprocess_to_vec(term, embeddings):
    tokens = term.split()
    vec = np.zeros(embeddings.vector_size)
    count = 0
    for token in tokens:
        if token in embeddings:
            vec += embeddings[token]
            count += 1
    if count != 0:
        vec /= count
    return vec


def tokenize_terms(unique_terms):
    """
    Uses the GloVe pretrained embeddings to transform the terms into vectors (tokens)
    :param unique_terms:
    :return:
    """
    # save embedding in correct format
    file_path = save_embedding_w2v()
    # load embedding
    embeddings = KeyedVectors.load_word2vec_format(file_path, binary=False)
    term_vectors = [preprocess_to_vec(term, embeddings) for term in unique_terms]
    return term_vectors


def cluster_kmeans(k: int, term_tokens):
    kmeans = KMeans(n_clusters=k, random_state=0)
    return kmeans.fit(term_tokens)


def display_clusters(unique_terms, labels):
    cluster_dict = defaultdict(list)
    for i, process in enumerate(unique_terms):
        cluster_id = labels[i]
        cluster_dict[cluster_id].append(process)

    # Print the processes in each cluster
    for cluster_id, processes in cluster_dict.items():
        print(f"Cluster {cluster_id + 1}:")
        for process in processes:
            print(f"  - {process}")
    return cluster_dict

# Usage


def replace_strings(_words: Union[List[str], np.ndarray], to_replace: str, replace_with: str) -> np.ndarray:
    return np.asarray([word.replace(to_replace, replace_with) for word in _words])


def remove_periods(words_list) -> np.ndarray:
    """
    Remove points from the list of terms for avoiding problems
    :param words_list:
    :return:
    """
    [word.replace('.', '') for word in words_list]
    return np.array([word.replace('.', '') for word in words_list])


def cluster_affprop(unique_terms):
    _n = unique_terms.size
    lev_similarity = -1 * pdist(unique_terms.reshape(-1, 1), levenshtein)
    mat = np.zeros((_n, _n))
    idx = np.triu_indices(_n, k=1)
    mat[idx] = lev_similarity
    affprop = AffinityPropagation(affinity="precomputed", damping=0.7, verbose=True, max_iter=50000)
    affprop.fit(mat)
    output = []
    for cluster_id in np.unique(affprop.labels_):
        exemplar = unique_terms[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(unique_terms[np.nonzero(affprop.labels_ == cluster_id)])
        output.append(cluster)
        cluster_str = ", ".join(cluster)
        print(" - *%s:* %s" % (exemplar, cluster_str))
    print(f'{np.unique(affprop.labels_).size} Clusters produced')
    print(output)


if __name__ == '__main__':
    data = file_io.load_data('./out_data/finalisimo.json')  # sailboat data
    data = formatting.format_data(data)  # dataframe

    attrs = ['Hull Type', 'Rigging Type', 'Construction']  # TODO keep this list elsewhere
    i = 1

    words = data[attrs[i]].values  # select column
    words = np.unique(np.asarray(words, dtype='str'))  # remove duplicates
    words = replace_strings(words, 'Frac.', 'Fractional')
    # cluster_affprop(words)

    tokens = tokenize_terms(words)
    kmeans = cluster_kmeans(10, tokens)
    display_clusters(words, kmeans.labels_)
