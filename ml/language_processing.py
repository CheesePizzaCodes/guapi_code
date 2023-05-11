"""
Module for processing the scientific language.
Main objective is clustering the terms to reduce unique categories of e.g. manufacturing process, etc.
"""
import re
from typing import List

import pandas as pd
from distance import levenshtein
from collections import defaultdict
import numpy as np
from sklearn.cluster import AffinityPropagation, KMeans
from scipy.spatial.distance import pdist
from gensim.models import KeyedVectors

import file_io
import formatting
from file_io import save_embedding_w2v
from formatting import replace_strings



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


def tokenize_terms(unique_terms: np.ndarray, billions_of_tokens=6, dim=50):
    """
    Uses the GloVe pretrained embeddings to transform the terms into vectors (tokens)
    :param dim:
    :param billions_of_tokens:
    :param unique_terms:
    :return:
    """
    # save embedding in correct format
    file_path = f'./out_data/glove/glove.{billions_of_tokens}B.{dim}d.txt'
    # load embedding
    embeddings = KeyedVectors.load_word2vec_format(file_path, binary=False)
    # vec_fn = np.vectorize(preprocess_to_vec)
    term_vectors = [preprocess_to_vec(term, embeddings) for term in unique_terms]
    # term_vectors = vec_fn(unique_terms, )
    return term_vectors


def cluster_kmeans(k: int, term_tokens) -> KMeans:
    kmeans = KMeans(n_clusters=k, random_state=0)
    fitted: KMeans = kmeans.fit(term_tokens)
    return fitted


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


def cluster_affprop(unique_terms):
    _n = unique_terms.size
    lev_similarity = -1 * pdist(unique_terms.reshape(-1, 1), levenshtein)
    mat = np.zeros((_n, _n))
    idx = np.triu_indices(_n, k=1)
    mat[idx] = lev_similarity
    affprop = AffinityPropagation(affinity="precomputed", damping=0.9, verbose=True, max_iter=50000)
    affprop.fit(mat)
    output = []
    for cluster_id in np.unique(affprop.labels_):
        exemplar = unique_terms[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(unique_terms[np.nonzero(affprop.labels_ == cluster_id)])
        output.append(cluster)
        cluster_str = ", ".join(cluster)
        print(" - *%s:* %s" % (exemplar, cluster_str))
    print(f'{np.unique(affprop.labels_).size} Clusters produced')
    for i in output: print(i)
    return output


def manual_classify(terms):
    foam = filter_any(['foam', 'cell', 'pvc', 'airex', 'klegcell'], terms)
    wood = filter_any(['balsa', 'wood', 'balsa'], terms)
    solid = filter_any(['solid', 'no core'], terms)
    other = filter_any(['carbon', 'steel', 'aluminum', 'alu'], terms)

    return dict(foam=foam,
                wood=wood,
                solid=solid,
                other=other,)


def filter_any(options: List[str], terms: List[str]) -> List[str]:
    """
    Filters a list of terms based on the presence of any word from a list of options, using
    regular expressions for exact word matching and considering word boundaries. The search
    is case-insensitive and avoids matching substrings (e.g., "sand" won't match "sandwich").

    Example:
    options = ["sand", "castle"]
    terms = ["The sand is warm.", "I love sandwiches.", "Sand castles are fun.", "This is not a sandwich."]
    filtered_terms = filter_any(options, terms)  # Output: ['The sand is warm.', 'Sand castles are fun.']

    :param options: A list of words to search for in the terms list.
    :param terms: A list of terms to filter based on the presence of any word from the options list.
    :return: A list containing the filtered terms that include at least one word from the options list.
    """
    return list(filter(lambda term: any(re.search(fr'\b{word}\b', term, re.IGNORECASE) for word in options), terms))


def preprocess_sentences(sentences: pd.Series) -> np.ndarray:
    """
    Preprocesses a pandas Series of sentences by filling NA values, replacing certain characters,
    removing extra spaces, and converting to lowercase.

    The function performs the following preprocessing steps:
    1. Replaces NA values with empty strings.
    2. Replaces '&' characters with 'and'.
    3. Replaces consecutive whitespace characters with a single space.
    4. Removes non-alphanumeric characters (except spaces) and converts the sentences to lowercase.

    :param sentences: A pandas Series containing sentences to be preprocessed.
    :return: A numpy array containing the unique preprocessed sentences.
    """
    sentences = sentences.fillna('')
    sentences = sentences.str.replace('&', 'and')
    sentences = sentences.str.replace(r'\bw\b', 'with', regex=True)
    sentences = sentences.str.replace(r'\s+', ' ', regex=True)
    sentences_clean: pd.Series = sentences.str.replace(r'[^\w\s]', ' ', regex=True).str.lower()
    clean_count: pd.Series = sentences_clean.value_counts()
    sentences_exploded: pd.Series = sentences_clean.str.split().explode()
    exploded_count = sentences_exploded.value_counts()
    return sentences_clean.unique()


def main():
    data = file_io.load_formatted_data('final')
    attrs = ['Hull Type', 'Rigging Type', 'Construction']  # TODO keep this list elsewhere
    i = 2

    words = data[attrs[i]]  # select column to use for clustering
    words = preprocess_sentences(words)
    # words = replace_strings(words, 'Frac.', 'Fractional')

    affprop = cluster_affprop(words)

    manual = manual_classify(words)

    tokens = tokenize_terms(words)
    kmeans = cluster_kmeans(100, tokens)
    display_clusters(words, kmeans.labels_)


    data['Construction Type'] = data['Construction'].map(
        {value: key for key, values in manual.items() for value in values}
    )
    file_io.write_formatted_data_to_json(data, 'final')


if __name__ == '__main__':
    main()

