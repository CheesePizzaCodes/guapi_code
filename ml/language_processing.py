"""
Module for processing the scientific language.
Main objective is clustering the terms to reduce unique categories of e.g. manufacturing process, etc.
"""
from distance import levenshtein
from collections import defaultdict

import numpy as np
import pickle
from sklearn.cluster import AffinityPropagation, KMeans
from scipy.spatial.distance import pdist
from gensim.models import KeyedVectors

import file_io
import formatting
from file_io import save_embedding_w2v
from formatting import replace_strings
import gpt_language_processing


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


def tokenize_terms(unique_terms: np.ndarray):
    """
    Uses the GloVe pretrained embeddings to transform the terms into vectors (tokens)
    :param unique_terms:
    :return:
    """
    # save embedding in correct format
    file_path = save_embedding_w2v(billions_of_tokens=6, dim=50)
    # load embedding
    embeddings = KeyedVectors.load_word2vec_format(file_path, binary=False)
    # vec_fn = np.vectorize(preprocess_to_vec)
    term_vectors = [preprocess_to_vec(term, embeddings) for term in unique_terms]
    # term_vectors = vec_fn(unique_terms, )
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


def classify(terms):
    foam = filter_any(['foam', 'cell', 'pvc', 'airex'], terms)
    wood = filter_any(['balsa', 'wood', ], terms)
    solid = filter_any(['solid', 'no core'], terms)

    return dict(foam=foam,
                wood=wood,
                solid=solid)


def filter_any(options, terms):
    return list(filter(lambda term: any(word in term.lower() for word in options), terms))

def main():
    data = file_io.load_formatted_data('final')
    attrs = ['Hull Type', 'Rigging Type', 'Construction']  # TODO keep this list elsewhere
    i = 2

    words = data[attrs[i]].values  # select column to use for clustering
    words = np.unique(np.asarray(words, dtype='str'))  # remove duplicates
    words = replace_strings(words, 'Frac.', 'Fractional')
    # cluster_affprop(words)

    out = classify(words)
    # _tokens = tokenize_terms(words)
    # kmeans = cluster_kmeans(100, _tokens)
    # display_clusters(words, kmeans.labels_)

    # with open('./out_data/fitted_nlp_models/p1.pkl', 'wb') as file:
    #     pickle.dump(kmeans, file)

    # response = gpt_language_processing.main(words)
    data['Construction Type'] = data['Construction'].map(
        {value: key for key, values in out.items() for value in values}
    )
    file_io.write_formatted_data_to_json(data, 'final')


if __name__ == '__main__':
    main()

