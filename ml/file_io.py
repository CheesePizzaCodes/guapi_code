import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from gensim.scripts.glove2word2vec import glove2word2vec


def write_data_to_xlsx(data: pd.DataFrame, filename) -> str:
    pth = f'./out_data/xl/{filename}.xlsx'
    data.to_excel(pth)
    return pth

def write_data_to_json(data: List[Dict[str, str]], out_filename: str) -> str:
    """
    IMPURE FUNCTION
    Loads an output file, extends it, and saves it. Returns the name of the file.
    :param out_filename:
    :param data:
    :return: File name
    """
    file_path = f'./out_data/{out_filename}.json'
    # # read existing data
    try:
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = []
    # Extend data
    existing_data.extend(data)
    # Write file
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)
        print(f"Writing to {f.name}")
    return f.name


def load_data(json_file_path: str) -> List[Dict[str, str]]:
    """
    Loads the out_data in the raw format
    :param json_file_path:
    :return:
    """
    with open(json_file_path) as f:
        data = json.load(f)
    return data


def load_embedding(file_path: str = './data/glove/glove.840B.300d.txt', verbose: bool = False) -> Dict[str, np.ndarray]:
    """
    TODO does this really belong here??
    Loads the GloVe pretrained embeddings as a dictionary
    https://nlp.stanford.edu/projects/glove/
    :param verbose: print full traceback for failed data collection
    :param file_path: path of the GloVe file
    :return: Dictionary mapping a word to a 300D vector as a ndarray
    """
    embeddings_index = {}  # initialize dictionary
    with open(file_path, encoding='utf-8') as f:
        c = 1
        for line in f:
            values = line.split()
            word = values[0]
            # print(c)
            if c == 52342:
                pass
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except ValueError as e:  # some entries contain literals that cannot be converted to float
                print(f'Word number {c} failed. Word followed by first values:  {values[:10]}')
                if verbose:
                    print(e)
            embeddings_index[word] = coefs
            c += 1
    return embeddings_index


def save_embedding_w2v(file_path: str = './data/glove/glove.840B.300d.txt',
                       out_file: str = './out_data/glove/glove.840B.300d.txt') -> str:
    """
    CAREFUL: Impure function
    Uses external tool to save embedding in a specific format
    :param file_path:
    :param out_file:
    :return:
    """
    _ = glove2word2vec(file_path, out_file)
    print(f'(Number of vectors, Dimensionality of the vectors): {_}')
    return out_file
