import json
from typing import List, Dict, Tuple, Union

import numpy as np
import pandas as pd
from gensim.scripts.glove2word2vec import glove2word2vec

import formatting


def write_dataframe_to_xlsx(data: pd.DataFrame, filename) -> str:
    pth = f'./out_data/xl/{filename}.xlsx'
    data.to_excel(pth)
    return pth


def write_formatted_data_to_json(data: pd.DataFrame, out_filename: str) -> str:
    """

    :param data:
    :param out_filename:
    :return:
    """
    pth = f'./out_data/formatted_json/{out_filename}.json'
    data.to_json(pth, indent=4)
    return pth


def load_formatted_data(filename: str) -> pd.DataFrame:
    """

    :param filename:
    :return:
    """
    pth = f'./out_data/formatted_json/{filename}.json'
    return pd.read_json(pth)

def write_scrape_data_to_json(data: List[Dict[str, str]], out_filename: str) -> str:
    """
    IMPURE FUNCTION
    Loads an output file, extends it, and saves it. Returns the name of the file.
    :param out_filename:
    :param data:
    :return: File name
    """
    file_path = f'./out_data/raw_json/{out_filename}.json'
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


def load_scrape_data(out_filename: str) -> List[Dict[str, str]]:
    """
    Loads the out_data in the raw string format
    :param out_filename:
    :return:
    """
    pth = f'./out_data/raw_json/{out_filename}.json'
    with open(pth) as f:
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


def save_embedding_w2v(billions_of_tokens: int, dim: int) -> str:
    """
    CAREFUL: Impure function
    Uses external tool to save embedding in a specific format
    :param dim:
    :param billions_of_tokens:
    :return: Path to the processed file
    """
    root = f'/glove/glove.{billions_of_tokens}B.{dim}d.txt'
    in_file_path = './data' + root
    out_file_path = './out_data' + root
    _ = glove2word2vec(in_file_path, out_file_path)
    print(f'(Number of vectors, Dimensionality of the vectors): {_}')
    return out_file_path


if __name__ == '__main__':  # todo refactor to make it better to work with paths / names of files. Create all directories and declare locations inside the functions, make functions only take file names as arguments
    # _data = load_scrape_data('final_4')
    # _data = formatting.format_data(_data)
    # write_formatted_data_to_json(_data, 'final')
    data2 = load_formatted_data('final')

    terms = data2['Construction'].values


    ...
