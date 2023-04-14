from typing import List, Dict, Union

import re

import pandas as pd

from constants import UNITS


def remove_colon_from_titles(data: List[Dict]) -> List[Dict]:
    return [{k.replace(':', ''): v for (k, v) in dictio.items()} for dictio in data]


def add_units_to_titles(data: List[Dict]) -> List[Dict]:
    return [{f'{k}{extract_units(v)}': v for (k, v) in dictio.items()} for dictio in data]


def convert_values_to_numeric(data: List[Dict]) -> List[Dict]:
    return [{k: extract_value(v) for (k, v) in dictio.items()} for dictio in data]


def convert_decimal_separator(data: List[Dict]) -> List[Dict]:
    return [{k: v.replace(',', '.') for (k, v) in dictio.items()} for dictio in data]


def has_numbers(_string):
    return bool(re.search(r'\d', _string))


def ends_with_any_of(target_string: str, string_list: List[str]) -> bool:
    return any(target_string.endswith(ending) for ending in string_list)

def is_number_without_units(string):
    # Define a regular expression pattern to match a number
    pattern = r'^[\d,.]+$'
    # Check whether the string matches the pattern
    return bool(re.match(pattern, string))

def is_number_with_units(string):  # TODO careful, this was made by chatgpt
    # Define a regular expression pattern to match a number followed by units
    pattern = r'^[\d,.]+ [a-zA-Z]+$'
    # Check whether the string matches the pattern
    return bool(re.match(pattern, string))

def extract_value(raw_string: str) -> Union[float, str]:
    """
    Extracts either the numerical value and converts it to a float or the string value
    :param raw_string:
    :return:
    """
    # Remove commas
    s = raw_string.replace(',', '')
    # first case: data already numeric

    if is_number_without_units(s):
        return float(s)
    # second case: check for numbers and ending in units
    if has_numbers(s) & ends_with_any_of(s, UNITS):
        match = re.match(r'(\d*\.*\d+)', s)  # extract numbers
        if match is None:  # insurance against false positives such as 'MD 22 L'
            return raw_string
        return float(match.group(1))
    # third case: none, string type, categorical variable
    return raw_string


def extract_units(raw_string: str) -> str:
    """
    Extracts units from the string
    :param raw_string:
    :return:
    """
    # Deal with false positives
    if not is_number_with_units(raw_string):
        return ''
    units = list(filter(lambda u: u in raw_string, UNITS))
    if not units:  # TODO tis may be obsolete
        return ''
    if 'm²' in units[0]:
        print('2')
    return f' [{units[0].strip()}]'


def format_data(data: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Transforms the raw string out_data to the correct numeric type
    :param data:
    :return:
    """
    data = remove_colon_from_titles(data)
    # data = convert_decimal_separator(data)
    data = add_units_to_titles(data)
    data = convert_values_to_numeric(data)
    return pd.DataFrame(data)


def test_1():
    from file_io import load_data

    data = load_data('./out_data/finalisimo.json')
    data = format_data(data)
    ...


if __name__ == '__main__':
    test_1()
