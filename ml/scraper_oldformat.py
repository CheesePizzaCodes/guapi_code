# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 21:45:09 2022

@author: User
"""
from typing import List, Dict, Union
from bs4 import BeautifulSoup as BS
from urllib.request import urlopen

from file_io import write_data_to_json


def get_html_from_url(_url: str) -> BS:
    """
    Utility function
    :param _url:
    :return:
    """
    html_bytes = urlopen(_url).read()
    soup = BS(html_bytes, "html.parser")
    return soup


def get_next_page_url(current_html: BS) -> str:
    """

    :param current_html:
    :return:
    """
    next_page_url = current_html.find("a", {"aria-label": "Next "})["href"]
    return next_page_url


def get_urls_from_table(html_doc) -> List[str]:
    """
    Parses the root url to obtain a list of the urls of each element
    :param html_doc:
    :return:
    """
    tr_tags = html_doc.tbody.contents
    tr_tags = [i for i in tr_tags if i != '\n']  # remove annoying newlines
    url_list = [i.find('a')['href'] for i in tr_tags]

    return url_list


def parse_boat_data(boat_html: BS) -> Dict[str, Union[str, None]]:
    # Extract out_data
    data = boat_html.find_all('div', attrs={"class": "sailboatdata-out_data"})
    names = boat_html.find_all('div', attrs={"class": "sailboatdata-label"})
    # Format strings
    for i, (d, n) in enumerate(zip(data, names)):
        ds = d.string
        ns = n.string
        data[i] = ds
        names[i] = ns
        if ds is not None:
            data[i] = ds.strip()
        if ns is not None:
            names[i] = ns.strip()
    # Arrange in a dictionary
    extracted_data = dict(zip(names, data))
    # Eliminate nonetypes

    # Return

    extracted_data = {k: v for (k, v) in extracted_data.items() if v is not None}

    return extracted_data


def main():
    # initialize values
    url = "https://sailboatdata.com/sailboat"
    c = 1  # starting page
    b = c * 25
    # url = f'https://sailboatdata.com/sailboat?paginate=25&page={c}'
    out = []  # output out_data collector

    while True:

        html = get_html_from_url(url)
        boat_urls = get_urls_from_table(html)  # get url to all boats listed in a page
        for boat_url in boat_urls:  # Parse one page of boats
            boat_url += "?units=metric"  # add option to get units in metric
            boat_html = get_html_from_url(boat_url)
            boat_data = parse_boat_data(boat_html)
            boat_data["url"] = boat_url
            out.append(boat_data)

            print(f"Finished parsing boat number{b}")
            b += 1

        print(f"Finished parsing boats on page {c}")
        c += 1
        write_data_to_json(out, 'final_2')
        # When done parsing, get next page
        try:
            url = get_next_page_url(html)
        except TypeError:
            print("Next page not available. Finished parsing.")
            break


if __name__ == "__main__":
    main()
