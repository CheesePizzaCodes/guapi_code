import time
from typing import List, Dict, Union

from bs4 import BeautifulSoup as BS
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

from file_io import write_data_to_file  # (replace this with your actual import)


def get_html_from_url(_url: str) -> BS:
    """
    Utility function
    :param _url:
    :return:
    """
    attempts = 1
    req = Request(
        url=_url,
        headers={'User-Agent': 'Mozilla/5.0'})

    while attempts <= 5:
        try:
            html_bytes = urlopen(req).read()
            return BS(html_bytes, "html.parser")
        except HTTPError as e:
            print(f'Attempt {attempts}: failed to retrieve page at {_url}')
            print(e)
            attempts += 1
    return BS()  # return empty soup


def get_html_from_url_dynamic(_url: str) -> Union[BS, None]:
    """
    Tries 5 times to get webpage, if no success, returns None
    :param _url:
    :return:
    """
    attempts = 1
    while attempts <= 5:
        try:
            driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
            driver.get(_url)
            # while driver.execute_script("return document.readyState") != "complete":
            time.sleep(3)
            return BS(driver.page_source, 'html.parser')
        except WebDriverException as e:
            print(f'Attempt {attempts}: failed to retrieve page at {_url}')
            print(e)
            attempts += 1
    return None



def get_next_page_url(current_html: BS) -> str:
    """
    Устаребший
    :param current_html:
    :return:
    """
    next_page_url = current_html.find("a", {"class": "page-link next"})["href"]
    return next_page_url


def get_urls_from_boat_listing(html_doc: BS) -> Union[List[str], None]:
    """
    Attempts to retrieve list of boat urls. If fails, returns None
    :param html_doc:
    :return:
    """
    tr_tags = html_doc.find_all('td', {'data-title': "MODEL"})
    url_list = [i.find('a')['href'] for i in tr_tags]
    if not url_list:
        return None
    return url_list


def parse_boat_data(boat_html: BS) -> Dict[str, Union[str, None]]:
    tags = boat_html.find_all('td')
    out = {}
    if len(tags) == 0:  # In case getting boat data failed
        return out
    for i in range(0, len(tags), 2):
        try:  # try to transform to strings
            out[tags[i].string.strip()] = tags[i + 1].string.strip()
        except AttributeError as e:
            out[tags[i].string.strip()] = None
        except Exception as e:  # if this still fails
            continue  # do not add anything
    return out


def generate_listing_url(_page_number: int, items_per_page: int = 100) -> str:

    return f'https://sailboatdata.com/?sailboats_per_page={items_per_page}&page_number={_page_number - 1}'


def main():
    page_number = 1
    boats_per_page = 100
    boat_number = boats_per_page * (page_number - 1)
    url = generate_listing_url(page_number, boats_per_page)
    out = []
    while True:
        html = get_html_from_url_dynamic(url)
        if html is None:
            print(f"Page {page_number} not available. Skipping to page {page_number + 1}.")
            continue
        boat_urls = get_urls_from_boat_listing(html)
        if boat_urls is None:  # if returned list is empty, no more boats are available
            print("No more boats available. Finished parsing.")
            break
        for boat_url in boat_urls:
            boat_data = process_boat_data(boat_url)
            out.append(boat_data)
            print(f"Finished parsing boat number {boat_number}")
            boat_number += 1
        print(f"Finished parsing boats on page {page_number}")
        page_number += 1
        url = generate_listing_url(page_number)
        write_data_to_file(out, 'final_2')  # (Uncomment and replace with your actual function) TODO make this append instead of overwrite


def process_boat_data(boat_url: str) -> Dict[str, str | None]:
    """
    Processes the information o a single boat and returns a dictionary containing the attributes and values as strings
    :param boat_url:
    :return:
    """
    boat_url += "?units=metric"
    boat_html = get_html_from_url(boat_url)
    boat_data = parse_boat_data(boat_html)
    boat_data["url"] = boat_url
    return boat_data


if __name__ == "__main__":
    main()
