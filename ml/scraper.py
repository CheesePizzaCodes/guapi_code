from typing import List, Dict, Union

from bs4 import BeautifulSoup as BS
from urllib.request import Request, urlopen
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

from file_io import write_data_to_file  # (replace this with your actual import)


def get_html_from_url(_url: str) -> BS:
    """
    Utility function
    :param _url:
    :return:
    """
    req = Request(
        url=_url,
        headers={'User-Agent': 'Mozilla/5.0'})
    html_bytes = urlopen(req).read()
    return BS(html_bytes, "html.parser")


def get_html_from_url_dynamic(_url: str) -> BS:
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    driver.get(_url)
    return BS(driver.page_source, 'html.parser')


def get_next_page_url(current_html: BS) -> str:
    next_page_url = current_html.find("a", {"class": "page-link next"})["href"]
    return next_page_url


def get_urls_from_boat_listing(html_doc: BS) -> List[str]:
    tr_tags = html_doc.find_all('td', {'data-title': "MODEL"})
    url_list = [i.find('a')['href'] for i in tr_tags]
    return url_list


def parse_boat_data(boat_html: BS) -> Dict[str, Union[str, None]]:
    tags = boat_html.find_all('td')
    out = {}
    for i in range(0, len(tags), 2):
        try:  # try to transform to strings
            out[tags[i].string.strip()] = tags[i + 1].string.strip()
        except AttributeError as e:
            out[tags[i].string.strip()] = None
        except Exception as e:  # if this still fails
            continue  # do not add anything
    return out


def generate_listing_url(_page_number: int) -> str:
    items_per_page: int = 100
    return f'https://sailboatdata.com/?sailboats_per_page={items_per_page}&page_number={_page_number - 1}'


def main():
    page_number = 1
    boat_number = 25 * (page_number - 1)
    url = generate_listing_url(page_number)
    out = []
    while True:
        try:
            html = get_html_from_url_dynamic(url)
        except Exception as e:
            print("Next page not available. Finished parsing.")
            break
        boat_urls = get_urls_from_boat_listing(html)
        for boat_url in boat_urls:
            boat_data = process_boat_data(boat_url)
            out.append(boat_data)
            print(f"Finished parsing boat number {boat_number}")
            boat_number += 1
        print(f"Finished parsing boats on page {page_number}")
        page_number += 1
        url = generate_listing_url(page_number)



        write_data_to_file(out, 'final_2')  # (Uncomment and replace with your actual function)


def process_boat_data(boat_url):
    boat_url += "?units=metric"
    boat_html = get_html_from_url(boat_url)
    boat_data = parse_boat_data(boat_html)
    boat_data["url"] = boat_url
    return boat_data


if __name__ == "__main__":
    main()
