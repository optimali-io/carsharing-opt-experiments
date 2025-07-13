import logging
import os
from typing import Dict, List, Union
from urllib.parse import urljoin

import requests
from retry import retry

log = logging.getLogger("core")


@retry(
    exceptions=(requests.exceptions.ConnectionError, requests.exceptions.Timeout),
    tries=10,
    delay=1,
)
def make_get_request(
    url_postfix: str, params: Dict = None, headers: Dict = None, api_path: str = None
) -> Union[Dict, List, None]:
    """Make get request for given url and handle errors."""
    auth_header = {"Authorization": f"Token {os.environ.get('SERVICE_API_TOKEN', '')}"}
    if headers:
        headers.update(auth_header)
    else:
        headers = auth_header
    if api_path:
        urlpath = urljoin(api_path, url_postfix)
    else:
        urlpath = url_postfix

    try:
        r = requests.get(url=urlpath, params=params, headers=headers, timeout=100)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        log.error(e)
        return
    return r.json()


@retry(
    exceptions=(requests.exceptions.ConnectionError, requests.exceptions.Timeout),
    tries=10,
    delay=1,
)
def make_get_list_request(
    url_postfix: str, params: Dict = None, headers: Dict = None, api_path: str = None
) -> Union[Dict, List, None]:
    """Make get request for given url and handle errors."""
    auth_header = {"Authorization": f"Token {os.environ['SERVICE_API_TOKEN']}"}
    if headers:
        headers.update(auth_header)
    else:
        headers = auth_header
    if api_path:
        urlpath = urljoin(api_path, url_postfix)
    else:
        urlpath = url_postfix
    results: List[Dict] = []

    try:
        r = requests.get(url=urlpath, params=params, headers=headers, timeout=100)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        log.error(e)
        return
    response = r.json()

    if isinstance(response, list):
        return response

    results.extend(response["results"])
    while response["next"]:
        try:
            response = requests.get(url=response["next"], headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            log.error(e)
            return
        response = response.json()
        results.extend(response["results"])
    return results
