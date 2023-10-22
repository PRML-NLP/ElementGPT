import requests
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import re

BASE_URLS = {
    "우리아이_성장백과": ("https://terms.naver.com/list.naver?cid=64672&categoryId=64672&so=date.dsc", 12), # &page=1~12 pages
    "어린이백과_국어": ("https://terms.naver.com/list.naver?cid=47303&categoryId=47303", 115),
    "어린이백과_영어": ("https://terms.naver.com/list.naver?cid=47304&categoryId=47304", 8),
    "어린이백과_사회": ("https://terms.naver.com/list.naver?cid=47305&categoryId=47305", 87),
    "어린이백과_한국사": ("https://terms.naver.com/list.naver?cid=47306&categoryId=47306", 159),
    "어린이백과_세계사": ("https://terms.naver.com/list.naver?cid=47307&categoryId=47307", 38),
    "어린이백과_수학" : ("https://terms.naver.com/list.naver?cid=47308&categoryId=47308", 20),
    "어린이백과_과학" : ("https://terms.naver.com/list.naver?cid=47308&categoryId=47309", 155),
    "어린이백과_예체능" : ("https://terms.naver.com/list.naver?cid=47308&categoryId=47310", 36),
    "어린이백과_시사/논술" : ("https://terms.naver.com/list.naver?cid=47308&categoryId=47311", 7),
    "어린이백과_문학탐구" : ("https://terms.naver.com/list.naver?cid=47308&categoryId=47314", 14),
    "어린이백과_세계탐구" : ("https://terms.naver.com/list.naver?cid=47308&categoryId=47315", 29),
    "어린이백과_과학탐구" : ("https://terms.naver.com/list.naver?cid=47308&categoryId=47316", 27),
    "어린이백과_인물탐구" : ("https://terms.naver.com/list.naver?cid=47308&categoryId=47317", 43),
    "어린이백과_유적/유물탐구" : ("https://terms.naver.com/list.naver?cid=47308&categoryId=47318", 6),
    "어린이백과_동식물" : ("https://terms.naver.com/list.naver?cid=47308&categoryId=55616", 93),
    "어린이백과_지식e" : ("https://terms.naver.com/list.naver?cid=47308&categoryId=58436", -1),
    "어린이백과_천재학습백과" : ("https://terms.naver.com/list.naver?cid=47308&categoryId=58582", 535),
    "어린이백과_소프트웨어" : ("https://terms.naver.com/list.naver?cid=47308&categoryId=59930", 10),
    "여행 이야기": ("https://terms.naver.com/list.naver?cid=58151&categoryId=58151", 30),
    "행정 서류": ("https://terms.naver.com/list.naver?cid=42279&categoryId=42279", 120)
}
ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"


def get_page_len(base_url: str) -> int:
    final_page = 1
    while True:
        page_num = 1

        url = base_url + "&page=" + str(page_num)
        response = requests.get(url, headers={"User-Agent": ua})

        page = response.text
        soup = BeautifulSoup(page, "html.parser")
        page_list = soup.select("div#paginate > a")
        last_page_num = int(page_list[-1]["href"].split("=")[-1])

        if last_page_num == final_page:
            break

        final_page = last_page_num

    return final_page


def get_contents(url):
    base = "https://m.terms.naver.com"
    url = base + url
    response = requests.get(url, headers={"User-Agent": ua})

    page = response.text
    soup = BeautifulSoup(page, "html.parser")
    contents = soup.find_all("div", {"class": "atomic-block"})

    result = []

    for content in contents:
        if "이미지 갤러리 가기" in content.text:
            continue
        elif content.find("h2"):
            texts = re.sub("\n+", "\n", content.text)
            texts = re.sub("\s+", " ", texts)
            result.append("## " + texts + "\n")
        else:
            texts = re.sub("\n+", "\n", content.text)
            texts = re.sub("\s+", " ", texts)
            result.append(texts)

    return "".join(result)


def parse_e(url):
    base = "https://m.terms.naver.com"
    url = base + url
    response = requests.get(url, headers={"User-Agent": ua})

    page = response.text
    soup = BeautifulSoup(page, "html.parser")
    contents = soup.find_all("div", {"class": "atomic-block"})

    result = []

    title = soup.select("h1 > span")
    title = " ".join([t.text.strip() for t in title])

    for content in contents:
        if "이미지 갤러리 가기" in content.text:
            continue
        elif content.find("strong"):
            headding = content.select("strong")
            headding = [h.text.strip() for h in headding]
            num = len(headding)
            headding = "\n".join(headding)
            result.append("## " + headding + "\n")
            texts = re.sub(f"\n+", f"\n", content.text[len(headding) + num :].strip())
            texts = re.sub(f"\s+", " ", texts)
            result.append(texts)
        else:
            texts = re.sub(f"\n+", f"\n", content.text.strip())
            texts = re.sub(f"\s+", " ", texts)
            result.append(texts)

    return "".join(result), title


def save_json(data):
    with open("icecream_doc.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_e_contents():
    base_url = "https://terms.naver.com/list.naver?cid=58436&categoryId=58436"
    res = requests.get(base_url)
    soup = BeautifulSoup(res.text, "html.parser")
    url_infos = soup.select(".contents_list > .contents_sub")[0]

    urls = [tags["href"] for tags in url_infos.select("a") if tags["href"] != "#"]

    total_results = []

    for url in urls:
        content, title = parse_e(url)
        results = {}

        content_url = url

        results["category"] = "어린이백과_지식e"
        results["title"] = title
        results["contents"] = content
        results["url"] = content_url

        total_results.append(results)
    return total_results


def main():
    total_results = []

    for k, (cat_name, base_url) in tqdm(enumerate(BASE_URLS.items())):
        if k!=(len(BASE_URLS.keys())-1):
            continue
        if cat_name == "어린이백과_지식e":
            e_results = get_e_contents()
            total_results.extend(e_results)
            continue
        # page_len = get_page_len(base_url)
        base_url, page_len = base_url
        print(page_len)
        for i in range(page_len + 1):
            url = base_url + "&page=" + str(i)
            response = requests.get(url, headers={"User-Agent": ua})
            soup = BeautifulSoup(response.text, "html.parser")

            content_list = content_list = soup.select("ul.content_list > li")

            for c in content_list:
                results = {}
                try:
                    title = c.find("img")["alt"]
                except:
                    title = c.find("a").text

                content_url = c.find("a")["href"]
                contents = get_contents(content_url)

                results["category"] = cat_name
                results["title"] = title
                results["contents"] = contents
                results["url"] = content_url

                total_results.append(results)

    save_json(total_results)


if __name__ == "__main__":
    main()
