import argparse
import requests
from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta
import json, os
from tqdm import tqdm
from utils.doc_processor import process_child_growth


PEDIA_LINKS = {
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
    "대한민국_구석구석": ("https://terms.naver.com/list.naver?cid=42856&categoryId=42856&so=date.dsc", 952),
    # "학생백과_국어/문학": ("https://terms.naver.com/list.naver?cid=47319&categoryId=47319", 467),
    # "학생백과_한자": ("https://terms.naver.com/list.naver?cid=47320&categoryId=47320", 36),
    # "학생백과_사회": ("https://terms.naver.com/list.naver?cid=47321&categoryId=47321", 722),
    # "학생백과_한국사": ("https://terms.naver.com/list.naver?cid=47322&categoryId=47322",),
    # "학생백과_세계사": "https://terms.naver.com/list.naver?cid=47323&categoryId=47323",
    # "학생백과_수학": "https://terms.naver.com/list.naver?cid=47324&categoryId=47324",
    # "학생백과_과학": "https://terms.naver.com/list.naver?cid=47325&categoryId=47325",
    # "학생백과_기술/가정": "https://terms.naver.com/list.naver?cid=47326&categoryId=47326",
    # "학생백과_예체능": "https://terms.naver.com/list.naver?cid=47327&categoryId=47327",
    # "학생백과_인물": "https://terms.naver.com/list.naver?cid=47328&categoryId=47328",
    # "학생백과_시사/논술": "https://terms.naver.com/list.naver?cid=47329&categoryId=47329",
    # "학생백과_동식물": "https://terms.naver.com/list.naver?cid=55619&categoryId=55619",
    # "학생백과_소프트웨어": "https://terms.naver.com/list.naver?cid=59940&categoryId=59940",   
}


class NaverpediaCrawler(object):
    def __init__(self, base_links):
        self.base_links = base_links
    
    def parse_content_links(self, base_link, num_pages):
        """Parse content links from base link and number of pages"""
        
        page_links = [base_link + f"&page={i}" for i in range(1, num_pages+1)]
        content_links = []
        
        num_pages = len(page_links)
        for page_link in tqdm(page_links, total=num_pages):
            res = requests.get(page_link)
            res.raise_for_status()
            res.encoding = None
            html = res.text
            soup = BeautifulSoup(html, 'html.parser')
            content_list = soup.find('ul', {'class': 'content_list'}).findAll('div', {'class': 'image_area'})
            for div in content_list:
                link = div.find('a')
                real_link = 'https://terms.naver.com' + link.get('href')
                content_links.append(real_link)
        
        return content_links
    
    def get_all_content_links(self, save_dir=None):
        """Get all content links from base links"""
        
        results = {}
        for name, base_link in self.base_links.items():
            if base_link[1]==-1:
                continue
            print("Connect", name)
            content_links = self.parse_content_links(base_link[0], base_link[1])
            results[name] = content_links
        
        if save_dir is not None:
            with open(os.path.join(save_dir, "all_content_links.json"), "w") as fout:
                json.dump(results, fout, indent=2, ensure_ascii=False)
        
        return results
            
    def load_all_content_links(self, cache_path):
        with open(cache_path, "r") as fin:
            link_dict = json.load(fin)
            
        return link_dict
    
    def scrap_contents(self, links, process_func):
        """Scrap contents from links"""
        
        contents = []
        for link in tqdm(links, total=len(links), desc="Scraping"):
            res = requests.get(link)
            res.raise_for_status()
            res.encoding = None
            html = res.text
            soup = BeautifulSoup(html, "html.parser")
            titlebox = soup.find("div", {"class": "headword_title"})
            title = titlebox.find("h2", {"class": "headword"}).text
            subtitle = titlebox.find("p", {"class": "desc"}).text
            main_texts = soup.findAll("p", {"class": "se_textarea"})
            doc = []
            for seg in main_texts:
                doc.extend(seg.findAll("span"))
            
            # preprocess document
            doc = process_func(doc)
            doc = "\n".join(doc)
            
            contents.append({
                "title": title, "subtitle": subtitle,
                "doc": doc, "url": link
            })
        
        return contents
            
    def crawl(self, save_dir, cache_path=None):
        """Crawl all contents from base links"""
        
        if cache_path is not None:
            link_dict = self.load_all_content_links(cache_path)
        else:
            link_dict = self.get_all_content_links(save_dir)
        
        all_contents = {}
        for name, links in link_dict.items():
            print(name)
            if name=="우리아이_성장백과":
                all_contents[name] = self.scrap_contents(links, process_child_growth)
            else:
                break
                # raise NotImplementedError
        
        with open(os.path.join(save_dir, "parsed_contents.json"), "w") as fout:
            json.dump(all_contents, fout, indent=2, ensure_ascii=False)
        
        return all_contents
            
        

if __name__=="__main__":
    parser = argparse.ArgumentParser("Crawler")
    parser.add_argument("--save_dir", type=str, default="data/web_crawled")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    crawler = NaverpediaCrawler(PEDIA_LINKS)
    crawler.crawl(args.save_dir, cache_path="data/web_crawled/all_content_links.json")
    