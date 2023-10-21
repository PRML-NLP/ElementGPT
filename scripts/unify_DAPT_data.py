import json, os
from glob import glob
from tqdm import tqdm
import re

SUBJECT_MAP = {
    "art": "미술",
    "math": "수학",
    "physical": "체육",
    "science": "과학",
    "society": "사회"
}

if __name__=="__main__":
    PATH = "data/DAPT_data"
    WEB_CRWALING_PATH = "data/web_crawled/icecream.json"
    BOOK_SUMMARY_PATH = "data/web_crawled/AIhub_book.json"
    filelist = glob(os.path.join(PATH, "*.json"))
    
    unified_data = []
    for file_path in tqdm(filelist):
        filename = os.path.basename(file_path)
        splited = filename.split("_")
        subject = SUBJECT_MAP[splited[1]]
        grade = splited[2]
        if "." in grade:
            grade = grade.split(".")[0]
            
        semester = None
        if len(splited) > 3:
            semester = splited[3]
            if "." in semester:
                semester = semester.split(".")[0]
            
        with open(file_path, "r") as fin:
            data = json.load(fin)
        
        for sample in data:
            title = None
            subtitle = None
            if "title" in sample:
                title = sample["title"].strip()
                if "subtitle" in sample:
                    subtitle = sample["subtitle"].replace("부재 : ", "").strip() 
                if subtitle==title:
                    subtitle = None
            
            prefix = f"초등학교 {subject} {grade}학년"
            if semester is not None:
                prefix += f" {semester}학기 ({grade}-{semester})"
            
            if title is not None and "나는 초등학생을 가르치는데 도움이 필요합니다." in title:
                continue
            
            if title is not None:
                prefix += f"\n제목: {title}"
            if subtitle is not None:
                prefix += f"\n부제: {subtitle}"
                
            doc = sample["doc"]
            if "나는 초등학생을 가르치는데 도움이 필요합니다." in doc:
                continue
            try:
                if re.match(r"\([\d-]+[가-힣]+\)", doc.split("\n\n")[0]) is not None:
                    doc = "\n\n".join(doc.split("\n\n")[1:])
                
                doc = re.sub(r"([^\n]) (\d[.])", r"\1\n\2", doc)
                doc = re.sub(r"([^\n]) - ([가-힣])", r"\1\n- \2", doc)
            except Exception as e:
                print(e)
            
            unified_data.append(prefix+"\n"+doc)
            
    # with open("data/edu_textbook.json", "w") as fout:
    #     json.dump(unified_data, fout, ensure_ascii=False, indent=2)
    
    # raise NotImplementedError
    
    title_set = []
    with open(WEB_CRWALING_PATH, "r") as fin:
        crawled_data = json.load(fin)
        for data in crawled_data:
            if data["category"] in ["어린이백과_천재학습백과", "어린이백과_지식e"]:
                doc = ""
            else:
                splited = data["category"].split("_")
                if len(splited) > 1:
                    doc = " ".join(splited[1:]).strip() + ": "
                else:
                    doc = data["category"]+": "
            
            title = data['title'].replace("  ", " ")
            if title+data['contents'][:10] in title_set:
                continue
            doc += f"{title}\n\n{data['contents']}"
            unified_data.append(doc)
            title_set.append(title+data['contents'][:10])
            
    with open(BOOK_SUMMARY_PATH, "r") as fin:
        data = json.load(fin)
        unified_data.extend(data)
        
    print("Preprocessing...")
    processed = []
    for sample in tqdm(unified_data):
        output = re.sub(r"[.](\d+)[.]", r".\n\1.", sample)
        output = re.sub(r"개요 ([^:\n])", r"개요\n\1", output)
        output = re.sub(r"([가-힣.]) - ([가-힣])", r"\1\n- \2", output)
        output = re.sub(r"([가-힣.]) ([\d][.]) ([가-힣])", r"\1\n\2 \3", output)
        output = re.sub(r"([가-힣]): [-]", r"\1:\n-", output)
        output = re.sub(r"([^\n]) (\d[.])", r"\1\n\2", output)
        output = re.sub(r"([^\n]) - ([가-힣])", r"\1\n- \2", output)
        output = re.sub(r"([一-龥] [가-힣]+ [가-힣])([^,])", r"\1\n\2", output)
        output = re.sub(r"([가-힣])\?([가-힣])", r"\1?\n\2", output)
        output = re.sub(r" [*] ([가-힣])", r"\n* \1", output)
        output = re.sub(r"([가-힣])[.]([가-힣]+)", r"\1. \2", output)
        # output = output.replace(" 소.\n", "소개.\n")
        if len(output) < 35:
            continue
        processed.append(output)
        
            
    with open("data/unified_DAPT_data.json", "w") as fout:
        json.dump(processed, fout, ensure_ascii=False, indent=2)