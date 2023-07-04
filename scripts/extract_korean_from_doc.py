import os, argparse
import re
from bs4 import BeautifulSoup
from glob import glob
from tqdm import tqdm

def extract_korean_text(tag):
    korean_text = []

    for child in tag.children:
        if child.name is None:  # 텍스트 노드인 경우
            text = child.strip()
            if text:
                korean_text.append(text)
        else:  # 자식 태그인 경우 재귀적으로 호출
            korean_text.extend(extract_korean_text(child))

    return korean_text


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    unnessary = ["닫기", "이전", "다음", "체크", "다시 하기", "페이지 콘텐츠", "메뉴", "단원순", "가나다순", "페이지 이동", "내용", "용어 사전", "이 단원에서는 무엇을 할까요", "재생", "다시 한번 생각해 보세요."]
    
    filelist = glob(f"{args.data_dir}/*.xhtml")
    
    for filename in tqdm(filelist):
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()

        soup = BeautifulSoup(content, 'lxml')
        korean_text = extract_korean_text(soup.body)

        # 추출된 텍스트를 순서대로 출력
        clean_text = []
        for text in korean_text:
            if re.match(r"[가-힣ox.?!+/\*-=]+", text) is not None and re.match("//", text) is None \
                and re.match(r"<.+\s+>?", text) is None and re.match(r"<?\s+.+>", text) is None:
                text = re.sub(r' {2,}', '  ', text).strip()
                if text in unnessary or "정답 확인" in text:
                    continue
                clean_text.append(text)
        
        clean_text = "\n".join(clean_text)
        
        with open(args.save_dir+"/"+os.path.basename(filename)[:-5]+"txt", "w", encoding="utf-8") as fout:
            fout.writelines(clean_text)