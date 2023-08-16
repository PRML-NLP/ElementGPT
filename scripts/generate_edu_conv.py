import os, argparse
from tqdm import tqdm
import json
from glob import glob

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
chatgpt = openai.ChatCompletion()

instruction = "제목: {title}\n내용: {contents}\n\n이 내용을 '초등학교 교사'와 '어시스턴트'와의 대화로 변환해주세요. '교사'는 학생들을 지도할 때 도움이 되는 정보를 얻기 위해 질문이나 요청을 하고, '어시스턴트'는 교사의 요청에 대해 내용을 기반으로 정중히 답변합니다. 대화로 변환할 때, 교사나 어시스턴트는 이 내용을 제공받지 않은 상황이라 가정하십시오."
system_prompt = "You are a helpful assistant."

def generate_conv(data):
    total_data = []
    for i, sample in tqdm(enumerate(data), total=len(data)):
        if sample["category"]=="대한민국_구석구석":
            continue
        
        inst = instruction.format(**sample)
        
        try:
            response = chatgpt.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system", "content": system_prompt},
                    {"role":"user", "content": inst}
                    ],
                temperature=0.33,
                max_tokens=1200,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
        except Exception as e:
            print(e)
            print("====>", inst[:50], len(inst))
            continue
        
        doc = response.choices[0].message.content
        turns = doc.split("교사:")[1:]
        dialog = []
        for j, turn in enumerate(turns):
            try:
                usr, sys = turn.split("어시스턴트:")
            except Exception as e:
                print(e, j, turn)
                break
                
            usr = usr.strip()
            sys = sys.strip()
            dialog.append({"from": "human", "value": usr})
            dialog.append({"from": "bot", "value": sys})
        
        if dialog:
            dialog[1]["domain"] = "education"
            total_data.append(dialog)
        
        if i%50==0:
            print(doc)
        
    return total_data
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="parsed_conv.json")
    parser.add_argument
    args = parser.parse_args()
    
    with open(args.data_path, "r") as fin:
        data = json.load(fin)
    
    generated = generate_conv(data)
    with open(args.save_path, "w") as fout:
        json.dump(generated, fout, indent=2, ensure_ascii=False)
        