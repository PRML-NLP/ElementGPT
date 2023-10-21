import os, argparse
from tqdm import tqdm
import json
from glob import glob

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
chatgpt = openai.ChatCompletion()

instruction = "이 텍스트를 기반으로 관련 주제에 대해 세부적으로 설명하는 {subject} 수업 자료를 작성해주세요."
system_prompt = "I want you to act as an elementary teacher in Korea. You will be responsible for teaching students how to read, write, and act as a teacher. My first request is '나는 초등학생을 가르치는데 도움이 필요합니다.'"

def generate_doc(filelist, subject_name):
    inst = instruction.format(**{"subject":subject_name})
    
    total_data = []
    for i, filename in tqdm(enumerate(filelist)):
        with open(filename, "rt") as fin:
            texts = fin.readlines()
        
        prompt = "".join(texts)
        if prompt.strip() == "":
            continue
        
        if len(prompt) > 1000:
            prompt = prompt[:1000]

        try:
            response = chatgpt.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system", "content": system_prompt},
                    {"role":"user", "content": f"{prompt}\n\n{inst}"}
                    ],
                temperature=0.7,
                max_tokens=2048,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
        except Exception as e:
            print(e)
            print("====>", filename, len(prompt))
            continue
        
        doc = response.choices[0].message.content
        if i%20==0:
            print(doc)
        
        total_data.append(doc)
            
    return total_data
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_name", type=str, default="사회")
    parser.add_argument("--data_dirs", nargs="+", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="pretraining")
    parser.add_argument
    args = parser.parse_args()
    
    filelist = []
    for data_dir in args.data_dirs:
        filelist.extend(glob(f"{data_dir}/*.txt"))
    
    generated = generate_doc(filelist, args.subject_name)
    with open(args.save_path, "w") as fout:
        json.dump(generated, fout, indent=2, ensure_ascii=False)
        