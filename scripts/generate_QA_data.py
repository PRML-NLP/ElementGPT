import os, argparse
from tqdm import tqdm
import json
from glob import glob

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
chatgpt = openai.ChatCompletion()

instruction = "이 텍스트를 기반으로 주어진 문맥 없이 대답할 수 있는 Q(질문)과 A(대답) 형식으로 {subject} 수업 자료를 작성해주세요.\n\n"
system_prompt = "I want you to act as an elementary teacher in Korea. You will be responsible for teaching students how to read, write, and act as a teacher. My first request is '나는 초등학생을 가르치는데 도움이 필요합니다.'"

def generate_QA(filelist, subject_name):
    inst = instruction.format(**{"subject":subject_name})
    
    total_qa = []
    for filename in tqdm(filelist):
        with open(filename, "rt") as fin:
            texts = fin.readlines()
        
        prompt = "".join(texts)
        
        if len(prompt) > 1500:
            prompt = prompt[:1500]

        try:
            response = chatgpt.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system", "content": system_prompt},
                    {"role":"user", "content": f"{prompt} {inst}"}
                    ],
                temperature=0.7,
                max_tokens=2048,
                top_p=0.93,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
        except openai.error.InvalidRequestError:
            print(filename, len(prompt))
        
        qa = response.choices[0].message.content
        qa = qa.split("\n\n")
        if len(qa)==1:
            qa = qa[0].split("\nQ: ")
            qa = ["Q: "+ t for t in qa]

        try:
            for k, text in enumerate(qa):
                if "A:" in text:
                    Q, A = text.split("A:")
                    Q = Q.split("Q:")[-1].strip()
                    A = A.strip()
                else:
                    Q, A = text.split("대답:")
                    Q = Q.split("질문:")[-1].strip()
                    A = A.strip()
                total_qa.append({"question": Q, "answer": A})
            print(total_qa[-1])
        except ValueError:
            print("Discare samples:\n", qa[k:])
            
    return total_qa
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_name", type=str, default="과학")
    parser.add_argument("--data_dirs", nargs="+", type=str, required=True)
    parser.add_argument("--save_name", type=str, default="QA")
    parser.add_argument
    args = parser.parse_args()
    
    filelist = []
    for data_dir in args.data_dirs:
        filelist.extend(glob(f"{data_dir}/*.txt"))
    
    generated = generate_QA(filelist, args.subject_name)
    with open(f"data/{args.subject_name}_{args.save_name}.json", "w") as fout:
        json.dump(generated, fout, indent=2)
        