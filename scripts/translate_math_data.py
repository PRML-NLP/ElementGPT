import os, argparse
from tqdm import tqdm
import json
import re

import openai
from datasets import load_dataset
from torch.utils.data import DataLoader

openai.api_key = os.getenv("OPENAI_API_KEY")
chatgpt = openai.ChatCompletion()

translate_prompt = "Translate this into Korean for grade school students:\n\n"
one_shot_Q = "1. Q) A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take? A) It takes \\equ{2/2=1} bolt of white fiber So the total amount of fabric is \\equ{2+1=3} bolts of fabric -> correct answer: 3"
one_shot_R = "1. Q) 로브를 만들기 위해서 파란색 섬유 2개와 그 절반만큼의 흰색 섬유가 필요합니다. 총 몇 개의 섬유가 필요할까요? A) 흰색 섬유 \\equ{2/2=1}개만 필요합니다. 따라서 전체 섬유의 양은 \\equ{2+1=3}개입니다. -> 정답: 3"

def translate(dataloader):
    total_translated = []
    for samples in tqdm(dataloader):
        prompt = []
        questions = samples["question"]
        answers = samples["answer"]
        
        for i, (q, a) in enumerate(zip(questions, answers)):
            a = a.replace("####", "-> correct answer:")
            a = re.sub(r"[\d+\-*/.x() =]+([$]?)<<([\d+\-*/.()]+)=([\-\d.]+)>>[ \d.]+", r" \1\\equ{\2=\3} ", a)
            a = a.replace(" $ $\\equ", " $\\equ")
            prompt.append(f"{i+2}. Q) {q} A) {a}")
            
        prompt = "\n---\n".join(prompt)
            
        response = chatgpt.create(
          model="gpt-3.5-turbo",
          messages=[{"role":"user", "content": f"{translate_prompt}{one_shot_Q}\n---\n{prompt}\n\n번역:\n{one_shot_R}\n---\n"}],
          temperature=0.3,
          max_tokens=2048,
          top_p=1.0,
          frequency_penalty=0.0,
          presence_penalty=0.0
        )
        
        translated = response.choices[0].message.content
        translated = translated.split("\n---\n")

        try:
            assert len(translated)==len(samples["question"])
        except AssertionError:
            translated = response.choices[0].message.content
            translated = translated.split("\n\n")
            if len(translated)!=len(samples["question"]):
                print("Discare samples:\n", translated)
                continue
        try:
            for text in translated:
                Q, A = text.split("A)")
                Q = Q.split("Q)")[-1].strip()
                A = A.strip()
                total_translated.append({"question": Q, "answer": A})
            
            print(total_translated[-1])
        except ValueError:
            print("Discare samples:\n", translated)
            
    return total_translated
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=str, default=4)
    args = parser.parse_args()
    
    main_dataset = load_dataset("gsm8k", "main")["train"]
    socratic_dataset = load_dataset("gsm8k", "socratic")["train"]
    main_loader = DataLoader(main_dataset, batch_size=args.batch_size)
    socratic_loader = DataLoader(socratic_dataset, batch_size=args.batch_size)
    
    main_translated = translate(main_loader)
    with open("data/gsm8k_ko_main.json", "w") as fout:
        json.dump(main_translated, fout, indent=2, ensure_ascii=False)
        
    socratic_translated = translate(socratic_loader)
    with open("data/gsm8k_ko_socratic.json", "w") as fout:
        json.dump(socratic_translated, fout, indent=2, ensure_ascii=False)