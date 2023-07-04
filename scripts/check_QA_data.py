import os, argparse
from tqdm import tqdm
import json
from glob import glob
from torch.utils.data import DataLoader
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
chatgpt = openai.Completion()

instruction = "다음 QA에 대해 질문(Q)이 주어진 맥락 없이 대답이 가능한지 식별하고 [Y/N], QA에서 두 개 이하의 키워드를 추출하세요."
one_shot_Q = "1) Q: 실험 결과로 어떤 결론을 얻을 수 있었나요? A: 실험 결과로 푸트는 수증기의 양이 다른 공기의 온도 변화를 비교할 수 있었습니다. 이를 통해 수증기의 양이 온도 변화에 영향을 미친다는 결론을 얻을 수 있었습니다."
one_shot_R = "1) N, 키워드: 푸트의 실험, 수증기가 미치는 영향\n"

def check_QA(dataloader):
    total_qa = []
    
    for samples in tqdm(dataloader):
        prompt = []
        questions = samples["question"]
        answers = samples["answer"]
        
        for i, (q, a) in enumerate(zip(questions, answers)):
            prompt.append(f"{i+2}) Q: {q} A: {a}")
    
        prompt = "\n".join(prompt)
    
        try:
            response = chatgpt.create(
                model="text-davinci-003",
                prompt=f"{instruction}\n\n{one_shot_Q}\n{prompt}\n\n{one_shot_R}",
                temperature=0.2,
                max_tokens=2048,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
        except Exception as e:
            print(e)
            continue
        
        qa = response.choices[0].text
        qa = qa.split("\n")
        try:
            assert len(qa)==len(samples["question"])
        except AssertionError:
            qa = response.choices[0].message.content
            qa = qa.split("\n\n")
            if len(qa)!=len(samples["question"]):
                print("Discare samples:\n", qa)
                continue
        
        for i, res in enumerate(qa):
            try:
                ans, keywords = res.split(", 키워드: ")
            except ValueError:
                print("Discare samples:\n", res)
                continue
            
            ans = ans[-1]
            if ans=="Y":
                total_qa.append({
                    "question": samples["question"][i],
                    "answer": samples["answer"][i],
                    "keywords": keywords
                })
        
    return total_qa
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.data_path, "r") as fin:
        data = json.load(fin)
    
    dataloader = DataLoader(data, batch_size=args.batch_size)
    filtered = check_QA(dataloader)
    
    splited_name = args.data_path.split(".")
    save_name = ".".join(splited_name[:-1]) + "_filtered.json"
    with open(f"data/{save_name}", "w") as fout:
        json.dump(filtered, fout, indent=2, ensure_ascii=False)
        