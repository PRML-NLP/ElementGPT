import argparse, random, os, time
import json
import gradio as gr
import torch
from model_utils.conversation import get_conv_template
from transformers import AutoTokenizer, AutoModelForCausalLM
from chat import generate_stream

from peft import PeftModel

examples=[
    ['조선의 건국자는?'],
    ["소행성 충돌로 몇 년 안에 지구가 멸망한다던데요? 관련 과학 수업에서 사용할 자료를 제작해줘."],
    ["6학년 2학기 과학, 연소와 소화에 관한 논술형 문제 두 문제 제출해줘."],
    ["4학년 2학기 과학 지구와 달 단원에 대한 객관식 문항 한 문제 만들어줘."],
    ["초등학생들에게 별똥별에 대한 과학수업 자료 작성해줘."],
    ["문제: 우리말과 의미가 같도록 빈칸에 알맞은 말을 쓰시오.\n쓰레기는 공휴일에는 수거되지 않을 것이다. = __________ will not be collected on holidays."],
    ['옐로 카펫에 대해 설명해줘.'],
    ['제가 가르치는 반 학생 중 한 명이 저를 계속 때리는데 어떻게 대처해야 될지 모르겠어요.'],
    ["학년 초에 학부모에게 보내는 입학 안내 이메일 양식을 작성해줘."],
]

MODEL_NAME= "experiments/poly12.8b-DAPT"
ADAPTER_NAME = "experiments/poly12.8b-DAPT2INST_v4"
DEVICE_ID = 4
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16, # torch_dtype=torch.float16,
).to(f'cuda:{DEVICE_ID}')
model = PeftModel.from_pretrained(model, ADAPTER_NAME)
model = model.merge_and_unload()
model.eval()

LOG_DIR = "logs/"+MODEL_NAME.split("/")[-1]
os.makedirs(LOG_DIR, exist_ok=True)

def predict(message, history, temp, rep):
    conv = get_conv_template("elementgpt_for_teacher_inference")
    for turn in history:
        conv.append_message(conv.roles[0], turn[0])
        conv.append_message(conv.roles[1], turn[1])
    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
        
    input_prompt = conv.get_prompt()
    
    gen_params = {
        "model": MODEL_NAME,
        "prompt": input_prompt,
        "temperature": temp, # 0.75
        "repetition_penalty": rep,
        "max_new_tokens": 700,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }
    
    device= torch.device(f"cuda:{DEVICE_ID}")
    output_stream = generate_stream(model, tokenizer, gen_params, device=device)
    
    pre = 0
    partial_message = ""
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            partial_message = partial_message + " ".join(output_text[pre:now]) + " "
            pre = now
            yield partial_message
    
    partial_message = partial_message + " ".join(output_text[pre:])
    
    end_time = str(time.time())
    with open(os.path.join(LOG_DIR, end_time+".json"), "w") as fout:
        json.dump({"prompt": input_prompt, "instruction": message, "response": partial_message}, fout, ensure_ascii=False, indent=2)
             
    yield partial_message
        

if __name__=="__main__":
    # Launch the demo
    demo = gr.ChatInterface(predict, additional_inputs=[
                                gr.Slider(0.0, 1.0, value=0.75, label="Temperature"),
                                gr.Slider(1.0, 1.3, value=1.0, label="Repetition penalty")],
                            examples=examples,)
    demo.queue().launch(share=True)