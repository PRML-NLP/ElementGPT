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

MODEL_NAME= "experiments/poly12.8b-DAPT2INST_wo_edu"
ADAPTER_NAME = None#"experiments/poly12.8b-DAPT2INST_third"
DEVICE_ID = 4
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.bfloat16 if ADAPTER_NAME is not None else torch.float16,
# ).to(f'cuda:{DEVICE_ID}')

# if ADAPTER_NAME is not None:
#     model = PeftModel.from_pretrained(model, ADAPTER_NAME)
#     model = model.merge_and_unload()
#     model.eval()

# FOLDER_NAME = MODEL_NAME if ADAPTER_NAME is None else ADAPTER_NAME
# LOG_DIR = "logs/"+FOLDER_NAME.split("/")[-1]
# os.makedirs(LOG_DIR, exist_ok=True)

# system_prompt = "호기심 많은 유저와 어시스턴트 간의 대화. 어시스턴트는 유저의 질문이나 지시에 도움이 되고 상세하며 정중한 답변을 합니다."

def get_prompt(selete_user):
    prompt_dict = {
        "교사" : "초등학교 교사와 어시스턴트 간의 대화. 어시스턴트는 선생님의 질문이나 지시에 도움이 되고 상세하며 정중한 답변을 합니다.",
        "학부모" : "초등학생 자녀의 학부모와 어시스턴트 간의 대화. 어시스턴트는 학부모의 질문이나 지시에 도움이 되고 상세하며 정중한 답변을 합니다.",
        "학생" : "초등학생과 어시스턴트 간의 대화. 어시스턴트는 학생의 질문이나 지시에 도움이 되고 상세하며 정중한 답변을 합니다"
    }
    return prompt_dict[selete_user]


def predict(message, history, selete_user, temp, rep):
    print(selete_user)
    
    conv = get_conv_template("elementgpt_for_inference")

    conv.system = get_prompt(selete_user)

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

def change_textbox(selete):
    return gr.Textbox(value=get_prompt(selete), label = "System prompt")


if __name__=="__main__":
    # Launch the demo
    with gr.Blocks() as demo:
        radio = gr.Radio(["교사", "학생", "학부모"], label="사용자를 선택해주세요", type='value')
        text = gr.Textbox(interactive=True, label="System prompt")
        radio.change(change_textbox, inputs=radio, outputs=text)
        
        chat = gr.ChatInterface(predict, additional_inputs=[
                                radio,
                                gr.Slider(0.0, 1.0, value=0.75, label="Temperature"),
                                gr.Slider(1.0, 1.3, value=1.0, label="Repetition penalty"),
                            ],
                            examples=examples,)
    demo.queue().launch(share=True)