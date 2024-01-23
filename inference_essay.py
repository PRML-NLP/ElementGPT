import argparse, json
import abc, gc
from typing import Iterable
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
import pandas as pd
from model_utils.conversation import get_conv_template
from openai import OpenAI
from utils import *
from rich.console import Console
from typing import List



def prepare_logits_processor(temperature: float, repetition_penalty: float, top_p: float, top_k: int) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def partial_stop(output, stop_str):
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


def stream_output(output_stream):
    pre = 0
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        now = len(output_text) - 1
        if now > pre:
            pre = now
    return " ".join(output_text)


@torch.inference_mode()
def generate_stream(
        model, tokenizer, params, device, context_len=1024, stream_interval=2
):
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.05))
    top_p = float(params.get("top_p", 0.9))
    top_k = int(params.get("top_k", 50))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 512))
    stop_str = params.get("stop", None)
    echo = bool(params.get("echo", True))
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    input_ids = tokenizer(prompt).input_ids
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:
        max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )

    past_key_values = out = None
    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)  ## here
                logits = out.logits
            past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor([[token]], device=device),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values,
                )

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""


def load_model_and_tokenizer(model_name, device_id):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=args.cache
    ).to(f'cuda:{device_id}')
    model.eval()

    return model, tokenizer

class GPTchatClass:
    def __init__(
            self,
            gpt_model: str = "gpt-3.5-turbo",
            role_msg: str = """Your are a binary answering assistant. 
                    If the given '결과' is even remotely close to '정답'
                    Generate a single output 'True' 
                    and if '결과' in any way signifies '정답'
                    Generate a single output 'True'
                    and if the given '결과' is absolutely ludacris to '정답',
                    Generate a single output 'False'        
                    In short,Aanswer True or False!
        """,
            key_path: str = 'key.txt',
    ):
        self.gpt_model = gpt_model
        self.role_msg = role_msg
        self.key_path = key_path

        self.messages = [{"role": "system", "content": f"{role_msg}"}]
        self.init_messages = [{"role": "system", "content": f"{role_msg}"}]
        self.response = None
        self.console = Console()

        self._setup_client()

    def _setup_client(self):
        with open(self.key_path, "r") as f:
            OPENAI_API_KEY = f.read()
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def _add_message(
            self,
            role="assistant",
            content="",
    ):
        """
        role: 'assistant' / 'user'
        """
        self.messages.append({"role": role, "content": content})


    def _get_response_content(self):
        if self.response:
            return self.response.choices[0].message.content
        else:
            return None

    def _get_response_status(self):
        if self.response:
            return self.response.choices[0].message.finish_reason
        else:
            return None

    def reset(
            self,
            role_msg: str = """Your are a binary answering assistant. 
                    If the given '결과' is even remotely close to '정답'
                    Generate a single output 'True' 
                    and if '결과' in any way signifies '정답'
                    Generate a single output 'True'
                    and if the given '결과' is absolutely ludacris to '정답',
                    Generate a single output 'False'        
                    In short,Aanswer True or False!
        """,
    ):
        self.init_messages = [{"role": "system", "content": f"{role_msg}"}]
        self.messages = self.init_messages

    def chat(
            self,
            user_msg="hi",
            PRINT_USER_MSG=True,
            PRINT_GPT_OUTPUT=True,
            RESET_CHAT=True,
            RETURN_RESPONSE=False,
    ):
        self._add_message(role="user", content=user_msg)
        self.response = self.client.chat.completions.create(
            messages=self.messages,
            model=self.gpt_model,
        )
        # Backup response for continous chatting
        self._add_message(role="assistant", content=self._get_response_content())
        if PRINT_USER_MSG:
            self.console.print("[deep_sky_blue3][USER_MSG][/deep_sky_blue3]")

        if PRINT_GPT_OUTPUT:
            self.console.print("[spring_green4][GPT_OUTPUT][/spring_green4]")
        # Reset
        if RESET_CHAT:
            self.reset()
        # Return
        if RETURN_RESPONSE:
            return self._get_response_content()


def completion_loop(args):

    model, tokenizer = load_model_and_tokenizer(args.model, args.device_id)
    device = torch.device(f"cuda:{args.device_id}")
    print(f"-----------------------------device는 {device}--------------------------------------------------")


    with open(args.subject, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    valid = df['문제'].reset_index()
    label = df['정답'].reset_index()
    datasets = pd.concat([valid["문제"], label["정답"]],  axis=1)
    GPT = GPTchatClass(
        gpt_model="gpt-3.5-turbo",
        role_msg="""Your are a binary answering assistant. 
                    If the given '결과' is even remotely close to '정답'
                    Generate a single output 'True' 
                    and if '결과' in any way signifies '정답'
                    Generate a single output 'True'
                    and if the given '결과' is absolutely ludacris to '정답',
                    Generate a single output 'False'        
                    In short,Aanswer True or False!
        """,
        key_path=args.key_path,
    )
    
    
    data_list = []
    for q, t in zip(tqdm(datasets['문제']), datasets['정답']):
            
        conv = get_conv_template(name=args.prompt) # 추론을 위한 프롬프트
        conv.append_message(conv.roles[0], q)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        gen_params = {
            "model": args.model,
            "prompt": prompt,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "stop": "\n",
            "stop_token_ids": None,
            "echo": False,
        }
        
        output_stream = generate_stream(model, tokenizer, gen_params, device=device)
        outputs = stream_output(output_stream)

        check = q + "\n정답: "+ t + "\n결과 : "+outputs
        key = 0
        key = GPT.chat(
        user_msg=check, PRINT_USER_MSG=False, PRINT_GPT_OUTPUT=False,
        RESET_CHAT=True, RETURN_RESPONSE=True)
        
        d = {
            '문제':q,
            '정답':t,
            '결과':outputs,
            '점수':key
        }
        data_list.append(d)

    with open(args.output_dir, "w", encoding="utf-8") as file:
        # JSON 형태로 파일에 쓰기, 들여쓰기 4로 설정
        json.dump(data_list, file, ensure_ascii=False, indent=4)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./experiments/poly12.8b-DAPT2INST_wo_edu", choices= ["./experiments/poly12.8b-DAPT2INST_wo_edu", "./experiments/solar10.7b-wo-DAPT"])
    parser.add_argument("--subject", type=str, default="./demos/society_essay.json", help="problem_file", choices=['./demos/science_essay.json', './demos/society_essay.json'])
    parser.add_argument("--prompt", type=str, default="elementgpt_for_society_essay", help="Instruction and sample exercise", choices=['elementgpt_for_science_essay', 'elementgpt_for_society_essay'])
    parser.add_argument("--output_dir", type=str, default="./outputs/society_essay_answer_solar.json", help="answer_sheet")
    parser.add_argument("--cache", type=str, default="./.cache")
    parser.add_argument("--key_path", type=str, default="./key.txt", help="OPEN_AI_KEY_text")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device_id", type=int, default=7)
    args = parser.parse_args()

    completion_loop(args)
