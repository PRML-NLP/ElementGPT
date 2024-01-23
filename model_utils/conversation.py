"""
Conversation prompt templates.
Based on https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
"""

import dataclasses
from enum import auto, Enum
from typing import List, Any, Dict
import random


class SeparatorStyle(Enum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    ADD_NEW_LINE_SINGLE = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    # The name of this template
    name: str
    # Two roles
    roles: List[str]
    # All messages. Each item is (role, message).
    messages: List[List[str]]
    # The number of few shot examples
    offset: int
    # Separators
    sep_style: SeparatorStyle
    sep: str
    # List of the system prompts
    systems: List[str]
    # The system prompt
    system: str = ""
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            if self.system:
                ret = self.system + self.sep
            else:
                ret = ""
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            
            if self.systems:
                system = random.choice(self.systems)
            else:
                system = self.system
            ret = system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i%2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system=self.system,
            systems=self.systems,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, f"{template.name} has been registered."
    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# A template with a one-shot conversation example
register_conv_template(
    Conversation(
        name="one_shot",
        system="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        systems=[],
        roles=("Human", "Assistant"),
        messages=(
            (
                "Human",
                "Got any creative ideas for a 10 year old’s birthday?",
            ),
            (
                "Assistant",
                """Of course! Here are some creative ideas for a 10-year-old's birthday party:
1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.
2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.
3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.
4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.
5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.
6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.
7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.
8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.
Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!""",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)


# A template similar to the "one_shot" template above but remove the example.
register_conv_template(
    Conversation(
        name="zero_shot",
        systems=[],
        system="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# ElementGPT template1
register_conv_template(
    Conversation(
        name="elementgpt_for_teacher",
        system="초등학교 교사와 어시스턴트 간의 대화. 어시스턴트는 교사의 질문이나 지시에 도움이 되는 답변을 합니다.",
        systems=[
            "초등학교 교사와 어시스턴트 간의 대화. 어시스턴트는 교사의 질문이나 지시에 도움이 되는 답변을 합니다.",
            "초등학교 교사와 어시스턴트 사이의 상호작용. 어시스턴트는 교사의 질문과 지시에 유용한 답변을 제공합니다.",
            "어시스턴트는 초등학교 교사와의 대화에서 교사의 질문과 지시에 도움을 주며 유용한 답변을 제공합니다.",
            "초등학교 교사와 어시스턴트 간의 소통. 어시스턴트는 교사의 질문과 지시에 대답하여 도움을 제공합니다."
        ],
        roles=("### 질문", "### 답변"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="<|endoftext|>",
    )
)

# ElementGPT template1
register_conv_template(
    Conversation(
        name="elementgpt_for_inference",
        system="초등학교 교사와 어시스턴트 간의 대화. "
        "어시스턴트는 선생님의 질문이나 지시에 도움이 되고 상세하며 정중한 답변을 합니다.",
        systems=[],
        roles=("### 질문", "### 답변"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="<|endoftext|>",
    )
)

# ElementGPT template1
register_conv_template(
    Conversation(
        name="elementgpt_for_teacher_inference",
        system="초등학교 교사와 어시스턴트 간의 대화. "
        "어시스턴트는 교육에 대한 유저의 질문이나 지시에 도움이 되고 상세하며 정중한 답변을 합니다.",
        systems=[],
        roles=("### 질문", "### 답변"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="<|endoftext|>",
    )
)

# ElementGPT template2
register_conv_template(
    Conversation(
        name="elementgpt_for_general",
        system="호기심 많은 유저와 어시스턴트 간의 대화. 어시스턴트는 유저의 질문이나 지시에 도움이 되고 상세하며 정중한 답변을 합니다.",
        systems=[
            "호기심 많은 유저와 어시스턴트 간의 대화. 어시스턴트는 유저의 질문이나 지시에 도움이 되고 상세하며 정중한 답변을 합니다.",
            "유저와 어시스턴트 간의 상호작용. 어시스턴트는 유저의 질문과 지시를 세심하게 처리하며 정중하게 응답합니다.",
            "호기심 있는 유저와 어시스턴트 간의 상호작용. 어시스턴트는 유저의 질문과 지시를 세심하게 대응하며 공손한 답변을 제공합니다.",
            "유저와 어시스턴트 간의 대화. 호기심을 가진 유저의 질문과 지시를 받아들이고, 상세하고 예의 바른 방식으로 응답합니다."
        ],
        roles=("### 질문", "### 답변"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="<|endoftext|>",
    )
)

# ElementGPT template for short answer
register_conv_template(
    Conversation(
        name="elementgpt_for_science_short",
        
        system=
        """
        문제와 정답간의 대화. 정답은 문제에 알맞은 답변을 합니다.
        
        "문제 : 물체를 이루고 있는 물질의 성질에 대하여 바르게 말하지 못한 친구의 이름을 쓰시오. •민수: 물질마다 고유한 성질을 가지고 있습니다.•은지: 물질은 모두 색깔이나 만졌을 때의 느낌이 같습니다.•영민: 물질마다 긁히는 정도, 구부러지는 정도, 물에 뜨는 정도가 다릅니다.",
        "정답 : 물체를 이루고 있는 물질의 성질에 대하여 알아보는 문제입니다. 물질은 저마다 고유한 성질을 가지고 있으며, 물질마다 만졌을 때의 느낌, 긁히는 정도, 구부러지는 정도, 물에 뜨는 정도 등이 다릅니다. 그래서 정답은 은지"
        
        "문제 : 40g의 추를 매달았을 때 6cm가 늘어나는 용수철이 있습니다. 이 용수철에 필통을 매달았더니 용수철이 12cm가 늘어났습니다. 필통의 무게는 얼마인지 쓰시오. g, ",
        "정답 : 다양한 물질로 쓰임새가 같은 물체를 만들 때의 좋은 점을 알아보는 문제입니다. 섬유와 같은 부드러운 물질로 의자를 만들면 부딪혀도 다칠 염려가 적습니다. 철, 나무, 돌처럼 단단한 물질로 의자를 만들면 튼튼하고, 플라스틱으로 만든 의자는 가벼워서 이동이 편리합니다. 그래서 정답은 80"
        
        "문제 : 광학 현미경으로 물체를 40배 확대하여 관찰하려고 할 때 접안렌즈의 배율이 10배라면 대물렌즈의 배율은 몇 배로 해야 하는지 숫자로 쓰시오. 배, ",
        "정답 : 현미경의 배율은 접안렌즈 배율X대물렌즈 배율입니다. 접안렌즈가 10배, 대물렌즈가 4배라면 물체를 40배로 확대해 관찰할 수 있습니다. 그래서 정답은 4"
        
        "문제 : 물 120g에 설탕을 모두 녹여 만든 설탕물의 무게가 143g이었습니다. 물에 넣은 설탕의 무게는 몇 g인지 쓰시오. g, ",
        "정답 : 설탕이 물에 용해되기 전과 용해된 후의 무게를 비교하는 문제입니다. 설탕이 물에 용해되면 사라지는 것이 아니라 크기가 매우 작은 입자로 나누어져 물속에 섞여 들어가 그대로 남아 있기 때문에 무게가 변하지 않습니다. 즉, 설탕이 물에 용해되기 전 설탕의 무게와 물의 무게를 합친 무게는 용해된 후 설탕물의 무게와 같습니다. 따라서 물 120g에 설탕을 모두 녹여 만든 설탕물의 무게가 143g이므로 설탕의 무게는 23g(=143g-120g)입니다. 그래서 정답은 23(g)"
        
        "문제 : 설탕 20g이 들어있는 봉지에서 설탕을 두 숟가락 떠서 50g의 물에 넣고 저어 설탕물 55g을 만들었습니다. 설탕 봉지에 남은 설탕의 무게는 얼마인지 숫자로 쓰시오. g, ",
        "정답 : 설탕이 물에 용해되기 전과 용해된 후의 무게는 같습니다. 따라서 물 50g에 설탕 두 숟가락을 넣어 만든 설탕물의 무게가 55g이라면 설탕 두 숟가락의 무게는 5g입니다. 설탕이 든 봉지에서 설탕 두 숟가락인 5g을 사용하였으므로 설탕 봉지 안에는 설탕 15g이 남아 있습니다. 그래서 정답은 15"

        "문제 : 생산자인 식물이 사라진다면 생태계에는 어떤 일이 일어날지 바르게 말한 학생의 이름을 쓰시오.세희: 식물을 먹는 소비자는 결국 죽게 될 거야.진우: 우리 주변이 죽은 생물과 생물의 배출물로 가득 차게 될 거야.민수: 생태계에 다양한 생물이 서로 조화를 이루며 살아가게 될 거야., ",
        "정답 : 생산자인 식물이 사라진다면 식물을 먹는 소비자는 먹이가 사라지므로 결국 죽게 될 것입니다. 또 식물을 먹는 소비자를 먹이로 하는 소비자들도 죽게 될 것입니다. 결국 생태계에는 어떤 생물 요소도 살아남지 못할 것입니다. 그래서 정답은 세희"
        
        "문제 : 땅에서 사는 동물의 특징을 바르 게 말한 학생의 이름을 쓰시오.민수: 개미는 땅속에서 살고, 다리가 두 쌍이 있어.윤주: 지렁이는 앞다리로 땅속에 굴을 파서 이동해.정우: 다람쥐는 몸이 갈색 털로 덮여 있고, 등에 줄무늬가 있어., ",
        "정답 : 다람쥐는 몸이 갈색 털로 덮여 있고 등에 진한 갈색 줄무늬가 있으며, 다리는 두 쌍이고 땅 위에서 걷거나 뛰어다닙니다. 개미는 땅 위와 땅속을 오가며 살고 다리가 세 쌍이 있으며, 지렁이는 땅속에서 살고 다리가 없어서 기어 다닙니다. 그래서 정답은 정우"
        
        "문제 : 하늘에서 내리는 빗물과 운동장에 흐르는 빗물 중 거름 장치로 걸렀을 때 거름종이에 남은 물질이 있는 것은 무엇인지 쓰시오.,",
        "정답 : 하늘에서 내리는 빗물과 운동장에 흐르는 빗물을 거름 장치로 거르는 실험에 대해 알아보는 문제입니다. 하늘에서 내리는 빗물은 거름종이에 걸리진 것의 거의 없지만, 운동장에 흐르는 빗물은 거름종이에 진흙과 매우 작은 모래 알갱이가 남아 있습니다. 그래서 정답은 운동장에 흐르는 빗물"
        
        "문제 : 지구처럼 태양 주위를 돌고 있는 천체를 무엇이라고 하는지 쓰시오.,",
        "정답 : 행성이란 무엇인지에 대해 알아보는 문제입니다. 지구처럼 태양의 주위를 돌고 있는 천체를 행성이라고 합니다. 태양계 행성에는 수성, 금성, 지구, 화성, 목성, 토성, 천왕성, 해왕성이 있습니다. 그래서 정답은 행성"
        
        "문제": 동전 모양 자석의 윗면에 막대자석의 N극을 가까이 했더니 동전 모양 자석이 막대자석에 붙었습니다. 동전 모양 자석의 윗면은 무슨 극인지 한 글자로 쓰시오. 극,",
        "정답: 동전 모양 자석과 막대자석이 서로 끌어당겼으므로 서로 다른 극을 가까이 한 것입니다. 따라서 동전 모양 자석의 윗면은 S극입니다. 그래서 정답은 S"
        
        """,
        systems=[
            "호기심 많은 유저와 어시스턴트 간의 대화. 어시스턴트는 유저의 질문이나 지시에 도움이 되고 상세하며 정중한 답변을 합니다.",
            "유저와 어시스턴트 간의 상호작용. 어시스턴트는 유저의 질문과 지시를 세심하게 처리하며 정중하게 응답합니다.",
            "호기심 있는 유저와 어시스턴트 간의 상호작용. 어시스턴트는 유저의 질문과 지시를 세심하게 대응하며 공손한 답변을 제공합니다.",
            "유저와 어시스턴트 간의 대화. 호기심을 가진 유저의 질문과 지시를 받아들이고, 상세하고 예의 바른 방식으로 응답합니다."
        ],
        roles=("문제", "정답"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="<|endoftext|>",
    )
)

# ElementGPT template for short answer
register_conv_template(
    Conversation(
        name="elementgpt_for_society_short",
        system=
        """
        문제와 정답간의 대화. 정답은 문제에 알맞은 답변을 합니다.  
          
        "문제 : 옛날 사람들이 했던 일과 오늘날 바뀐 모습에 대해 바르게 설명하지 못한 어린이는 누구인지 쓰시오.정수: 옛날에 했던 일과 도구는 모두 사라졌어.현민: 옛날에 쓰던 도구와 오늘날 쓰는 도구가 달라졌어.광현: 옛날에 일하는 모습이 오늘날에는 새로운 모습으로 바뀌었어.",
        "정답 : 옛날에 했던 일과 오늘날 하는 일을 비교해 보면 옛날에 쓰던 도구와 오늘날 쓰는 도구가 많이 달라지고, 옛날에 일하는 모습이 오늘날에는 새로운 모습으로 변화되었음을 확인할 수 있습니다. 하지만 그림만 보고 옛날에 했던 일과 도구가 모두 사라졌다고 말 할 수는 없습니다. 그래서 정답은 정수."
        
        "문제 : 지역 주민들과 그들이 뽑은 대표들이 지역의 일을 스스로 결정하고 처리하는 것을 무엇이라 하는지 쓰시오., ",
        "정답 : 지방의 행정을 지방 주민이 선출한 기관이 처리하는 제도를 지방 자치라고 합니다. 지역의 일을 가장 잘 아는 것은 그 지역에서 생활을 하는 사람들이므로 지역 주민들이 그 지역의 일을 처리해 나갈 때 더 좋은 결과를 가져오게 됩니다. 이것이 지방 자치의 좋은 점입니다. 그래서 정답은 지방.",
        
        "문제 : 지방 의회에서 예산과 관련하여 하는 일로 옳은 것을 고르시오. ㄱ. 학생들을 더 잘 가르칠 수 있도록 교사들을 지원한다. ㄴ. 지역의 살림살이에 필요한 예산을 꼼꼼하게 검토하고 결정한다. ㄷ. 시ㆍ도청과 교육청에서 예산을 계획에 맞게 사용하였는지 확인하고 점검한다. ,, ",
        "정답 : 지방 의회에서는 예산을 검토한 후 결정하고, 후에 지방 정부가 예산을 잘 사용했는지 점검합니다. ㄱ. 학생들을 더 잘 가르칠 수 있도록 교사들을 지원하는 곳은 학교와 관련된 일을 맡아보는 곳인 교육청에서 하는 일입니다. 그래서 정답은 ㄴ.",
        
        "문제 : 우리 고장 중심지가 앞으로 어떻게 바뀌면 좋을지 바르게 말한 학생은 누구인지 쓰시오. 서영: 높은 건물과 아파트만 더 지었으면 좋겠어.선우: 교통이 혼잡하므로 도로가 더 넓어졌으면 좋겠어.민혜: 산책로를 모두 없애고 차가 다닐 수 있는 도로를 만들었으면 좋겠어., ",
        "정답 : 우리 고장 사람들이 바라는 중심지의 미래 모습을 알고 있는지 묻는 문제입니다. 고장 사람들이 바라는 미래의 중심지 모습은 높은 건물과 아파트도 있지만 낮은 건물과 낮은 집들도 있어 조화를 이루고, 산책로와 공간이 있어 사람들이 편히 쉴 수 있는 곳입니다. 그래서 정답은 선우.",
        
        "문제 : ‘서빙고동’이라는 지명을 통해 알 수 있는 사실을 고르시오. ㄱ. 이곳에 얼음을 저장하는 창고가 있었다.ㄴ. 옛날 사람들은 얼음을 창고에 저장했다.ㄷ. 옛날에는 집집마다 얼음 창고가 있었다. ㄹ. 옛날에는 남자들만 얼음을 사용할 수 있었다. ㅁ. 옛날에는 겨울에만 얼음을 사용할 수 있었다. ,, ",
        "정답 : 서빙고동은 서울특별시 용산구에 있는 곳입니다. 옛날에는 냉장고가 없었기 때문에 겨울에 강에서 얼음을 가져와 창고에 저장하였다가 여름에 사용하였습니다. 이때 얼음을 넣어 두는 창고를 ‘빙고’라고 하였습니다. 서빙고동이라는 지명을 통해서 옛날 이 지역에는 얼음을 저장하는 창고가 있었다는 사실과, 옛날 사람들이 여름에 얼음을 어떻게 구했는지 등을 알 수 있습니다. ㅁ. 옛날에는 냉장고가 없었기 때문에 얼음을 얼릴 수 없었습니다. 그래서 겨울에 강이 얼면 얼음을 잘라다가 빙고에 보관하였습니다. 얼음을 빙고에 저장한 이유는 여름에 얼음을 사용하기 위해서입니다.그래서 정답은 ㄱ.",
        
        "문제 :   에 공통으로 들어갈 알맞은 말을 두 글자로 쓰시오. 지역은 사람들이 살아가는 곳이므로 지명에는 사람들의   모습이 담겨있습니다. 사람들이 이용한 교통・통신 수단과 만든 물건, 사람들의   방식 등은 지명에 영향을 미칩니다., ",
        "정답 : 고장의 지명에는 지역의 자연환경, 교통ㆍ통신 수단과 옛날 사람들이 만든 물건, 생활 방식 등이 담겨 있습니다. 따라서 고장의 지명에 담긴 뜻을 찾아보면 옛날 사람들의 생활 모습을 짐작할 수 있습니다. 그래서 정답은 생활.",
        
        "문제 : 설명에 공통으로 들어가는 알맞은 말을 네 글자로 쓰시오. (    )은 생활에 필요한 것들을 만들거나 팔고, 그것을 사는 것과 관련된 모든 일이다. (    )을 하면서 사람들은 선택의 문제에 부딪히게 된다., ",
        "정답 : 경제 활동은 생활에 필요한 것들을 만들거나 팔고, 그것을 사는 것과 관련된 모든 일을 뜻하는 말입니다. 사람들은 이러한 경제 활동을 하면서 선택의 문제에 부딪힙니다. 그래서 정답은 경제.",

        "문제 : 지도는 다른 지역에서 생산하여 부산으로 들어오는 물건을 나타낸 것입니다. 이 물건의 공통점을 바르게 말한 사람의 이름을 쓰시오. ▲ 다른 지역에서 부산으로 들어오는 물건ㆍ수영: 부산에서 생산되지 않는 물건이야. ㆍ지민: 부산에서 주로 생산되는 물건이야. ㆍ효민: 다른 지역에서는 생산하지 않는 물건이야. ㆍ유리: 다른 지역에서는 전혀 필요하지 않은 물건이야., ",
        "정답 : 그림은 다른 지역에서 생산된 물건이 부산으로 이동하는 모습을 보여 줍니다. 이와 같이 다른 지역의 물건이 부산으로 이동하는 까닭은 이 물건들이 부산에서는 많이 생산되지 않는 것이기 때문입니다. 지역에서 많이 생산되는 물건이 다른 지역으로 이동하는 까닭은 지역에서 필요로 하는 양보다 더 많은 양이 생산되기 때문입니다. 따라서 유리가 말한 것처럼, 다른 지역에서는 전혀 필요하지 않기 때문에 물건이 이동하는 것은 아닙니다. 그래서 정답은 수영.",
        
        "문제 : 중국과 일본에서 볼 수 있는 인사 모습에 공통적으로 영향을 끼친 것은 무슨 문화인지 두 글자로 쓰시오.• 중국에서는 처음 만나는 사람에게 “니하오마”라고 말하며 허리를 굽혀 인사한다.• 일본에서는 아침, 점심, 저녁에 하는 인사말이 다르다. 아침에는 “오하요고자이마스”, 점심에는 “곤니치와”, 저녁에는 “곤반와”라고 말한다. 인사를 할 때는 양손을 모으고 허리를 굽힌다.,",
        "정답 : 중국은 “니하오마”라고 말하며 허리를 굽혀 인사합니다. 일본은 아침, 점심, 저녁에 하는 인사말이 다르며, 양손을 모으고 허리를 굽혀 인사를 합니다. 우리나라를 비롯한 중국, 일본에서 허리를 굽혀 공손히 인사하는 모습이 비슷한 까닭은 세 나라 모두 유교의 영향을 받았기 때문입니다. 그래서 정답은 유교"
        
        "문제 : 들이 펼쳐진 고장에서 사람들이 할 수 있는 여가 활동에 대하여 바르게 말한 학생의 이름을 쓰시오.•동하: 낚시를 해요.•현주: 패러글라이딩을 해요.•민아: 휴양림에서 휴식을 해요.•유나: 주말농장에서 체험을 해요.,",
        "정답 :들이 펼쳐진 고장에서는 사람들이 주말농장 체험, 하천 주변 공원에서 자전거 타기 등 다양한 여가 활동을 할 수 있습니다. 낚시는 들이 펼쳐진 고장에서 할 수 없고, 패러글라이딩과 휴양림에서의 휴식은 산이 많은 고장에서 할 수 있는 여가 활동입니다. 그래서 정답은 유나"
                                     
        """,
        systems=[
            "호기심 많은 유저와 어시스턴트 간의 대화. 어시스턴트는 유저의 질문이나 지시에 도움이 되고 상세하며 정중한 답변을 합니다.",
            "유저와 어시스턴트 간의 상호작용. 어시스턴트는 유저의 질문과 지시를 세심하게 처리하며 정중하게 응답합니다.",
            "호기심 있는 유저와 어시스턴트 간의 상호작용. 어시스턴트는 유저의 질문과 지시를 세심하게 대응하며 공손한 답변을 제공합니다.",
            "유저와 어시스턴트 간의 대화. 호기심을 가진 유저의 질문과 지시를 받아들이고, 상세하고 예의 바른 방식으로 응답합니다."
        ],
        roles=("문제", "정답"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="<|endoftext|>",
    )
)

# ElementGPT template for short answer
register_conv_template(
    Conversation(
        name="elementgpt_for_math_short",
        
        system=
        """
        문제와 정답간의 대화. 정답은 문제에 알맞은 답변을 합니다.
        
        "문제 : 못 9000개를 한 상자에 1000개씩 담으려고 합니다. 못을 모두 담으려면 상자는 몇 개 필요합니까?,",
        "정답 : 9000은 1000이 9인 수이므로 상자는 9개 필요합니다. 따라서 정답은 9"
          
        "문제 : 어떤 수에서 10씩 4번 뛰어서 센 수가 4726이었습니다. 어떤 수는 얼마입니까?,",
        "정답 : 4726에서 거꾸로 10씩 4번 뛰어서 세면 4726-4716-4706-4696-4686이므로 어떤 수는 4686입니다. 따라서 정답은 4686"

        "문제 : 한준이는 100원짜리 동전 6개와 10원짜리 동전 10개를 가지고 있습니다. 1000원이 되려면 얼마가 더 있어야 합니까?,",
        "정답 : 10원짜리 동전 10개는 100원이므로 100원짜리 동전 1개와 같습니다. 한준이가 가지고 있는 돈은 100원짜리 동전 7개와 같으므로 700원입니다.1000은 700보다 300 큰 수이므로 1000원이 되려면 300원이 더 있어야 합니다. 따라서 정답은 300"
 
        "문제 : 장미를 한 다발에 100송이씩 묶었더니 60다발이 되었습니다. 묶은 장미는 모두 몇 송이입니까?,",
        "정답 : 100이 60이면 6000이므로 묶은 장미는 모두 6000송이입니다. 따라서 정답은 6000"

        "문제 : 성렬이는 1000원짜리 지폐 3장을 가지고 있고, 민성이는 100원짜리 동전 40개를 가지고 있습니다. 누가 돈을 더 많이 가지고 있습니까?,",
        "정답 : 성렬이는 1000이 3이므로 3000원을 가지고 있고, 민성이는 100이 40이므로 4000원을 가지고 있습니다. 따라서 돈을 더 많이 가지고 있는 사람은 민성입니다. 따라서 정답은 민성"

        "문제 : 5478에서 10씩 5번 뛰어서 센 수를 구하시오.",
        "해설 : 5478+(50) = 5528 따라서 정답은 5528"
        
        "문제 : 4에 어떤 수를 곱하였더니 12가 되었습니다. 어떤 수와 8의 곱은 얼마입니까?",
        "정답 : 어떤 수를 □라고 하면 4×□=12, □=3입니다.→ 3×8=24 따라서 정답은 24"
        
        "문제 : 한 봉지에 8개씩 들어 있는 구슬이 7봉지 있습니다. 이 구슬을 한 사람에게 7개씩 나누어 준다면 몇 명에게 줄 수 있습니까?,",
        "정답 : 구슬의 수는 8×7=56(개)입니다. 곱하는 두 수의 순서를 바꾸어도 결과는 같습니다. 8×7=7×8이므로 한 사람에 8개씩 8명에게 나누어 줄 수 있습니다. 따라서 정답은 8"
        
        "문제 : 직사각형 모양 연못의 가로는 2 m 34 cm이고, 세로는 243 cm입니다. 이 연못의 어느 쪽의 길이가 더 깁니까?,",
        "정답 : 연못의 세로는 243 cm=2 m 43 cm이므로 연못의 가로 2 m 34 cm와 연못의 세로 2 m 43 cm를 비교하면 2 m 34 cm<2 m 43 cm입니다. 따라서 연못의 세로 길이가 더 깁니다. 따라서 정답은 세로"
        
        "문제 : 재선이네 아파트 정문에서 입구까지의 거리는 48 m 92 cm입니다. 재선이가 25 m 37 cm를 걸어왔다면 몇 m 몇 cm를 더 가야 아파트에 도착합니까?,",
        "정답 : (더 걸어야 하는 거리)=(전체 거리)-(걸어온 거리)48 m 92 cm-25 m 37 cm=23 m 55 cm 따라서 정답은 23 m 55 cm"
        
        "문제 : 수하는 여름 방학 동안 동화책을 매일 47쪽씩 읽으려고 합니다.7월과 8월 두 달 동안에는 동화책을 모두 몇 쪽 읽게 됩니까?,",
        "해설 : 7월은 31일, 8월도 31일이므로 하루 47쪽씩 62일을 읽게 됩니다.따라서 동화책을 모두 47×62=2914(쪽)을 읽게 됩니다. 따라서 정답은 2914쪽"
        
        "문제 : 승현이는 한 변의 길이가 25 cm인 정삼각형을 변끼리 겹치지 않도록 30개 그렸습니다. 승현이가 그린 정삼각형의 모든 변의 길이의 합은 몇 cm입니까?,",
        "정답 : 승현이가 그린 한 개의 정삼각형의 세 변의 길이의 합은 25×3=75 (cm)입니다.따라서 승현이가 그린 정삼각형 30개의 모든 변의 길이의 합은 75×30=2250 (cm)입니다. 따라서 정답은 2250 cm"
        
        "문제 : 태현이가 어떤 건물의 엘리베이터를 탔는데 이 엘리베이터는 1초에 3 m를 올라간다고 합니다. 목적지까지 올라가는데 28초가 걸렸다면 목적지의 높이는 몇 m입니까?,",
        "정답 : (목적지의 높이)=(엘리베이터가 1초에 올라가는 높이) ×(목적지까지 올라가는데 걸린 시간)=3×28=84(m) 따라서 정답은 84 m"
        
        """,
        systems=[
            "호기심 많은 유저와 어시스턴트 간의 대화. 어시스턴트는 유저의 질문이나 지시에 도움이 되고 상세하며 정중한 답변을 합니다.",
            "유저와 어시스턴트 간의 상호작용. 어시스턴트는 유저의 질문과 지시를 세심하게 처리하며 정중하게 응답합니다.",
            "호기심 있는 유저와 어시스턴트 간의 상호작용. 어시스턴트는 유저의 질문과 지시를 세심하게 대응하며 공손한 답변을 제공합니다.",
            "유저와 어시스턴트 간의 대화. 호기심을 가진 유저의 질문과 지시를 받아들이고, 상세하고 예의 바른 방식으로 응답합니다."
        ],
        roles=("문제", "정답"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="<|endoftext|>",
    )
)


# ElementGPT template for short answer
register_conv_template(
    Conversation(
        name="elementgpt_for_korean_short",
        
        system=
        """
        문제와 정답간의 대화. 정답은 문제에 알맞은 답변을 합니다.
        
        "문제 : 모음자에 대해 잘못 말하고 있는 사람의 이름을 쓰시오.영주: 혼자서 소리를 낼 수 있어. 수희: 모음자에는 ‘ㄱ’, ‘ㄴ’, ‘ㄷ’ 등이 있어.민우: ‘아’, ‘야’, ‘어’, ‘여’는 모음자의 이름이야., ",
        "정답 : 모음자는 자음자와는 달리 혼자서 소리를 낼 수 있습니다. 그리고 모음자의 이름은 ‘아, 야, 어, 여’ 등입니다. ‘ㄱ, ㄴ, ㄷ’은 자음자입니다. 따라서 정답은 수희"
          
        "문제 : 이어 주는 말을 적절하게 사용하지 못한 친구의 이름을 쓰시오.현우: 꼭 필요한 곳에만 이어 주는 말을 사용하였어.준호: 모든 문장과 문장 사이에 이어 주는 말을 사용하였어.선미: 앞뒤 문장의 연결 관계를 잘 살펴서 이어 주는 말을 사용하였어., ",
        "정답 : 이어 주는 말을 사용할 때에는 앞뒤 문장의 연결 관계를 잘 살펴서 이어 주는 말을 알맞게 사용하여야 읽는 이가 글의 내용을 쉽게 이해할 수 있습니다. 또한 이어 주는 말을 전혀 사용하지 않거나 너무 자주 사용하면 문장이 어색해질 수 있으므로, 꼭 필요한 곳에만 적절하게 사용하여야 합니다. 따라서 준호가 이어 주는 말을 적절하게 사용하지 못하였습니다. 따라서 정답은 준호"

        "문제 : 연설을 듣고 주장의 타당성을 알맞게 판단하지 못한 친구의 이름을 쓰시오.휘재: 주장의 근거가 타당한지 판단하였어.성하: 알맞은 목소리로 연설하였는지 판단하였어.민우: 주장이 실천할 수 있는 내용인지 판단하였어., ",
        "정답 : 연설을 듣고 주장의 타당성을 판단하기 위해서는 연설자의 주장이 가치 있고 중요한 주장인지, 실천할 수 있는 주장인지, 주장에 대한 근거가 적절한지 판단해야 합니다. 성하는 연설을 하는 방법과 관련 있는 내용을 말하였으므로 알맞지 않습니다. 따라서 정답은 성하"
 
        "문제 : 관용 표현을 사용하였을 때의 좋은 점을 알맞게 말하지 못한 친구의 이름을 쓰시오. 정미: 관용 표현을 사용하면 쉽게 다른 사람을 속일 수 있어. 송주: 관용 표현을 사용하면 짧은 말로 자신의 생각을 표현할 수 있어. 선비: 관용 표현을 사용하면 재미있는 표현이어서 듣는이의 관심을 불러일으킬 수 있어., ",
        "정답 : 관용 표현을 사용하면 짧은 말로 자신의 생각을 표현할 수 있고, 재미있는 표현이어서 듣는 이의 관심을 불러일으킬 수 있습니다. 또한 하고자 하는 말을 돌려서 할 수 있으므로 듣는 이의 기분을 상하지 않게 표현할 수 있습니다. 하지만 관용 표현을 사용한다고 하여 다른 사람을 쉽게 속일 수 있는 것은 아닙니다. 따라서 정답은 정미"

        "문제 : 두 낱말에 모두 들어 있는 자음자를 찾아 쓰시오. 저고리 바지, ",
        "정답 : 저고리에 들어 있는 자음자는 ‘ㅈ’, ‘ㄱ’, ‘ㄹ’입니다. 그리고 바지에 들어 있는 자음자는 ‘ㅂ’, ‘ㅈ’입니다. 따라서 두 낱말에 모두 들어 있는 자음자는 ‘ㅈ’입니다. 따라서 정답은 ㅈ"

        "문제 : 모음자가 글자의 오른쪽에 있는 낱말을 찾아 쓰시오. 우유 구두 바나나, ",
        "정답 : ‘바나나’에서 모음자 ‘ㅏ’는 자음자 ‘ㅂ’, ‘ㄴ’, ‘ㄴ’의 오른쪽에 있습니다. 따라서 모음자가 글자의 오른쪽에 있는 낱말은 ‘바나나’입니다. ‘우유’의 ‘ㅜ’와 ‘ㅠ’, ‘구두’의 ‘ㅜ’는 모두 자음자의 아래쪽에 있습니다. 따라서 정답은 바나나"
        
        "문제 : 모음자 ‘ㅙ’의 이름을 쓰시오., ",
        "정답 : 모음자 ‘ㅗ’와 ‘ㅐ’가 합쳐져 만들어진 모음자 ‘ㅙ’의 이름은 ‘왜’입니다. 따라서 정답은 왜"
        
        "문제 : ‘리’ 자로 끝나는 낱말이 아닌 것을 골라 쓰시오. 머리 나비 개구리, ",
        "정답 : ‘나비’는 ‘비’ 자로 끝나는 낱말입니다. ‘머리’와 ‘개구리’는 모두 ‘리’ 자로 끝납니다. 따라서 정답은 나비"
        
            {
        "문제 : 빈칸에 들어갈 말을 두 글자로 쓰시오. (  )은 말이나 글의 내용을 간추려 전체의 의미를 알 수 있도록 정리하는 것입니다., ",
        "정답 : 요약이란 글의 중심 내용을 간추려 정리하는 것입니다. 요약할 때에는 전체의 의미를 알 수 있도록 주요 내용을 잘 간추려야 합니다. 따라서 정답은 요약"
        
        """,
        systems=[
            "호기심 많은 유저와 어시스턴트 간의 대화. 어시스턴트는 유저의 질문이나 지시에 도움이 되고 상세하며 정중한 답변을 합니다.",
            "유저와 어시스턴트 간의 상호작용. 어시스턴트는 유저의 질문과 지시를 세심하게 처리하며 정중하게 응답합니다.",
            "호기심 있는 유저와 어시스턴트 간의 상호작용. 어시스턴트는 유저의 질문과 지시를 세심하게 대응하며 공손한 답변을 제공합니다.",
            "유저와 어시스턴트 간의 대화. 호기심을 가진 유저의 질문과 지시를 받아들이고, 상세하고 예의 바른 방식으로 응답합니다."
        ],
        roles=("문제", "정답"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="<|endoftext|>",
    )
)

# ElementGPT template for long answer
register_conv_template(
    Conversation(
        name="elementgpt_for_science_essay",
        
        system=
        """
        문제와 정답간의 대화. 정답은 문제에 알맞은 응답을 합니다.
        
        "문제 : 생활 속에서 알갱이의 크기 차이를 이용하여 혼합물을 분리하는 예를 두 가지 쓰시오.",
        "정답 : 해변 쓰레기 수거 장비로 해변에서 쓰레기를 수거한다. 강에서 모래와 진흙 속에 사는 재첩을 체를 사용하여 잡는다. 공사장에서 모래와 자갈을 체를 사용하여 분리한다. 등, 해설 : 생활 속에서 알갱이의 크기 차이를 이용하여 혼합물을 분리하는 예를 알아보는 문제입니다. 알갱이의 크기가 다른 고체 혼합물을 분리할 때 체를 사용하면 쉽게 분리할 수 있습니다. 생활 속에서 알갱이의 크기 차이를 이용하여 혼합물을 분리하는 예에는 해변 쓰레기 수거 장비로 해변에서 쓰레기 수거하기, 강 하구에서 모래와 진흙 속에 사는 재첩을 체를 사용하여 분리하기, 건물을 짓는 공사장에서 모래와 자갈을 체를 사용하여 분리하기 등이 있습니다.,  "
        
        "문제 : 나침반을 편평한 곳에 놓았을 때 나침반 바늘이 가리키는 방향과 그 방향을 가리키는 까닭을 쓰시오.",
        "정답 : 나침반 바늘은 항상 북쪽과 남쪽을 가리키는데 나침반 바늘도 자석이기 때문이다., 해설 : 나침반 바늘이 가리키는 방향과 그 까닭을 알아보는 문제입니다. 나침반은 자석이 항상 북쪽과 남쪽을 가리키는 성질을 이용하여 방향을 찾을 수 있도록 만든 도구입니다. 나침반 바늘이 자석이기 때문에 나침반 바늘이 일정한 방향을 가리킵니다. 나침반 바늘의 N극은 북쪽을 가리키며, 빨간색이나 화살표로 표시되어 있습니다.,  "
        
        "문제 : 달에서 생물이 살 수 없는 까닭을 두 가지 쓰시오.",
        "정답 : 물과 공기가 없기 때문이다. 생물이 살기에 알맞은 온도가 아니기 때문이다. 등, 해설 : 달에서 생물이 살 수 없는 까닭을 알아보는 문제입니다. 달에는 물과 공기가 없어서 생물이 살 수 없습니다. 또 지구와 다르게 달은 생물이 살기에 알맞은 온도가 아닙니다.,  "
        
        "문제 : 한해살이 식물과 여러해살이 식물의 공통점과 차이점을 각각 쓰시오.",
        "정답 : (1) 씨가 싹 터서 자라 꽃이 피고 열매를 맺어 번식한다. (2) 한해살이 식물은 열매를 맺고 죽지만 여러해살이 식물은 죽지 않고 여러 해 동안 열매를 맺는 것을 반복한다., 해설 : 한해살이 식물과 여러해살이 식물의 공통점과 차이점을 알아보는 문제입니다. 한해살이 식물은 한 해 동안 한살이를 거치고 일생을 마치는 식물이고, 여러해살이 식물은 여러 해 동안 살면서 한살이의 일부를 반복하는 식물입니다. 한해살이 식물과 여러해살이 식물은 모두 씨가 싹 터서 자라며 꽃이 피고 열매를 맺어 번식합니다. 한해살이 식물은 열매를 맺고 한 해만 살고 죽지만 여러해살이 식물은 여러 해를 살면서 열매 맺는 것을 반복합니다.,  "
        
        "문제 : 물이 얼 때와 얼음이 녹을 때의 부피와 무게의 변화를 각각 쓰시오.",
        "정답 :"(1) 물이 얼면 부피는 늘어나지만 무게는 변하지 않는다. (2) 얼음이 녹으면 부피는 줄어들지만 무게는 변하지 않는다., 해설 : 물이 얼 때와 얼음이 녹을 때의 부피와 무게 변화를 알아보는 문제입니다. 물이 얼면 부피는 늘어나지만 무게는 변하지 않고, 얼음이 녹으면 부피는 줄어들지만 무게는 변하지 않습니다. 물과 얼음의 상태가 변할 때 부피는 변하지만 무게는 변하지 않습니다.,  "

        "문제 : 생산자인 식물이 사라진다면 생태계에는 어떤 일이 일어날지 바르게 말한 학생의 이름을 쓰시오.세희: 식물을 먹는 소비자는 결국 죽게 될 거야.진우: 우리 주변이 죽은 생물과 생물의 배출물로 가득 차게 될 거야.민수: 생태계에 다양한 생물이 서로 조화를 이루며 살아가게 될 거야., ",
        "정답 : 생산자인 식물이 사라진다면 식물을 먹는 소비자는 먹이가 사라지므로 결국 죽게 될 것입니다. 또 식물을 먹는 소비자를 먹이로 하는 소비자들도 죽게 될 것입니다. 결국 생태계에는 어떤 생물 요소도 살아남지 못할 것입니다. 그래서 정답은 세희"
        
        "문제 : 물의 증발과 끓음의 공통점과 차이점을 각각 쓰시오.",
        "정답 : (1) 물이 수증기로 상태가 변해 공기 중으로 흩어진다. 등 (2) 증발은 물 표면에서 천천히 일어나고, 끓음은 물 표면과 물속에서 빠르게 일어난다. 등, 해설 : 물의 증발과 끓음의 공통점과 차이점을 알아보는 문제입니다. 증발은 액체인 물이 표면에서 기체인 수증기로 상태가 변하는 현상이고, 끓음은 물의 표면뿐만 아니라 물속에서도 액체인 물이 기체인 수증기로 상태가 변하는 현상입니다. 증발과 끓음 모두 물이 수증기로 상태가 변하는 현상이지만, 증발은 물 표면에서 일어나고, 끓음은 물 표면과 물속에서 일어난다는 점이 다릅니다. 물이 끓을 때에는 증발할 때보다 더 빨리 수증기로 변해 물의 양이 빠르게 줄어듭니다.,  "
        
        "문제 : 바위나 돌이 작게 부서진 알갱이와 생물이 썩어 생긴 물질들이 섞여서 흙이 됩니다. 이때 바위를 부서지게 하는 원인을 쓰시오.",
        "정답 : 바위틈에 있는 물이 얼었다 녹았다를 반복하면서 바위가 부서진다. 바위틈에서 나무뿌리가 자라면서 바위가 부서진다. 등, 해설 : 바위를 부서지게 하는 원인을 알아보는 문제입니다. 겨울에 바위틈에 있는 물이 얼었다 녹았다를 반복하면서 바위틈이 벌어져 바위가 부서지거나 바위틈에서 나무뿌리가 자라면서 바위틈이 벌어져 바위가 부서집니다.,  "
        
        "문제 : 화산 분출물 중 화산재가 주는 피해와 이로운 점을 한 가지씩 쓰시오.",
        "정답 : (1) 비행기 엔진을 망가뜨려 항공기 운항을 어렵게 한다. 물을 오염시킨다. 호흡기 질병에 걸릴 수 있다. 태양 빛을 차단해 동식물에게 피해를 주고 날씨의 변화가 나타나기도 한다. 등 (2) 땅을 기름지게 하여 농작물이 자라는 데 도움을 준다. 등, 해설 : 화산재가 주는 피해와 이로운 점을 알아보는 문제입니다. 화산 활동으로 생긴 화산재는 비행기 엔진을 망가뜨려 항공기 운항을 어렵게 하고, 물을 오염시키기도 합니다. 또 화산재가 물과 함께 흘러와 마을을 덮치면 마을에 큰 피해가 발생하며, 화산재와 화산 가스의 영향으로 호흡기 질병에 걸릴 수 있습니다. 화산재가 태양 빛을 차단해 동식물에게 피해를 주고 날씨의 변화가 나타나기도 하지만, 땅을 기름지게 하여 농작물이 자라는 데 도움을 주기도 합니다.,  "
        
        "문제 : 승강기 안에 있을 때 지진이 발생하였다면 어떻게 해야 하는지 쓰시오.",
        "정답 : 모든 층의 버튼을 눌러 가장 먼저 열리는 층에서 내려 계단을 이용하여 대피한다. 등, 해설 : 승강기 안에 있을 때 지진의 대처 방법을 알아보는 문제입니다. 지진이 발생하면 승강기는 이용하지 않아야 하며, 승강기 안에 있을 경우 모든 층의 버튼을 눌러 가장 먼저 열리는 층에서 내린 뒤 계단을 이용하여 대피합니다.,  "
        
        """,
        systems=[
            "호기심 많은 유저와 어시스턴트 간의 대화. 어시스턴트는 유저의 질문이나 지시에 도움이 되고 상세하며 정중한 답변을 합니다.",
            "유저와 어시스턴트 간의 상호작용. 어시스턴트는 유저의 질문과 지시를 세심하게 처리하며 정중하게 응답합니다.",
            "호기심 있는 유저와 어시스턴트 간의 상호작용. 어시스턴트는 유저의 질문과 지시를 세심하게 대응하며 공손한 답변을 제공합니다.",
            "유저와 어시스턴트 간의 대화. 호기심을 가진 유저의 질문과 지시를 받아들이고, 상세하고 예의 바른 방식으로 응답합니다."
        ],
        roles=("문제", "정답"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="<|endoftext|>",
    )
)

# ElementGPT template for long answer
register_conv_template(
    Conversation(
        name="elementgpt_for_society_essay",
        
        system=
        """
        문제와 정답간의 대화. 정답은 문제에 알맞은 답변을 합니다.
        
        "문제":"지도에서 땅의 높낮이를 나타내는 방법을 쓰시오, ",
        "정답":"지도에서는 등고선과 색깔로 땅의 높낮이를 나타낸다., 해설 : 지도에서 땅의 높낮이를 나타내는 방법을 알아보는 문제입니다. 등고선은 지도에서 높이가 같은 곳을 연결하여 땅의 높낮이를 나타낸 선입니다. 지도에서는 등고선과 색깔을 이용하여 땅의 높낮이를 나타내는데, 땅의 높이가 높을수록 색이 진해 집니다."
        
        "문제":"세계 지도의 장점과 단점은 무엇인지 쓰시오, ",
        "정답":"(1) 한눈에 전 세계를 파악할 수 있다.(2) 대륙이나 바다의 모양, 거리가 실제와 달라진다. 세계 지도의 장점과 단점을 아는지 묻는 문제입니다., 해설 : 세계 지도는 둥근 지구 표면의 모습을 평면으로 나타낸 것으로 한눈에 전 세계를 파악할 수 있지만, 대륙이나 바다의 모양, 거리가 실제와 달라진다는 단점이 있습니다. 반면 지구본은 비교적 정확하게 세계 여러 나라의 위치, 거리, 면적 등을 표현하고 있지만, 한눈에 전 세계를 보기 어렵다는 단점이 있습니다."
        
        "문제":"중국, 일본, 러시아는 이웃 나라이지만 우리나라의 문화와 다른 점이 많습니다. 그 까닭을 쓰시오, ",
        "정답":"자연환경에 따라 그 나라의 독특한 문화가 만들어지기 때문이다. 사람들의 사고방식에 따라 그 나라의 독특한 문화가 만들어지기 때문이다. 등 이웃 나라와 우리나라의 문화가 다른 까닭을 아는지 묻는 문제입니다., 해설 : 우리나라, 중국, 일본, 러시아는 이웃 나라이지만 각 나라마다 자연환경이 다르고, 사람들의 사고방식 등에 따라 그 나라의 독특한 문화가 만들어지기 때문에 나라마다 문화적으로 다른 점이 많습니다."
        
        "문제":"원 간섭기 고려의 모습을 두 가지 이상 쓰시오, ",
        "정답":"원은 고려에 그들의 관청을 설치하였다. 고려의 왕자를 인질로 삼아 원으로 데려갔다. 고려의 정치에 간섭하였다. 경제적으로 수탈을 일삼았다. 고려의 왕을 자기 마음대로 바꾸기도 하였다. 원은 고려에 쌍성총관부 등을 설치하여 고려 영토의 일부를 직접 통치하였다. 등 원 간섭기 고려의 모습을 아는지 묻는 문제입니다., 해설 : 원 간섭기에 고려의 모습은 다음과 같습니다. 우선 원은 정동행성과 같은 그들의 관청을 고려에 설치하여 내정을 간섭하고 경제적 수탈을 일삼았습니다. 그리고 고려의 왕자를 인질로 잡아 원으로 데려갔으며, 원의 공주와 결혼시켰습니다. 또한, 고려의 왕을 자기들 마음대로 바꾸는 등 고려의 정치에 간섭하였으며, 고려에 동녕부, 쌍성총관부, 탐라총관부 등을 설치하여 고려 영토의 일부를 직접 통치하였습니다."
        
        "문제":"수와 당을 물리친 고구려가 한반도에 어떤 역할을 했는지 쓰고, 고구려가 수와 당을 물리칠 수 있었던 힘을 쓰시오, ",
        "정답":"(1) 고구려가 한반도를 지키는 방파제 역할을 하였다. (2) 나라를 지키려는 정신력과 의지, 지형과 날씨, 성을 활용한 전술, 뛰어난 전투 능력, 만반의 전투 준비 등 수와 당을 물리친 고구려가 한반도에 어떤 역할을 하였는지와 고구려가 수와 당을 물리칠 수 있었던 힘이 무엇인지를 묻는 문제입니다., 해설 : 고구려는 나라를 지키려는 정신력과 의지, 지형과 날씨, 성을 활용한 전술, 뛰어난 전투 능력, 만반의 전투 준비를 바탕으로 수와 당의 침략을 모두 막아 내어 한반도를 지키는 방파제 역할을 하였습니다. 이처럼 수와 당의 계속된 침략을 막은 고구려이지만 계속된 전쟁으로 국력이 약해져 이후 삼국 간 경쟁에서 어려움을 겪게 됩니다."

        "문제":"에너지 절약을 위하여 대중교통을 이용하면 해결할 수 있는 문제를 두 가지 쓰시오, ",
        "정답":"배기가스가 감소된다. 연료비가 절약된다. 미세 먼지가 감소한다. 교통량이 감소한다. 등 대중교통으로 해결할 수 있는 문제를 쓸 수 있는지 묻는 문제입니다., 해설 : 에너지가 부족하면 우리 생활이 불편해지고 사람들이 살기 힘들어질 것입니다. 따라서 이와 같은 에너지 부족 문제를 해결해야 하는데 그중 하나로 승용차 대신 지하철과 같은 대중교통을 이용하는 방법이 있습니다. 대중교통을 이용한다면 배기가스가 감소하고, 연료비가 절약되며, 미세 먼지가 감소합니다. 더불어 교통량도 감소합니다."
        
        "문제":"헌법 재판소에서 하는 일을 두 가지 쓰시오, ",
        "정답":"법률이 헌법에 어긋나는지, 그렇지 않은지 판단한다. 국가가 하는 일들이 국민의 권리를 침해하는지, 그렇지 않은지 판단한다. 헌법과 법률에서 정한 공무원이 헌법이나 법률을 위반하였을 때 그 자리에서 물러나게 한다. 목적이나 활동이 민주적 기본 질서에 위배되는 정당을 해산시킨다. 국가 기관 간, 국가 기관과 지방 자치 단체 간, 지방 자치 단체 간에 다툼이 있을 때 이를 해결해 주기도 한다. 헌법 재판소에서 하는 일을 아는지 묻는 문제입니다., 해설 : 법률이 헌법에 어긋나는지, 그렇지 않은지를 판단하는 기관이 헌법 재판소입니다. 만약 법률이 헌법에 맞지 않는다는 헌법 재판소의 결정이 내려지면 그 법률은 효력을 잃게 됩니다. 이처럼 헌법 재판소는 헌법을 지키고 국민의 권리를 보호하는 여러 가지 일을 합니다."
        
        "문제":"고장의 자랑할 만한 장소를 조사하는 방법을 두 가지 쓰시오, ",
        "정답":"고장의 누리집과 고장의 관광 누리집에서 찾아본다. 고장 안내 책자에서 찾아본다. 고장을 잘 알고 있는 어른들께 직접 여쭤본다. 등, 해설 : 고장의 자랑할 만한 장소를 조사하는 방법을 알아보는 문제입니다. 고장의 자랑할 만한 장소를 조사하기 위해서는 고장의 누리집, 고장의 관광 누리집이나 고장 안내 책자에서 찾아봅니다. 또한, 고장을 잘 알고 있는 어른들께 직접 여쭤봅니다."
        
        """,
        systems=[
            "호기심 많은 유저와 어시스턴트 간의 대화. 어시스턴트는 유저의 질문이나 지시에 도움이 되고 상세하며 정중한 답변을 합니다.",
            "유저와 어시스턴트 간의 상호작용. 어시스턴트는 유저의 질문과 지시를 세심하게 처리하며 정중하게 응답합니다.",
            "호기심 있는 유저와 어시스턴트 간의 상호작용. 어시스턴트는 유저의 질문과 지시를 세심하게 대응하며 공손한 답변을 제공합니다.",
            "유저와 어시스턴트 간의 대화. 호기심을 가진 유저의 질문과 지시를 받아들이고, 상세하고 예의 바른 방식으로 응답합니다."
        ],
        roles=("문제", "정답"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n\n",
        sep2="<|endoftext|>",
    )
)


# Vicuna v1.1 template
register_conv_template(
    Conversation(
        name="vicuna_v1.1",
        systems=[],
        system="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Koala default template
register_conv_template(
    Conversation(
        systems=[],
        name="koala_v1",
        system="BEGINNING OF CONVERSATION:",
        roles=("USER", "GPT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Alpaca default template
register_conv_template(
    Conversation(
        name="koalpaca",
        system="",
        systems=[],
        roles=("### 질문", "### 답변"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
        sep2="<|endoftext|>",
    )
)

# Dolly V2 default template
register_conv_template(
    Conversation(
        systems=[],
        name="dolly_v2",
        system="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.DOLLY,
        sep="\n\n",
        sep2="### End",
    )
)

# OpenAssistant Pythia default template
register_conv_template(
    Conversation(
        systems=[],
        name="oasst_pythia",
        system="",
        roles=("<|prompter|>", "<|assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="<|endoftext|>",
    )
)

# StableLM Alpha default template
register_conv_template(
    Conversation(
        systems=[],
        name="stablelm",
        system="""<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
""",
        roles=("<|USER|>", "<|ASSISTANT|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[50278, 50279, 50277, 1, 0],
    )
)

# Baize default template
register_conv_template(
    Conversation(
        systems=[],
        name="baize",
        system="The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.\n",
        roles=("[|Human|]", "[|AI|]"),
        messages=(
            ("[|Human|]", "Hello!"),
            ("[|AI|]", "Hi!"),
        ),
        offset=2,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
        stop_str="[|Human|]",
    )
)

# RWKV-4-Raven default template
register_conv_template(
    Conversation(
        systems=[],
        name="rwkv",
        system="",
        roles=("Bob", "Alice"),
        messages=(
            ("Bob", "hi"),
            (
                "Alice",
                "Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.RWKV,
        sep="",
        stop_str="\n\n",
    )
)

# Buddy default template
register_conv_template(
    Conversation(
        systems=[],
        name="openbuddy",
        system="""Consider a conversation between User (a human) and Assistant (named Buddy).
Buddy is an INTP-T, a friendly, intelligent and multilingual AI assistant, by OpenBuddy team. GitHub: https://github.com/OpenBuddy/OpenBuddy
Buddy cannot access the Internet.
Buddy can fluently speak the user's language (e.g. English, Chinese).
Buddy can generate poems, stories, code, essays, songs, parodies, and more.
Buddy possesses vast knowledge about the world, history, and culture.
Buddy's responses are always safe, creative, high-quality, human-like, and interesting.
Buddy strictly refuses to discuss political, NSFW, or other unsafe topics.

User: Hi.
Assistant: Hi, I'm Buddy, your AI assistant. How can I help you today?""",
        roles=("User", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    )
)

# Phoenix default template
register_conv_template(
    Conversation(
        systems=[],
        name="phoenix",
        system="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.PHOENIX,
        sep="</s>",
    )
)

# ChatGPT default template
register_conv_template(
    Conversation(
        name="chatgpt",
        systems=[],
        system="You are a helpful assistant.",
        roles=("user", "assistant"),
        messages=(),
        offset=0,
        sep_style=None,
        sep=None,
    )
)

# Claude default template
register_conv_template(
    Conversation(
        name="claude",
        system="",
        systems=[],
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
    )
)

# MPT default template
register_conv_template(
    Conversation(
        name="mpt",
        systems=[],
        system="""<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.
""",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="<|im_end|>",
        stop_token_ids=[50278, 0],
    )
)

# Bard default template
# Reference: https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L150
#            https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L40
register_conv_template(
    Conversation(
        systems=[],
        name="bard",
        system="",
        roles=("0", "1"),
        messages=(),
        offset=0,
        sep_style=None,
        sep=None,
    )
)

# BiLLa default template
register_conv_template(
    Conversation(
        systems=[],
        name="billa",
        system="",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n",
        stop_str="Human:",
    )
)

# RedPajama INCITE default template
register_conv_template(
    Conversation(
        systems=[],
        name="redpajama-incite",
        system="",
        roles=("<human>", "<bot>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="<human>",
    )
)

# h2oGPT default template
register_conv_template(
    Conversation(
        name="h2ogpt",
        system="",
        systems=[],
        roles=("<|prompt|>", "<|answer|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="</s>",
    )
)

# Snoozy default template
# Reference: https://github.com/nomic-ai/gpt4all/blob/d4861030b778da6db59d21d2927a4aba4f9f1f43/gpt4all-bindings/python/gpt4all/gpt4all.py#L232
register_conv_template(
    Conversation(
        systems=[],
        name="snoozy",
        system="### Instruction:\nThe prompt below is a question to answer, a task to complete, or a conversation to respond to; decide which and write an appropriate response.",
        roles=("### Prompt", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="###",
    )
)

# manticore default template
register_conv_template(
    Conversation(
        name="manticore",
        systems=[],
        system="",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="</s>",
    )
)


if __name__ == "__main__":
    conv = get_conv_template("vicuna_v1.1")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())