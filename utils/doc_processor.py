import re
import json
import copy


def remove_superfluous_char(doc):
    # Remove video tag
    doc = re.sub(r"(addLoadEvent)[^\n]+;", " ", doc)

    # Remove links
    doc = re.sub(r"((http)([s:/]{3,4})?)?[a-z0-9]+([\.][a-z0-9]+)+[\S]+", " ", doc)
    # Remove email
    doc = re.sub(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", " ", doc)
    # Remove special char
    doc = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s\?\!\.\-()$%+=\'\":]", " ", doc)
    # Remove tags
    doc = re.sub(r"<(\/)?[a-z]+>", "", doc)
    # Reduce repetition
    doc = re.sub(r"([^가-힣a-zA-Z0-9])\1{1,}", r"\1", doc)
    doc = re.sub(r"([^0-9])\1{3,}", r"\1\1\1", doc)

    # Remove chosung
    doc = re.sub(r"[ㄱ-ㅎㅏ-ㅣ]+", " ", doc)
    # Remove spaces
    doc = re.sub(r"[ ]{2,}", " ", doc)
    doc = re.sub(r"(\s\.\s)", ". ", doc)
    doc = doc.replace(".. ", "")
    doc = doc.replace(".?", "?")
    doc = doc.replace(".!", "!")

    # Remove line alignments
    doc = re.sub(r"([\s])\1{1,}", " ", doc)
    doc = re.sub(r"[\s]+", " ", doc)
    
    # Remove 휴대폰

    return doc.strip()


def main():
    with open(file="./icecream.json", mode="r") as f:
        data = json.load(f)

    preprocessed = copy.deepcopy(data)
    for i in range(len(data)):
        preprocessed[i]["contents"] = remove_superfluous_char(data[i]["contents"])

    with open(file="./preprocessed.json", mode="w") as f:
        json.dump(obj=preprocessed, fp=f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
