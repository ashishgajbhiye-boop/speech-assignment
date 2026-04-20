import re

def detect_language(text: str):
    words = text.split()
    labels = []

    for w in words:
        if re.search(r'[a-zA-Z]', w):
            labels.append("en")
        else:
            labels.append("hi")

    return list(zip(words, labels))