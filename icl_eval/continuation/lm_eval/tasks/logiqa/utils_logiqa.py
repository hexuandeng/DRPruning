# Copied from Master
def doc_to_text(doc) -> str:
    """
    Passage: <passage>
    Question: <question>
    Choices:
    A. <choice1>
    B. <choice2>
    C. <choice3>
    D. <choice4>
    Answer:
    """
    choices = ["a", "b", "c", "d"]
    prompt = "Passage: " + doc["context"] + "\n"
    prompt += "Question: " + doc["question"] + "\nChoices:\n"
    for choice, option in zip(choices, doc["options"]):
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer:"
    return prompt


def doc_to_target(doc) -> int:
    choices = ["a", "b", "c", "d"]
    return choices.index(doc["label"].strip())

def doc_to_text_1(doc) -> str:
    """
    Passage: <passage>
    Question: <question>
    Possible Answers:
    A) <choice1>
    B) <choice2>
    C) <choice3>
    D) <choice4>
    Answer:
    """
    choices = ["A", "B", "C", "D"]
    prompt = f"Passage: {doc['context']}\nQuestion: {doc['question']}\nPossible Answers:\n"
    for label, option in zip(choices, doc["options"]):
        prompt += f"{label}) {option}\n"
    prompt += "Answer:"
    return prompt

def doc_to_text_2(doc) -> str:
    """
    Here is a passage:
    <passage>

    Based on the above, answer the following question:
    <question>

    Select the correct option:
    A. <choice1>
    B. <choice2>
    C. <choice3>
    D. <choice4>

    Your answer:
    """
    choices = ["A", "B", "C", "D"]
    prompt = f"Here is a passage:\n{doc['context']}\n\nBased on the above, answer the following question:\n{doc['question']}\n\nSelect the correct option:\n"
    for choice, option in zip(choices, doc["options"]):
        prompt += f"{choice}) {option}\n"
    prompt += "\nYour answer:"
    return prompt

def doc_to_text_3(doc) -> str:
    """
    Question: <question>
    Passage: <passage>
    Options:
    A) <choice1>
    B) <choice2>
    C) <choice3>
    D) <choice4>
    Answer:
    """
    choices = ["A", "B", "C", "D"]
    prompt = f"Question: {doc['question']}\nPassage: {doc['context']}\nOptions:\n"
    for choice, option in zip(choices, doc["options"]):
        prompt += f"{choice}) {option}\n"
    prompt += "Answer:"
    return prompt

def doc_to_text_4(doc) -> str:
    """
    Passage: <passage>
    Question: <question>
    Options:
    A. <choice1>
    B. <choice2>
    C. <choice3>
    D. <choice4>
    Select the correct answer (A/B/C/D):
    """
    choices = ["A", "B", "C", "D"]
    prompt = f"Passage: {doc['context']}\nQuestion: {doc['question']}\nOptions:\n"
    for choice, option in zip(choices, doc["options"]):
        prompt += f"{choice}. {option}\n"
    prompt += "Select the correct answer (A/B/C/D):"
    return prompt
