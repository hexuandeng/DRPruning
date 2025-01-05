from lm_eval.utils import general_detokenize


def default_doc_to_text(x):
    raw_passage = x["text"]
    # NOTE: HuggingFace span indices are word-based not character-based.
    pre = " ".join(raw_passage.split()[: x["span2_index"]])
    post = raw_passage[len(pre) + len(x["span2_text"]) + 1 :]
    passage = general_detokenize(pre + " *{}*".format(x["span2_text"]) + post)
    noun = x["span1_text"]
    pronoun = x["span2_text"]
    text = (
        f"Passage: {passage}\n"
        + f'Question: In the passage above, does the pronoun "*{pronoun}*" refer to "*{noun}*"?\n'
        + "Answer:"
    )
    return text

def doc_to_text1(x):
    raw_passage = x["text"]
    pre = " ".join(raw_passage.split()[:x["span2_index"]])
    post = raw_passage[len(pre) + len(x["span2_text"]) + 1:]
    passage = general_detokenize(pre + " *{}*".format(x["span2_text"]) + post)
    noun = x["span1_text"]
    pronoun = x["span2_text"]
    text = (
        f"Read the following passage:\n{passage}\n"
        + f'Question: Does the pronoun "*{pronoun}*" refer to "*{noun}*"?\n'
        + "Answer:"
    )
    return text

def doc_to_text2(x):
    raw_passage = x["text"]
    pre = " ".join(raw_passage.split()[:x["span2_index"]])
    post = raw_passage[len(pre) + len(x["span2_text"]) + 1:]
    passage = general_detokenize(pre + " *{}*".format(x["span2_text"]) + post)
    noun = x["span1_text"]
    pronoun = x["span2_text"]
    text = (
        f"Examine the passage below:\n{passage}\n"
        + f'Question: Does the pronoun "*{pronoun}*" refer to "*{noun}*"?\n'
        + "Answer:"
    )
    return text

def doc_to_text3(x):
    raw_passage = x["text"]
    pre = " ".join(raw_passage.split()[:x["span2_index"]])
    post = raw_passage[len(pre) + len(x["span2_text"]) + 1:]
    passage = general_detokenize(pre + " *{}*".format(x["span2_text"]) + post)
    noun = x["span1_text"]
    pronoun = x["span2_text"]
    text = (
        f"Review the passage below:\n{passage}\n"
        + f'Question: Does the pronoun "*{pronoun}*" in the passage refer to "*{noun}*"?\n'
        + "Answer:"
    )
    return text

def doc_to_text4(x):
    raw_passage = x["text"]
    pre = " ".join(raw_passage.split()[:x["span2_index"]])
    post = raw_passage[len(pre) + len(x["span2_text"]) + 1:]
    passage = general_detokenize(pre + " *{}*".format(x["span2_text"]) + post)
    noun = x["span1_text"]
    pronoun = x["span2_text"]
    text = (
        f"Examine the following passage:\n{passage}\n"
        + f'Question: In this passage, does the pronoun "*{pronoun}*" refer to "*{noun}*"?\n'
        + "Answer:"
    )
    return text
