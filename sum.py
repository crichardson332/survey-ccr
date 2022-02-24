# import prodigy
# from prodigy.components.preprocess import add_tokens
# import requests
# import spacy

# @prodigy.recipe("summary-error-analysis")
# def summary_error_analysis(dataset, lang="en"):
#     # We can use the blocks to override certain config and content, and set
#     # "text": None for the choice interface so it doesn't also render the text
#     blocks = [
#         # {"view_id": "ner_manual"},
#         # {"view_id": "choice", "text": None},
#         # {"view_id": "text_input", "field_rows": 3, "field_label": "Explain your decision"}
#         {"view_id": "html"},
#         {"view_id": "spans_manual"},
#     ]

#     def get_stream():
#         res = requests.get("https://cat-fact.herokuapp.com/facts").json()
#         for fact in res["all"]:
#             yield {"text": fact["text"], "options": options}

#     # Load the stream from a JSONL file and return a generator that yields a
#     # dictionary for each example in the data.
#     stream = JSONL(source)

#     # Tokenize the incoming examples and add a "tokens" property to each
#     # example. Also handles pre-defined selected spans. Tokenization allows
#     # faster highlighting, because the selection can "snap" to token boundaries.
#     stream = add_tokens(nlp, stream)

#     nlp = spacy.blank(lang)           # blank spaCy pipeline for tokenization
#     stream = get_stream()             # set up the stream
#     stream = add_tokens(nlp, stream)  # tokenize the stream for ner_manual

#     return {
#         "dataset": dataset,          # the dataset to save annotations to
#         "view_id": "blocks",         # set the view_id to "blocks"
#         "stream": stream,            # the stream of incoming examples
#         "config": {
#             "labels": ["RELEVANT"],  # the labels for the manual NER interface
#             "blocks": blocks         # add the blocks to the config
#         }
#     }



import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens
from prodigy.util import split_string
import spacy
from typing import List, Optional


# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe(
    "summary.manual",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
)
def summary_manual(
    dataset: str,
    spacy_model: str,
    source: str,
    label: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
):
    """
    Mark spans manually by token. Requires only a tokenizer and no entity
    recognizer, and doesn't do any active learning.
    """
    # Load the spaCy model for tokenization
    nlp = spacy.load(spacy_model)

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # Tokenize the incoming examples and add a "tokens" property to each
    # example. Also handles pre-defined selected spans. Tokenization allows
    # faster highlighting, because the selection can "snap" to token boundaries.
    stream = add_tokens(nlp, stream)

    # We can use the blocks to override certain config and content, and set
    # "text": None for the choice interface so it doesn't also render the text
    blocks = [
        # {"view_id": "ner_manual"},
        # {"view_id": "choice", "text": None},
        # {"view_id": "text_input", "field_rows": 3, "field_label": "Explain your decision"}
        {"view_id": "html"},
        {"view_id": "spans_manual"},
    ]

    return {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "exclude": exclude,  # List of dataset names to exclude
        "config": {  # Additional config settings, mostly for app UI
            "lang": nlp.lang,
            "labels": label,  # Selectable label options
            "blocks": blocks         # add the blocks to the config
        },
    }
