# coding=utf-8
# Copyright 2021 - Sia Gholami


from utils import *

# nq = Natural Questions (https://ai.google.com/research/NaturalQuestions/)
logger = gconfig.logger


class NqExample(object):
  """A single training/test example."""

  def __init__(self,
               document_full_text,
               question_text,
               document_text_with_answer=None,
               short_answers_bounds=None,
               short_answer=None,
               long_answers_bounds=None,
               long_answer=None,
               yes_no_answer=None):
    self.document_full_text = document_full_text
    self.question_text = question_text
    self.document_text_with_answer = document_text_with_answer
    self.short_answers_bounds = short_answers_bounds
    self.short_answer = short_answer
    self.long_answers_bounds = long_answers_bounds
    self.long_answer = long_answer
    self.yes_no_answer = yes_no_answer


def nq_tokenizer(in_str):
    return in_str.lower().replace("\n", " ").split()

#dw
def bounds_to_text(bounds, document, tokenizer):

    doc_tokens = tokenizer(document)

    l = []
    for bound in bounds:
        l.extend(doc_tokens[bound[0]:bound[1]])

    txt = " ".join(l)
    return txt


def get_doc_span(answer_bounds, document, tokenizer, n_leading_tokens, max_len):
    """
    get the portion of the document that contains the answer
    :param answer_bounds: 
    :param document: 
    :param tokenizer: 
    :param n_leading_tokens: 
    :param max_len: 
    :return: 
    """
    
    doc_tokens = tokenizer(document)
    
    start_idx = max(0, answer_bounds[0]-n_leading_tokens)
    end_idx = min(len(doc_tokens), start_idx + max_len)
    doc_span_tokens = doc_tokens[start_idx:end_idx]
    doc_span = " ".join(doc_span_tokens) 
    
    return doc_span

def extract_from_single_json(json_example):
    """

    :param json_example:
    :return:
    """

    document_text = json_example['document_text']
    question_text = json_example['question_text']

    if gconfig.nq_dataset == 'train':
        if len(json_example['annotations']) != 1:
            raise ValueError(
                'Train set json_examples should have a single annotation.')
        annotation = json_example['annotations'][0]
        has_long_answer = annotation['long_answer']['start_token'] >= 0
        # my code
        has_short_answer = bool(annotation['short_answers'])
        has_yes_no_answer = bool(annotation['yes_no_answer'] != 'NONE')

    long_answers = [
        a['long_answer']
        for a in json_example['annotations']
        if a['long_answer']['start_token'] >= 0 and has_long_answer
    ]
    short_answers = [
        a['short_answers']
        for a in json_example['annotations']
        if a['short_answers'] and has_short_answer
    ]
    yes_no_answers = [
        a['yes_no_answer']
        for a in json_example['annotations']
        if a['yes_no_answer'] != 'NONE' and has_yes_no_answer
    ]

    if has_short_answer:
        short_answers_ids = [[
            (s['start_token'], s['end_token']) for s in a
        ] for a in short_answers]

        short_answers_bounds = short_answers_ids[0]
        short_answer = bounds_to_text(
            short_answers_bounds, document_text, nq_tokenizer)
        document_text_with_answer = get_doc_span(
            answer_bounds=short_answers_bounds[0],
            document=document_text,
            tokenizer=nq_tokenizer,
            n_leading_tokens=gconfig.n_leading_tokens,
            max_len=gconfig.doc_max_len)
    else:
        short_answers_bounds = short_answer = document_text_with_answer = None

    if has_long_answer:
        long_answers_bounds = [
            (la['start_token'], la['end_token']) for la in long_answers
        ]
        long_answer = bounds_to_text(
            long_answers_bounds, document_text, nq_tokenizer)
    else:
        long_answers_bounds = long_answer = None

    if has_yes_no_answer:
        yes_no_answer = "yes" if yes_no_answers[0].lower() == 'yes' else "no"
    else:
        yes_no_answer = "none"

    return NqExample(
        document_full_text=document_text,
        question_text=question_text,
        document_text_with_answer=document_text_with_answer,
        short_answers_bounds=short_answers_bounds,
        short_answer=short_answer,
        long_answers_bounds=long_answers_bounds,
        long_answer=long_answer,
        yes_no_answer=yes_no_answer
    )


def load_jsonl_examples(jsonl_file, max_examples):
    """
    Reads jsonlines containing NQ examples.
    :param jsonl_file: File object containing NQ examples.
    :return: list of NQ_TEXT_Examples
    """

    def _load(examples, f):
        """Read serialized json from `f`, create examples, and add to `examples`."""

        for l in f:
            try:
                json_example = json.loads(l)
                example = extract_from_single_json(json_example)
                examples.append(example)

            except Exception as exp:
                logger.error(f"error in processing {json_example['question_text']}: {exp}", exc_info=1)

            if max_examples is not None and len(examples) == max_examples:
                break

    examples = []
    with open(jsonl_file, "r") as fileobj:
            if gconfig.gzipped:
                _load(examples, gzip.GzipFile(fileobj=fileobj))
            else:
                _load(examples, fileobj)

    return examples


def examples_to_pd(examples):
    """

    :param examples:
    :return:
    """
    df = pd.DataFrame([vars(exm) for exm in examples])

    #drop columns
    df.drop(columns=['long_answers_bounds', 'long_answer', 'document_full_text', 'short_answers_bounds'], inplace=True)

    # drop rows if no answer is provided
    df = df[df['short_answer'].notna()]

    # rename column
    df.rename(
        columns={
            "short_answer": "text_answer",
            'document_text_with_answer': 'document',
            'question_text': 'question'
        },
        errors="raise", inplace=True)
    return df


# Unit testing
def log_examples(examples):
    for exm in examples:
        logger.info(
            f'found short answer: \n'
            f'question: {exm.question_text}\n'
            f'long answer index: {exm.long_answers}\n'
            f'long answer: {bounds_to_text(exm.long_answers, exm.document_text, nq_tokenizer)}\n'
            f'short answer index: {exm.short_answers}\n'
            f'short answer: {bounds_to_text(exm.short_answers, exm.document_text, nq_tokenizer)}\n'
            f'yn: {exm.yes_no_answer}\n'
        )
    return


def test():
    d = "I have a big boat, it is six miles long. Very good boat, huge boat."
    b = (7,9)
    print(get_doc_span(b, d, nq_tokenizer, 4, 30))
    return None


def main():
    # test()

    examples = load_jsonl_examples(gconfig.nq_text_file, gconfig.nq_max_examples)
    logger.info(len(examples))

    df = examples_to_pd(examples)
    logger.info(df.info())

    df.to_pickle(gconfig.nq_extracted_file)

    return None


if __name__ == '__main__':
    main()

