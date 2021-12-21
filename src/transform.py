# coding=utf-8
# Copyright 2020 - Sia Gholami

from utils import *

#me
#import nq_extract


logger = gconfig.logger


def make_bert_preprocess_model(
        tfhub_handle_preprocess,
        sentence_features,
        seq_length=128):
    """Returns Model mapping string features to BERT inputs.

    Args:
    sentence_features: a list with the names of string-valued features.
    seq_length: an integer that defines the sequence length of BERT inputs.

    Returns:
    A Keras Model that can be called on a list or dict of string Tensors
    (with the order or names, resp., given by sentence_features) and
    returns a dict of tensors for input to BERT.
    """

    I = [
        tf.keras.layers.Input(shape=(), dtype=tf.string, name=ft)
        for ft in sentence_features]

    # Tokenize the text to word pieces.
    bert_preprocess = hub.load(tfhub_handle_preprocess)
    tokenizer = hub.KerasLayer(bert_preprocess.tokenize, name='tokenizer')
    segments = [tokenizer(s) for s in I]

    # Optional: Trim segments in a smart way to fit seq_length.
    truncated_segments = segments

    # Pack inputs. The details (start/end token ids, dict of output tensors)
    # are model-dependent, so this gets loaded from the SavedModel.
    packer = hub.KerasLayer(
        bert_preprocess.bert_pack_inputs,
        arguments=dict(seq_length=seq_length),
        name='packer')
    Y_hat = packer(truncated_segments)
    preprocess_model = tf.keras.Model(I, Y_hat)
    return preprocess_model


#dw
def tokenize_text_to_word_id(stringList, bert_preprocess):
    """

    :param stringList:
    :param bert_preprocess:
    :return:
    """

    tok = bert_preprocess.tokenize(tf.constant(stringList))
    tok_list = tok.to_list()
    tok_list_flatten = [list(itertools.chain.from_iterable(_i)) for _i in tok_list]

    return tok_list_flatten

#dw
def encode_two_bert_inputs(stringList1, stringList2, bert_preprocess, max_seq_len):
    """
    quick and dirty way to do it. more scalable is using the model.
    :param stringList1:
    :param stringList2:
    :param bert_preprocess:
    :param max_seq_len:
    :return:
    """
    tok1 = bert_preprocess.tokenize(tf.constant(stringList1))
    tok2 = bert_preprocess.tokenize(tf.constant(stringList2))
    text_preprocessed = bert_preprocess.bert_pack_inputs([tok1, tok2], tf.constant(max_seq_len))
    return text_preprocessed


def encode_answer_tokens_to_idx(input_word_ids, text_answer_tokens, seperator=102):
    output_idx_shape = input_word_ids.shape
    start_idx = np.zeros(output_idx_shape)
    end_idx = np.zeros(output_idx_shape)

    for i in range(output_idx_shape[0]):
        wids = input_word_ids[i].tolist()
        q_len = wids.index(seperator) # BERT Seperator
        ii = [wids[q_len:].index(x)+q_len for x in text_answer_tokens[i]] # search for word idx after q

        s = [ii[0]]
        for _i in range(1, len(ii)):
            if ii[_i] != ii[_i-1]+1:
                s.append(ii[_i])

        e = [ii[-1]]
        for _i in range(-2, -len(ii)-1, -1):
            if ii[_i] != ii[_i+1]-1:
                e.insert(0, ii[_i])

        #print(s,e)
        start_idx[i][s] = 1
        end_idx[i][e] = 1

        #print(start_idx[i], end_idx[i])

    return start_idx, end_idx


def oneHot(pylist, uniq_values=None):
    if not uniq_values:
        uniq_values = np.unique(pylist).tolist()

    uniq_count = len(uniq_values)
    idx = [uniq_values.index(item) for item in pylist]
    features = tf.one_hot(idx, uniq_count)
    return features.numpy()


def df_to_features(df, preprocess_model, batch_size, verbose=1):
    input_text = [
        np.array(df['question']),
        np.array(df['document'])
    ]
    text_preprocessed = preprocess_model.predict(input_text, batch_size=batch_size, verbose=verbose)

    bert_preprocess = hub.load(gconfig.tfhub_handle_preprocess)
    text_answers = df['text_answer'].tolist()
    text_answer_tokens = tokenize_text_to_word_id(text_answers, bert_preprocess)
    start_idx, end_idx = encode_answer_tokens_to_idx(
        text_preprocessed['input_word_ids'], text_answer_tokens, seperator=102)
    yn_answer = oneHot(df['yes_no_answer'].tolist(), uniq_values=['no', 'yes', 'none'])

    text_preprocessed['text_answer_start_idx'] = start_idx
    text_preprocessed['text_answer_end_idx'] = end_idx
    text_preprocessed['yes_no_answer'] = yn_answer

    return text_preprocessed


def write_dict_to_h5(srcdict, output_file):
    with h5py.File(output_file, 'w') as h5Obj:
        for key, mat in srcdict.items():
            dataset = h5Obj.create_dataset(
                key,
                data=mat,
                dtype=mat.dtype.name,
                chunks=True)

    return dataset


# unit testing
def log_text_preprocessed(text_preprocessed, max_len=16):
    logger.info(f'Keys           : {list(text_preprocessed.keys())}')
    logger.info(f'Shape Word Ids : {text_preprocessed["input_word_ids"].shape}')
    logger.info(f'Word Ids       : {text_preprocessed["input_word_ids"][0, :max_len]}')
    logger.info(f'Shape Mask     : {text_preprocessed["input_mask"].shape}')
    logger.info(f'Input Mask     : {text_preprocessed["input_mask"][0, :max_len]}')
    logger.info(f'Shape Type Ids : {text_preprocessed["input_type_ids"].shape}')
    logger.info(f'Type Ids       : {text_preprocessed["input_type_ids"][0, :max_len]}')
    return None

def test_tokenizers():
    bert_preprocess = hub.load(gconfig.tfhub_handle_preprocess)

    d = ["I have a big boat, my boat is awesome. it is seven miles long. Very good boat, huge boat.",
         "I am a big boy, almost six feet or more tall"]
    q = ['How big is my boat?', 'how tall am I?']
    #b = [(7, 9), None]
    a = ['my boat is seven miles long', 'I am six feet tall']

    q_toks = tokenize_text_to_word_id(q, bert_preprocess)
    a_toks = tokenize_text_to_word_id(a, bert_preprocess)
    text_preprocessed = encode_two_bert_inputs(q, d, bert_preprocess, 50)

    #logger.info(f'a_toks: {a_toks}\n q_toks: {q_toks} ')
    #logger.info(f'word ids: {text_preprocessed["input_word_ids"]}')
    #log_text_preprocessed(text_preprocessed)

    output_idx_shape = text_preprocessed["input_word_ids"].shape
    start_idx = np.zeros(output_idx_shape)
    end_idx = np.zeros(output_idx_shape)


    for i in range(output_idx_shape[0]):
        wids = text_preprocessed["input_word_ids"][i].numpy().tolist()
        q_len = wids.index(102) # BERT Seperator
        ii = [wids[q_len:].index(x)+q_len for x in a_toks[i]] # search after q
        #ii = np.where(np.in1d(text_preprocessed["input_word_ids"][i], a_toks[i]))[0]
        print(ii)

        s = [ii[0]]
        for _i in range(1, len(ii)):
            if ii[_i] != ii[_i-1]+1:
                s.append(ii[_i])

        e = [ii[-1]]
        for _i in range(-2, -len(ii)-1, -1):
            if ii[_i] != ii[_i+1]-1:
                e.insert(0, ii[_i])

        print(s,e)
        start_idx[i][s] = 1
        end_idx[i][e] = 1

        print(start_idx[i], end_idx[i])
    return None


def test_preprocess_model():
    model = make_bert_preprocess_model(
        tfhub_handle_preprocess=gconfig.tfhub_handle_preprocess,
        sentence_features=['my_input1', 'my_input2'],
        seq_length=512
    )
    test_text = [np.array(['some random test sentence']),
                 np.array(['another sentence'])]
    text_preprocessed = model(test_text)

    log_text_preprocessed(text_preprocessed)
    return None


def test():
    test_tokenizers()
    #test_preprocess_model()
    return None


def main():
    # test()

    preprocess_model = make_bert_preprocess_model(
        tfhub_handle_preprocess=gconfig.tfhub_handle_preprocess,
        sentence_features=['question', 'document'],
        seq_length=gconfig.max_seq_len
    )

    for idx, f in enumerate(gconfig.to_be_transformed_files):
        df = pd.read_pickle(f)

        feature_dict = df_to_features(df, preprocess_model, gconfig.preprocess_batch_size)
        #log_text_preprocessed(feature_dict, 50)
        write_dict_to_h5(feature_dict, gconfig.to_be_trained_files[idx])

    return None


if __name__ == '__main__':
    main()

