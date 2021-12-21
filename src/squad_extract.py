# coding=utf-8
# Copyright 2021 - Sia Gholami


from utils import *

# squad: https://rajpurkar.github.io/SQuAD-explorer/
logger = gconfig.logger


def squad_json_to_dataframe_train(input_file_path, record_path=['data', 'paragraphs', 'qas', 'answers'],
                                  verbose=1):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    verbose: 0 to suppress it default is 1
    """
    if verbose:
        print("Reading the json file")
    file = json.loads(open(input_file_path).read())
    if verbose:
        print("processing...")

    # parsing different level's in the json file
    js = pd.io.json.json_normalize(file, record_path)
    m = pd.io.json.json_normalize(file, record_path[:-1])
    r = pd.io.json.json_normalize(file, record_path[:-2])

    # combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    ndx = np.repeat(m['id'].values, m['answers'].str.len())
    m['context'] = idx
    js['q_idx'] = ndx
    main = pd.concat([m[['id', 'question', 'context']].set_index('id'), js.set_index('q_idx')], 1,
                     sort=False).reset_index()
    main['c_id'] = main['context'].factorize()[0]
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")
    return main


def squad_json_to_dataframe_dev(input_file_path, record_path=['data', 'paragraphs', 'qas', 'answers'],
                                verbose=1):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    verbose: 0 to suppress it default is 1
    """
    if verbose:
        print("Reading the json file")
    file = json.loads(open(input_file_path).read())
    if verbose:
        print("processing...")
    # parsing different level's in the json file
    js = pd.io.json.json_normalize(file, record_path)
    m = pd.io.json.json_normalize(file, record_path[:-1])
    r = pd.io.json.json_normalize(file, record_path[:-2])

    # combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    #     ndx  = np.repeat(m['id'].values,m['answers'].str.len())
    m['context'] = idx
    #     js['q_idx'] = ndx
    main = m[['id', 'question', 'context', 'answers']].set_index('id').reset_index()
    main['c_id'] = main['context'].factorize()[0]
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")
    return main


def normalize_squad_df(df):

    # drop columns
    df.drop(columns=['index', 'c_id', 'answer_start'], inplace=True)

    # drop rows if no answer is provided
    df = df[df['text'].notna()]

    # rename column
    df.rename(
        columns={
            "text": "text_answer",
            'context': 'document'
        },
        errors="raise", inplace=True)

    # add yes_no
    df['yes_no_answer'] = ['none'] * len(df)

    logger.info(df.info())

    return df


def main():

    df1 = squad_json_to_dataframe_train(
        input_file_path=gconfig.squad1_file,
        record_path=['data', 'paragraphs', 'qas', 'answers'],
        verbose=1)
    df1 = normalize_squad_df(df1)
    logger.info(df1.info())

    df2 = squad_json_to_dataframe_train(
        input_file_path=gconfig.squad2_file,
        record_path=['data', 'paragraphs', 'qas', 'answers'],
        verbose=1)
    df2 = normalize_squad_df(df2)
    logger.info(df2.info())

    df12 = pd.concat([df1, df2])
    logger.info(df12.info())

    # sample
    if gconfig.squad_max_examples:
        df12 = df12.sample(n=gconfig.squad_max_examples, replace=False, random_state=42)

    logger.info(df12.info())
    df12.to_pickle(gconfig.squad_extracted_file)

    return None


if __name__ == '__main__':
    main()

