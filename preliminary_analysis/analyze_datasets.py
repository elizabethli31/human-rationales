import string
import pandas as pd
import sys

sys.path.append('../')
from boolQ.make_csv_data import get_data

def get_length(text):
    remove_punc = text.translate(str.maketrans('', '', string.punctuation))

    tokens = remove_punc.split()

    return len(tokens)

def get_rationale_lengths(row):
    return get_length(row['evidences'])

def get_text_lengths(row):
    return get_length(row['text'])

def convert_df(filepath):
    df = get_data(filepath)
    df['rationale_len'] = df.apply(lambda row: get_rationale_lengths(row), axis=1)
    df['text_len'] = df.apply(lambda row: get_text_lengths(row), axis=1)

    df2 = df[['annotation_id', 'classification', 'rationale_len', 'text_len', 'query']]

    return df2

def mrl(df):
    return df['rationale_len'].mean()

def mrp(df):
    return (df['rationale_len']/df['text_len']).mean()

def mtl(df):
    return df['text_len'].mean()

def get_stats(df):
    mean_rationale_length = mrl(df)
    mean_rationale_percent = mrp(df)
    mean_text_length = mtl(df)

    df_true = df.loc[df['classification'] == 'True']
    df_false = df.loc[df['classification'] == 'False']

    mrlc = []
    mrlc.append(mrl(df_true))
    mrlc.append(mrl(df_false))

    mrpc = []
    mrpc.append(mrp(df_true))
    mrpc.append(mrp(df_false))

    mtlc = []
    mtlc.append(mtl(df_true))
    mtlc.append(mtl(df_false))

    stats = {'dataset':'BoolQ',
             'classes':['True', 'False'],
             'mean_rationale_length': mean_rationale_length,
             'mean_rationale_percent': mean_rationale_percent,
             'mean_text_length': mean_text_length,
             'mean_rationale_length_classes': mrlc,
             'mean_rationale_percent_classes': mrpc,
             'mean_text_length_classes': mtlc
    }

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv('../calculations/stats.csv')

if __name__ == '__main__':
    train = convert_df("train_data")
    test = convert_df("test_data")
    dev = convert_df("dev_data")

    df = pd.concat([train, test, dev])

    get_stats(df)

    