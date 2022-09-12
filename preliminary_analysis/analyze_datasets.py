import string
import pandas as pd

def get_length(text):
    remove_punc = text.translate(str.maketrans('', '', string.punctuation))

    tokens = remove_punc.split()

    return len(tokens)

def get_rationale_lengths(row):
    return get_length(row['rationale'])

def get_text_lengths(row):
    return get_length(row['text'])

def convert_df(filepath):
    df = pd.read_csv(filepath)

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

    df_true = df.loc[df['classification'] == True]
    df_false = df.loc[df['classification'] == False]

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

    return stats

if __name__ == '__main__':
    train = convert_df("../boolQ/train_data.csv")
    test = convert_df("../boolQ/test_data.csv")
    dev = convert_df("../boolQ/dev_data.csv")

    df = pd.concat([train, test, dev])

    stats = get_stats(df)
    print(stats)

    