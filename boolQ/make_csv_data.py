import pandas as pd
import json

def read_json(file):
    try:
        with open(file) as f:
            data = [json.loads(line) for line in f]
    except:
        raise Exception(f"Couldn't read file")
    
    return data

def flatten_json(data):
    for d in data:
        rationales = d.get('evidences')[0].get('text')
        query = d.get('query')
        d.update(evidences=rationales)

        new_id = d.get('docids').split('/')[-1]
        d.update(docids=new_id)

        text_file = open(('../boolQ/input/docs/' + new_id), "r")
        text_data = text_file.read()
        text_file.close()
        d['text'] = text_data

    df = pd.DataFrame(data)
    return df

def get_data(name):
    data = read_json('../boolQ/input/' + name + '.jsonl')
    df = flatten_json(data)
    return df

if __name__ == '__main__':
    # datasets = ['train_data', 'test_data', 'dev_data']

    # for dset in datasets:
    #     data = read_json('input/' + dset +'.jsonl')
    #     df = flatten_json(data)
    #     df.rename(columns={'evidences':'rationale'}, inplace=True)
    #     df.to_csv(dset + '.csv')

    #data = read_json('input/train_data.jsonl')
    df = get_data('dev_data')
    print(df['rationales'].dtype)


    
    
    
