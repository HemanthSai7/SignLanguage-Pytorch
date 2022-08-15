import json
from sklearn.model_selection import StratifiedKFold

def data_split_StratifiedKFold(df):
    train_df=df
    with open(f'req_files/label_num_to_sign_map.json') as f:
        class_names=json.loads(f.read())
    f.close()

    train_df['label_name']=train_df['Label'].apply(lambda x: class_names[str(x)])

    sk=StratifiedKFold(n_splits=4,random_state=42,shuffle=True)
    for fold,(train,val) in enumerate(sk.split(train_df,train_df['Label'])):
        train_df.loc[val, 'fold'] = fold
    train_df.fold=train_df.fold.astype(int)
    return train_df
