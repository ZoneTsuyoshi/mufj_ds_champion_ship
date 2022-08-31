import bs4, copy, re, random
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import nlpaug.augmenter.word as naw
import torch
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from transformers import AutoTokenizer

        

class MyDataset(Dataset):
    def __init__(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, design_var:torch.Tensor, labels:torch.Tensor=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.design_var = design_var
        self.labels = labels
        
        
    def __len__(self):
        return len(self.input_ids)
    
    
    def __getitem__(self, index: int):
        encoding = {"input_ids":self.input_ids[index], "attention_mask":self.attention_mask[index], "design_var":self.design_var[index]}
        if self.labels is not None:
            encoding["labels"] = self.labels[index]
        return encoding
    
    
    
    
def get_train_data_for_mlm(config, debug=False):
    seed = config["base"]["seed"]
    kfolds = config["base"]["kfolds"]
    model_name = config["base"]["model_name"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, valid_dataset = [], []
    
    if "fold" in return_dataset:
        train_df = pd.read_csv(f"../data/{kfolds}fold-seed{seed}.csv", index_col=0)
        train_texts = train_df["description"].values
        all_indices = np.arange(len(train_df))
        for i in range(kfolds):
            train_indices = all_indices[train_df["fold"]!=i]
            valid_indices = all_indices[train_df["fold"]==i]
            train_dataset.append(DescriptionDataset(**embed_and_augment(tokenizer, train_texts[train_indices])))
            valid_dataset.append(DescriptionDataset(**embed_and_augment(tokenizer, train_texts[valid_indices])))
    if "all" in return_dataset:
        train_df = pd.read_csv("../data/train.csv", index_col=0) # id, description, jopflag
        train_texts = adjust_texts(train_df["description"].values)
        train_dataset.append(DescriptionDataset(**embed_and_augment(tokenizer, train_texts)))
        valid_dataset.append(None)
    return train_dataset, valid_dataset
    
    
def get_train_data(config, debug=False):
    model_name = config["network"]["model_name"]
    weight_on = config["train"]["weight"]
    valid_rate = config["train"]["valid_rate"]
    batch_size = config["train"]["batch_size"]
    kfolds = config["train"]["kfolds"]
    seed = config["train"]["seed"]
    design_var_list = config["train"]["design_var"]
    da_method = config["train"]["da"]
    mask_ratio = config["train"]["mask_ratio"]
    random.seed(seed)
    
    train_df = pd.read_csv("data/train.csv", index_col=0) # id, description, jopflag
    train_texts = remove_html_tags(train_df["html_content"].values)
    train_labels = train_df["state"].values
    train_df["fold"] = np.zeros(len(train_df), dtype=int)
    train_df["cleanted_text"] = train_texts
    transform_goal(train_df)
    train_design_var = get_design_var(train_df, design_var_list).astype("float32")
    design_dim = train_design_var.shape[1]
    train_df.index = range(len(train_df))
    
    skf = StratifiedKFold(n_splits=kfolds, random_state=seed, shuffle=True)
    for i, (train_indices, valid_indices) in enumerate(skf.split(train_texts, train_labels)):
        train_df.loc[valid_indices, "fold"] = i
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader, valid_loader, valid_labels, valid_indices_list, weight = [], [], [], [], []
    
    all_indices = np.arange(len(train_df))
    for i in range(kfolds):
        train_indices = all_indices[train_df["fold"]!=i]
        valid_indices = all_indices[train_df["fold"]==i]
        if debug:
            train_indices = train_indices[:32]
            valid_indices = valid_indices[:32]
        train_loader.append(DataLoader(MyDataset(**embed_and_augment(tokenizer, train_texts[train_indices], train_design_var[train_indices], train_labels[train_indices], da_method, mask_ratio)), batch_size=batch_size, shuffle=True))
        valid_loader.append(DataLoader(MyDataset(**embed_and_augment(tokenizer, train_texts[valid_indices], train_design_var[valid_indices], train_labels[valid_indices])), batch_size=batch_size, shuffle=False))
        valid_labels.append(train_labels[valid_indices])
        valid_indices_list.append(valid_indices)
        if weight_on: 
            weight.append(torch.tensor(compute_class_weight("balanced", classes=np.arange(2), y=train_labels[train_indices]), dtype=torch.float32))
        else:
            weight.append(None)

    if debug:
        train_texts = train_texts[:32]
        train_labels = train_labels[:32]
        train_design_var = train_design_var[:32]
    train_loader.append(DataLoader(MyDataset(**embed_and_augment(tokenizer, train_texts, train_design_var, train_labels, da_method, mask_ratio)), batch_size=batch_size, shuffle=True))
    valid_loader.append(None)
    valid_indices_list.append(None)
    valid_labels.append(None)
    if weight_on: 
        weight.append(torch.tensor(compute_class_weight("balanced", classes=np.arange(2), y=train_labels), dtype=torch.float32))
    else:
        weight.append(None)
    return train_loader, valid_loader, valid_labels, valid_indices_list, weight, design_dim, train_df
    
    
    
def get_test_data(config, debug=False):
    model_name = config["network"]["model_name"]
    batch_size = config["train"]["batch_size"]
    design_var_list = config["train"]["design_var"]
    
    test_df = pd.read_csv("data/test.csv", index_col=0)# id, description
    if debug:
        test_df = test_df.iloc[:32,:]
    test_texts = remove_html_tags(test_df["html_content"].values)
    transform_goal(test_df)
    test_design_var = get_design_var(test_df, design_var_list).astype("float32")
    test_index = test_df.index
    # design_dim = test_design_var.shape[1]
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataset = MyDataset(**embed_and_augment(tokenizer, test_texts, test_design_var))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader, test_index
    
    
    
def embed_and_augment(tokenizer, texts, design_var=None, labels=None, da_method=None, mask_ratio=0.1):
    """
    texts, labels: ndarray
    
    da_method
        m: mask words
    """
    # tokenizer
    tokenizer_setting = {"add_special_tokens":True, "max_length":512, "return_token_type_ids":False,
                         "padding":"max_length", "truncation":True, "return_attention_mask":True, "return_tensors":'pt'}
    
    encoding = tokenizer(texts.tolist(), **tokenizer_setting)
    if labels is None:
        encoding["design_var"] = torch.from_numpy(design_var)
    else:
        encoding["labels"] = torch.tensor(labels.tolist())
        if da_method is None:
            encoding["design_var"] = torch.from_numpy(design_var)
        else:
            input_ids = [i for i in encoding["input_ids"]]
            attention_mask = [i for i in encoding["attention_mask"]]
            new_design_var = [i for i in design_var]
            new_labels = labels.tolist()
            unique_labels, counts = np.unique(labels, return_counts=True)
            max_count = counts.max()

            for u, c in zip(unique_labels, counts):
                now_texts = texts[labels==u]
                now_design_var = design_var[labels==u]
                counter = c
                inner_counter = 0
                while counter < max_count:
                    da_text = now_texts[inner_counter]
                    encoding = tokenizer(da_text, **tokenizer_setting)
                    current_input_ids = encoding["input_ids"][0]
                    if "m" in da_method:
                        current_length = (current_input_ids > 0).sum().item() - 2
                        perms = np.random.choice(current_length, int(mask_ratio*current_length), replace=False)
                        for p in perms:
                            current_input_ids[p+1] = tokenizer.mask_token_id

                    input_ids.append(current_input_ids)
                    attention_mask.append(encoding["attention_mask"][0])
                    new_labels.append(u)
                    new_design_var.append(now_design_var[inner_counter])
                    inner_counter += 1
                    if inner_counter == c:
                        inner_counter = 0
                    counter += 1
            encoding = {"input_ids":torch.stack(input_ids), "attention_mask":torch.stack(attention_mask), "labels":torch.tensor(new_labels), "design_var":torch.tensor(new_design_var)}
    return encoding


def transform_goal(df):
    for i,g in enumerate(df["goal"].values):
        if "+" in g:
            df.at[df.index[i], "goal"] = 101000
        else:
            df.at[df.index[i], "goal"] = int(g.split("-")[1])

            
def get_design_var(df, design_var_list="all"):
    if design_var_list=="all": design_var_list = ["goal", "duration", "category1", "category2"]
    
    design_var = []
    if "goal" in design_var_list:
        design_var.append(np.log10(df["goal"].values.astype(float)[:,None]/1000)/2)
    if "duration" in design_var_list:
        design_var.append(df["duration"].values.astype(float)[:,None]/90)
    if "category1" in design_var_list:
        category_list = ['art', 'comics', 'crafts', 'dance', 'design', 'fashion', 'film & video', 'food', 'games', 'journalism', 
                         'music', 'photography', "publishing", "technology", 'theater']
        design_var.append(OneHotEncoder(categories=[category_list], sparse=False).fit_transform(df["category1"].values.reshape(-1,1)))
    if "category2" in design_var_list:
        category_list = ['mixed media', 'restaurants', 'performance art', 'webseries',
                       'plays', 'classical music', 'public art', 'metal',
                       'country & folk', 'diy electronics', 'footwear', 'art books',
                       'accessories', 'calendars', 'digital art', 'web', 'drinks',
                       'fiction', 'world music', 'mobile games', 'food trucks', 'musical',
                       'illustration', 'narrative film', 'shorts', 'architecture',
                       'movie theaters', 'fine art', "children's books", 'hardware',
                       'animals', 'playing cards', 'graphic novels', 'radio & podcasts',
                       'animation', 'festivals', 'hip-hop', 'print', 'webcomics',
                       'gadgets', 'people', 'diy', 'electronic music', 'live games',
                       'installations', 'thrillers', 'farms', 'spaces', 'events', 'rock',
                       'jewelry', 'software', 'woodworking', 'science fiction', 'poetry',
                       'places', 'bacon', 'community gardens', 'young adult', 'family',
                       'fantasy', 'painting', 'comedy', 'gaming hardware', 'jazz',
                       'nonfiction', 'performances', '3d printing', 'graphic design',
                       'small batch', 'vegan', 'photobooks', "farmer's markets", 'horror',
                       'couture', 'tabletop games', 'civic design', 'sculpture',
                       'makerspaces', 'sound', 'interactive design', 'action',
                       'indie rock', 'quilts', 'audio', 'romance', 'wearables',
                       'residencies', 'kids', 'video games', 'documentary', 'cookbooks',
                       'apparel', 'robots', 'conceptual art', 'childrenswear',
                       'camera equipment', 'product design', 'punk', 'apps', 'television',
                       'video art', 'zines', 'glass', 'drama', 'latin', 'academic',
                       'video', 'crochet', 'music videos', 'ready-to-wear',
                       'literary journals', 'pop', 'embroidery', 'flight', 'periodicals',
                       'faith', 'comic books', 'knitting', 'r&b', 'textiles',
                       'typography', 'ceramics', 'weaving', 'pottery', 'immersive',
                       'photo', 'chiptune', 'anthologies', 'experimental',
                       'fabrication tools', 'candles', 'workshops', 'puzzles',
                       'space exploration', 'pet fashion', 'blues', 'printing', 'nature',
                       'translations', 'stationery', 'literary spaces', 'letterpress',
                       'social practice', 'toys']
        design_var.append(OneHotEncoder(categories=[category_list], sparse=False).fit_transform(df["category2"].values.reshape(-1,1)))
    return np.concatenate(design_var, axis=1)
            
                

def adjust_text(text, parser="lxml"):
    new_text = bs4.BeautifulSoup(text, parser).get_text()
    delete_list = ["\n", "\xa0", "//", 
                   "このコンテンツを表示するにはHTML5対応のブラウザが必要です。", "動画を再生", "音ありでリプレイ", "音声ありで  再生", "00:00"]
    sub_list = [["/+", "/"], ["\.+", "."], ["-+", "-"], [" +", " "]]
    for t in delete_list:
        new_text = new_text.replace(t, " ")
    for t in sub_list:
        new_text = re.sub(t[0], t[1], new_text)
    if "http" in new_text or "www" in new_text:
        words = new_text.split(" ")
        for j,w in enumerate(words):
            if "http" in w or "www" in w:
                words[j] = "URL"
        new_text = " ".join(words)
    return new_text
    
    
def remove_html_tags(arr, parser="lxml"):
    results = copy.deepcopy(arr)
    for i in range(len(results)):
        results[i] = adjust_text(results[i], parser)
    return results