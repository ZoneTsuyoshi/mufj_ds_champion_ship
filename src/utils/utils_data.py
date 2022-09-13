import bs4, copy, re, random, json
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
    def __init__(self, input_ids:torch.Tensor, attention_mask:torch.Tensor, design_var:torch.Tensor=None, labels:torch.Tensor=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.design_var = design_var
        self.labels = labels
        
        
    def __len__(self):
        return len(self.input_ids)
    
    
    def __getitem__(self, index: int):
        encoding = {"input_ids":self.input_ids[index], "attention_mask":self.attention_mask[index]}
        if self.design_var is not None:
            encoding["design_var"] = self.design_var[index]
        if self.labels is not None:
            encoding["labels"] = self.labels[index]
        return encoding
    
    
    
def get_train_data(config, debug=False, pseudo_labeling_vars=None):
    model_name = config["network"]["model_name"]
    weight_on = config["train"]["weight"]
    valid_rate = config["train"]["valid_rate"]
    batch_size = config["train"]["batch_size"]
    kfolds = config["train"]["kfolds"]
    seed = config["train"]["seed"]
    st_cat1_on = config["train"]["st_cat1_on"]
    concat_var_list = config["train"]["concat_var"]
    design_var_list = config["train"]["design_var"]
    remove_non_english = config["train"]["remove_non_english"]
    da_method = config["train"]["da"]
    mask_ratio = config["train"]["mask_ratio"]
    random.seed(seed)
    pseudo_labeling = pseudo_labeling_vars is not None
    if pseudo_labeling:
        exp_id = pseudo_labeling_vars[0]
        confidence = float(pseudo_labeling_vars[1])
        with open(f"exp/{exp_id}/results/search_results.json", "r") as f:
            search_results = json.load(f)
        threshold = search_results["threshold"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if pseudo_labeling:
        train_df = pd.read_csv(f"exp/{exp_id}/results/valid_data.csv", index_col=0)
        train_texts = train_df["cleaned_text"].values
        train_labels = train_df["state"].values
        train_design_var, design_dim = get_design_var(train_df, design_var_list)
        train_df.index = range(len(train_df))
        
        test_df = pd.read_csv("data/test.csv", index_col=0)
        test_probs = np.load(f"exp/{exp_id}/results/test_probs.npy") #(kfolds+1,ns)
        if debug:
            test_df = test_df.iloc[:32,:]
            test_probs = test_probs[:,:32]
        transform_goal(test_df)
        test_texts = concat_text_with_other_infos(remove_html_tags(test_df["html_content"].values, remove_non_english), test_df, concat_var_list, tokenizer.sep_token)
        test_design_var, _ = get_design_var(test_df, design_var_list)
    else:
        train_df = pd.read_csv("data/train.csv", index_col=0) # id, description, jopflag
        transform_goal(train_df)
        train_texts = concat_text_with_other_infos(remove_html_tags(train_df["html_content"].values, remove_non_english), train_df, concat_var_list, tokenizer.sep_token)
        train_labels = train_df["state"].values
        train_df["fold"] = np.zeros(len(train_df), dtype=int)
        train_df["cleaned_text"] = train_texts
        train_design_var, design_dim = get_design_var(train_df, design_var_list)
        train_df.index = range(len(train_df))
    
        skf = StratifiedKFold(n_splits=kfolds, random_state=seed, shuffle=True)
        st_var = train_df["category1"].values + train_labels.astype(str) if st_cat1_on else train_labels
        for i, (train_indices, valid_indices) in enumerate(skf.split(train_texts, st_var)):
            train_df.loc[valid_indices, "fold"] = i
    
    train_loader, valid_loader, valid_labels, valid_indices_list, weight = [], [], [], [], []
    all_indices = np.arange(len(train_df))
    for i in range(kfolds):
        train_indices = all_indices[train_df["fold"]!=i]
        valid_indices = all_indices[train_df["fold"]==i]
        if debug:
            train_indices = train_indices[:32]
            valid_indices = valid_indices[:32]
        if pseudo_labeling:
            test_indices = np.any([test_probs[i]>=confidence, test_probs[i]<=1-confidence], axis=0)
            train_loader.append(DataLoader(MyDataset(**embed_and_augment(tokenizer, np.concatenate([train_texts[train_indices], test_texts[test_indices]]), np.concatenate([train_design_var[train_indices], test_design_var[test_indices]]) if design_dim>0 else None, np.concatenate([train_labels[train_indices], (test_probs[i]>=threshold).astype(int)[test_indices]]), da_method, mask_ratio)), batch_size=batch_size, shuffle=True))
        else:
            train_loader.append(DataLoader(MyDataset(**embed_and_augment(tokenizer, train_texts[train_indices], train_design_var[train_indices] if design_dim>0 else None, train_labels[train_indices], da_method, mask_ratio)), batch_size=batch_size, shuffle=True))
        valid_loader.append(DataLoader(MyDataset(**embed_and_augment(tokenizer, train_texts[valid_indices], train_design_var[valid_indices] if design_dim>0 else None, train_labels[valid_indices])), batch_size=batch_size, shuffle=False))
        valid_labels.append(train_labels[valid_indices])
        valid_indices_list.append(valid_indices)
        if weight_on: 
            weight.append(torch.tensor(compute_class_weight("balanced", classes=np.arange(2), y=train_labels[train_indices]), dtype=torch.float32))
        else:
            weight.append(None)

    if debug:
        train_texts = train_texts[:32]
        train_labels = train_labels[:32]
        if design_dim>0:
            train_design_var = train_design_var[:32]
    if pseudo_labeling:
        test_indices = np.any([test_probs[-1]>=confidence, test_probs[-1]<=1-confidence], axis=0)
        train_loader.append(DataLoader(MyDataset(**embed_and_augment(tokenizer, np.concatenate([train_texts, test_texts[test_indices]]), np.concatenate([train_design_var, test_design_var[test_indices]]) if design_dim>0 else None, np.concatenate([train_labels, (test_probs[-1]>=threshold).astype(int)[test_indices]]), da_method, mask_ratio)), batch_size=batch_size, shuffle=True))
    else:
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
    remove_non_english = config["train"]["remove_non_english"]
    concat_var_list = config["train"]["concat_var"]
    design_var_list = config["train"]["design_var"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_df = pd.read_csv("data/test.csv", index_col=0)
    if debug: test_df = test_df.iloc[:32,:]
    transform_goal(test_df)
    test_texts = concat_text_with_other_infos(remove_html_tags(test_df["html_content"].values, remove_non_english), test_df, concat_var_list, tokenizer.sep_token)
    test_design_var, design_dim = get_design_var(test_df, design_var_list)
    test_index = test_df.index
    
    # tokenizer
    test_dataset = MyDataset(**embed_and_augment(tokenizer, test_texts, test_design_var))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader, test_index


def concat_text_with_other_infos(texts, df, concat_var_list="gdc12", sep=";"):
    """
    texts: ndarray
    df: dataframe
    """
    extract_list = []
    if "g" in concat_var_list: extract_list.append("goal")
    if "d" in concat_var_list: extract_list.append("duration")
    if "c" in concat_var_list: extract_list.append("country")
    if "1" in concat_var_list: extract_list.append("category1")
    if "2" in concat_var_list: extract_list.append("category2")
    
    if len(extract_list)>0:
        extract_var = df[extract_list].values.astype(str)
        if "g" in concat_var_list:
            extract_var[:,0] = (df["goal"].values.astype(int)/1000).astype(int).astype(str)
        new_texts = []
        for i,t in enumerate(texts):
            new_texts.append(sep.join(extract_var[i]) + sep + t)
        return np.array(new_texts)
    else:
        return texts
            
    
    
    
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
        if design_var is not None:
            encoding["design_var"] = torch.from_numpy(design_var)
    else:
        encoding["labels"] = torch.tensor(labels.tolist())
        if da_method is None:
            if design_var is not None:
                encoding["design_var"] = torch.from_numpy(design_var)
        else:
            input_ids = [i for i in encoding["input_ids"]]
            attention_mask = [i for i in encoding["attention_mask"]]
            if design_var is not None:
                new_design_var = [i for i in design_var]
            new_labels = labels.tolist()
            unique_labels, counts = np.unique(labels, return_counts=True)
            max_count = counts.max()

            for u, c in zip(unique_labels, counts):
                now_texts = texts[labels==u]
                if design_var is not None:
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
                    if design_var is not None:
                        new_design_var.append(now_design_var[inner_counter])
                    inner_counter += 1
                    if inner_counter == c:
                        inner_counter = 0
                    counter += 1
            encoding = {"input_ids":torch.stack(input_ids), "attention_mask":torch.stack(attention_mask), "labels":torch.tensor(new_labels), "design_var":torch.tensor(new_design_var) if design_var is not None else None}
    return encoding


def transform_goal(df):
    for i,g in enumerate(df["goal"].values):
        if "+" in g:
            df.at[df.index[i], "goal"] = 101000
        else:
            df.at[df.index[i], "goal"] = int(g.split("-")[1])

            
def get_design_var(df, design_var_list="all"):
    if design_var_list in ["a", "all"]: design_var_list = "gd12"
    
    design_var = []
    if "g" in design_var_list:
        design_var.append(np.log10(df["goal"].values.astype(float)[:,None]/1000)/2)
    if "d" in design_var_list:
        design_var.append(df["duration"].values.astype(float)[:,None]/90)
    if "c" in design_var_list:
        category_list = ['US', 'CA', 'FR', 'GB', 'IT', 'AU', 'DE', 'SE', 'NO', 'DK', 'SG',
                        'BE', 'ES', 'MX', 'AT', 'NL', 'NZ', 'HK', 'CH', 'IE', 'JP', 'LU']
        design_var.append(OneHotEncoder(categories=[category_list], sparse=False).fit_transform(df["country"].values.reshape(-1,1)))
    if "1" in design_var_list:
        category_list = ['art', 'comics', 'crafts', 'dance', 'design', 'fashion', 'film & video', 'food', 'games', 'journalism', 
                         'music', 'photography', "publishing", "technology", 'theater']
        design_var.append(OneHotEncoder(categories=[category_list], sparse=False).fit_transform(df["category1"].values.reshape(-1,1)))
    if "2" in design_var_list:
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
        design_var.append(OneHotEncoder(categories=[category_list], sparse=False).fit_transform(df["category2"].values.reshape(-1,1))[:,:-1])
    if len(design_var)>0:
        design_var = np.concatenate(design_var, axis=1).astype("float32")
        return design_var, design_var.shape[1]
    else:
        return None, 0
            
                

def adjust_text(text, remove_non_english=True, parser="lxml"):
    new_text = bs4.BeautifulSoup(text, parser).get_text().lower()
    delete_list = ["\n", "\t", "\xa0", "//", "_", "*",
                   "このコンテンツを表示するにはHTML5対応のブラウザが必要です。", "このコンテンツを表示するにはhtml5対応のブラウザが必要です。",
                   "動画を再生", "音ありでリプレイ", "音声ありで  再生", "00:00",
                   "この動画はエンコード中です", "数分後にもう一度確認して見てください！"]
    domain_list = ["com", "edu", "go", "mil", "net", "info"]
    ret_list = [["http", "homepage"], ["www", "homepage"]]
    sub_list = [["/+", "/"], ["=+", "="], ["\.+", ". "], ["-+", "-"], [":+", ":"], ["ー+", "ー"], [" +", " "]]
    rep_list = [[" .", "."]]
    
    for t in delete_list:
        new_text = new_text.replace(t, " ")
        
    for r in ret_list:
        if r[0] in new_text:
            words = new_text.split(" ")
            for j,w in enumerate(words):
                if r[0] in w:
                    words[j] = r[1]
            new_text = " ".join(words)
            
    for d in domain_list:
        if "."+d in new_text:
            words = new_text.split(" ")
            for j,w in enumerate(words):
                if "."+d in w:
                    words[j] = w.split(".")[0]
            new_text = " ".join(words)
            
    if remove_non_english:
        words = []
        for w in re.split("[.!?。！？]", new_text):
            if w.isascii():
                words.append(w)
        new_text = ". ".join(words)
            
    for t in sub_list:
        new_text = re.sub(t[0], t[1], new_text)
    if len(new_text)>0:
        if new_text[0]==" ":
            new_text = new_text[1:]
    if len(new_text)>0:
        if new_text[-1]==" ":
            new_text = new_text[:-1]
            
    for r in rep_list:
        new_text = new_text.replace(r[0], r[1])
            
    return new_text
    
    
def remove_html_tags(arr, remove_non_english=True, parser="lxml"):
    results = copy.deepcopy(arr)
    for i in range(len(results)):
        results[i] = adjust_text(results[i], remove_non_english, parser)
    return results