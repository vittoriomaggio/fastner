import torch
from  fastner import training
import pandas as pd

class EntityDataset:
    def __init__(self, texts, tags=None):
        self.texts = texts
        self.tags = tags

        

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        if isinstance(self.tags, pd.Series):
            tags = self.tags[item]
        else:
            tags = [-100]*len(text)
        ids = []
        
        target_tag = []
        for i,s in enumerate(text):
            inputs = training.TOKENIZER.encode(
                s,
                add_special_tokens = False
            )

            input_len = len(inputs) # unknown: unk ##no ##wn
            if input_len > 0:
              ids.extend(inputs) 
              target_tag.extend([tags[i]])     # Tag the first token of the split
              target_tag.extend([-100] * (input_len-1)) # 0 tag to all the token of the splitted word

        ids = ids[:training.MAX_LEN - 2] # -2 because you need to add [CLS] and [SEP]
        target_tag = target_tag[:training.MAX_LEN -2]

        ids = [101] + ids + [102] # [101] = CLS tag; [102] = [SEP] tag
        target_tag = [0] + target_tag + [0]

        # Padding TAG 
        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)
        padding_len = training.MAX_LEN - len(ids)
        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        
        target_tag = target_tag + ([-100] * padding_len)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels" : torch.tensor(target_tag, dtype=torch.long)
        }