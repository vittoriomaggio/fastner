import pandas as pd
import numpy as np
import torch
import transformers
from fastner.dataset import EntityDataset 
from transformers import MODEL_NAMES_MAPPING, BertForTokenClassification, DistilBertForTokenClassification, logging
from datasets import load_metric

from fastner.trainer import CETrainer
from ast import literal_eval

logging.set_verbosity_error()

# Metrics
metric = load_metric("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    
    if  return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }



def preprocessing(training_set, validation_set, test_set = None):
    if type(training_set) == str:
        train_df = pd.read_csv(training_set, index_col=0, converters={'tags': literal_eval, 'tokens': literal_eval}).reset_index()
    else:
        train_df = training_set

    if type(validation_set) == str:
        val_df = pd.read_csv(validation_set, index_col=0, converters={'tags': literal_eval, 'tokens': literal_eval}).reset_index()
    else:
        val_df = validation_set
        
    if  isinstance(test_set, pd.DataFrame) or  type(test_set) == str:
        if type(test_set) == str:
            test_df = pd.read_csv(test_set, index_col=0, converters={'tags': literal_eval, 'tokens': literal_eval}).reset_index()
        else:
            test_df = test_set
    else:
        test_df = None


    labels = []
    [labels.extend(tags) for tags in train_df['tags']]

    # Creat encoding for B- and I-
    global label_list 
    label_list = []
    for label in set(labels):
        if label != 'O':
            label_list.append('B-'+label[2:])
            label_list.append('I-'+label[2:])
    label_list=list(set(label_list))
    label_list.sort()
    label_list.insert(0, 'O')
    
    label2id = {k: v for v, k in enumerate(label_list)}
    train_df['tags'] = train_df["tags"].apply(lambda e: list(map(label2id.get, e)))
    val_df['tags'] =  val_df["tags"].apply(lambda e: list(map(label2id.get, e)))
    if isinstance(test_df, pd.DataFrame):  test_df['tags'] = test_df["tags"].apply(lambda e: list(map(label2id.get, e))) 

    x_train = train_df['tokens']
    y_train = train_df['tags']
    x_val = val_df['tokens']
    y_val = val_df['tags']
    x_test = test_df['tokens'] if isinstance(test_df, pd.DataFrame) else  []
    y_test = test_df['tags'] if isinstance(test_df, pd.DataFrame) else  []
    
    return x_train, y_train, x_val, y_val, x_test, y_test, label2id


def train_test(training_set, validation_set, train_args, test_set=None,
                model_name='bert-base-uncased', max_len=512, loss='CE', callbacks=[], 
                device=0 ):

    global MAX_LEN
    global MODEL_NAME
    global TOKENIZER
    MAX_LEN = max_len
    MODEL_NAME = model_name
    x_train, y_train, x_val, y_val, x_test, y_test, label2id = preprocessing(training_set, validation_set, test_set)
    torch.cuda.set_device(device)
    num_labels = len(label_list)
    
    if MODEL_NAME=='bert-base-uncased':
         TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased', max_length = max_len,  do_lower_case=True)
    elif MODEL_NAME=='distilbert-base-uncased':
         TOKENIZER = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased', max_length = max_len, do_lower_case=True)

    train_dataset = EntityDataset(
        texts=x_train, tags=y_train
    )
    val_dataset = EntityDataset(
        texts=x_val, tags=y_val
    )
    test_dataset = EntityDataset(
        texts=x_test, tags=y_test
    ) if len(x_test) != 0 else None

    def model_init(model_name):

        id2label = dict(zip(label2id.values(), label2id.keys()))

        if MODEL_NAME=='distilbert-base-uncased':
            model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels, label2id=label2id, id2label=id2label) 

        elif MODEL_NAME=='bert-base-uncased':
            model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels,  label2id=label2id, id2label=id2label) 

        else:
            model = None

        return model

    if loss == 'CE':
        trainer = CETrainer(
            model_init = model_init,
            train_dataset = train_dataset,
            eval_dataset  = val_dataset,
            args=train_args,
            compute_metrics = compute_metrics,
            callbacks=callbacks
        )

    else:
        print('Wrong Loss')
        
    global return_entity_level_metrics
    return_entity_level_metrics = True
    print()
    print('TRAINING...')
    print()
    train_output = trainer.train()
    train_results = train_output.metrics
    
    eval_results = trainer.evaluate()

    if len(x_test) != 0:
        print('TEST...')
        test_results = trainer.predict(test_dataset).metrics  
        print('END TEST...')
    else:
        test_results = None

    return train_results, eval_results, test_results, trainer    