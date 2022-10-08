# fastner
fastner is a Python package to finetune transformer-based models for the Named Entity Recognition task in a simple and fast way.  
It is based on the torch and the transformer🤗 libraries.
## Main features
The last version of fastner is 0.0.1 and it provides:
### Models
The transformer-based models that you can use for the finetuning are:
 - Bert base uncased (bert-base-uncased)
 - DistilBert base uncased (distilbert-base-uncased)
###  Tagging scheme
 The labels of the dataset given as input must comply with the tagging scheme:
 - IOB (Inside, Outside, Beginning), also known as BIO
 ### Dataset scheme
The datasets given as input (train, validation, test) **must have two columns** named:
- **tokens**:  contains the tokens of the several examples
- **tags**: contains the labels of the respective tokens

Example:
 
| **tokens** |  **tags**|
|--|--|
|['Apple', 'CEO', 'Tim', 'Cook', 'introduces', 'the', 'new', 'iPhone']|['B-ORG', 'O', ''B-PER', 'I-PER', 'O', 'O','O', 'O']|



## Installation
### With pip
fastner can be installed using [pip](https://pypi.org/project/fastner/) as follows:

    pip install fastner

## How to use it
Use fastner is very easy! All you need is a dataset that respects the format previously given.
The core function is the ***train_test()*** function:

**Parameters:**
 - training_set (*string* or pandas *DataFrame*) - path of the *.csv* training set or the *pandas.DataFrame* object of the training set
 - validation_set (*string* or pandas *DataFrame*) - path of the *.csv* validation set or the *pandas.DataFrame* object of the validation set
 - test_set: default (*optional*,  *string* or pandas *DataFrame*) - path of the *.csv* test set or the *pandas.DataFrame* object of the test set 
 - model_name (*string*, default: *'bert-base-uncased'*) - name of the model to finetune (available: *'bert-base-uncased'* or *'distilbert-base-uncased'*)
 - train_args (*transformers.TrainingArguments*) - arguments for the training (see [hugginface documenation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments))
 - max_len (*integer*, default: *128*) - maximum input sequence length
 - loss (*string*, default=*'CE'*) - loss function, the only one available at the moment is the 'CE' Cross Entropy 
 - callbacks (*optional*, *list* of *transformers callbacks*) -  list of transformers callbacks (see [hugginface documentation](https://huggingface.co/docs/transformers/main_classes/callback))
 - device (*integer*, default: *0*) - id of the device on which to perform the training
 
**Outputs:**
- train_results (*dict*) - dict with training info (runtime, samples per second, steps per seconds, loss, epochs)
- eval_results (*dict*) - dict with evaluation metrics on the validation set (precision, recall, f1 both overall and for the single entities, loss)
- test_results (*dict*) -  dict with evaluation metrics on the test set (precision, recall, f1 both overall and for the single entities, loss)
- trainer (*transofrmers.Trainer*) - *transformers.Trainer* object used

## Example
An example of fastner in action:

    from transformers import TrainingArguments, EarlyStoppingCallback
    from fastner import train_test
    
    args = TrainingArguments(
                num_train_epochs = 5,
                per_device_train_batch_size = 32,
                per_device_eval_batch_size = 8,
                output_dir= "./models",
                evaluation_strategy="epoch",
                logging_strategy = "epoch",
                save_strategy = "epoch",
                load_best_model_at_end= True,
                metric_for_best_model = 'eval_loss')
							
	train_results, eval_results, test_results, trainer = train_test(
							training_set = conll2003_train,
							validation_set = conll2003_val,
							test_set=conll2003_test,
							train_args = args,
							model_name='distilbert-base-uncased',
							max_len=128, 
							loss='CE',
							callbacks= [EarlyStoppingCallback(early_stopping_patience=3)],
							device=0)
							

    
## Work in Progress
A few spoilers about future releases:
- New models
- New tagging formats 
- New function that takes as input the dataset without any tagging scheme and returns it with the chosen tagging scheme
