from transformers import Trainer
import torch.nn as nn

class CETrainer(Trainer):

    def __init__(self, ignore_index=-100, device=0, *args, **kwargs):
      super(CETrainer, self).__init__(*args, **kwargs)
      self.ignore_index = ignore_index
      self.device = device


    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        targets = labels
        valid_idx = targets != self.ignore_index
        logits = logits[valid_idx]
        targets = targets[valid_idx]

        ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='mean')
        loss = ce_loss(logits, targets)

        return (loss, outputs) if return_outputs else loss

