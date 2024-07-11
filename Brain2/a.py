#!/bin/python

import torch
from torch import nn
import numpy as np

import albumentations as A
import evaluate

import matplotlib.pyplot as plt

from datasets import load_dataset
from datasets import load_metric

from transformers import DeiTImageProcessor
from transformers import DeiTForImageClassification
from transformers import ViTImageProcessor
from transformers import ViTForImageClassification
from transformers import TrainingArguments
from transformers import Trainer

# model_name_or_path = 'google/vit-base-patch16-224-in21k'
# model_name_or_path = 'google/vit-base-patch16-224-in21k'
model_name_or_path = 'facebook/deit-tiny-distilled-patch16-224'
# processor = ViTImageProcessor.from_pretrained(model_name_or_path)
processor = DeiTImageProcessor.from_pretrained(model_name_or_path)

# ds = load_dataset('imagefolder', data_dir='/home/raiku/.datasets/brain/t')
ds = load_dataset('ethz/food101')
print(ds)
# print(*list(ds['train'][0]['image'].getdata()))

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.8, 1.2), translate_percent=0.1, rotate=(-10, 10), p=0.8),
    A.ColorJitter(brightness=(0.8, 1.1), contrast=(0.8, 1.1), saturation=(0.8, 1.1), hue=(-0.1, 0.1), p=0.7),
])

def atransform(cb):
    augments = [augment(image=np.array(x))['image'] for x in cb['image']]
    inputs = processor(augments, return_tensors='pt')
    inputs['label'] = cb['label']
    return inputs

def transform(example_batch):
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')
    inputs['label'] = example_batch['label']
    return inputs

prepared_ds = ds
# prepared_ds['train'] = prepared_ds['train'].with_transform(atransform)
print(np.max(np.array(ds['train'][0]['image'].getdata())))
print(np.min(np.array(ds['train'][0]['image'].getdata())))
prepared_ds['train'] = prepared_ds['train'].with_transform(atransform)
prepared_ds['validation'] = prepared_ds['validation'].with_transform(transform)
print(torch.max(prepared_ds['train'][0]['pixel_values']))
print(torch.min(prepared_ds['train'][0]['pixel_values']))
'''
for i in range(10):
    plt.imshow(prepared_ds['train'][i]['pixel_values'].cpu().numpy().transpose([1, 2, 0]))
    plt.show()
exit()
'''

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

metric = evaluate.load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

labels = ds['train'].features['label'].names

# model = ViTForImageClassification.from_pretrained(
model = DeiTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

training_args = TrainingArguments(
  output_dir="./trains",
  per_device_train_batch_size=64,
  eval_strategy="steps",
  num_train_epochs=4,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)

device = 'cuda'
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.58768218, 3.35120643], device=device))
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=processor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_ds['validation'])
trainer.log_metrics("validation", metrics)
trainer.save_metrics("validation", metrics)
