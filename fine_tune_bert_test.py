#!pip install transformers

import torch
from tqdm.auto import tqdm
from transformers import AdamW
from transformers import BertTokenizer, BertForMaskedLM


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

data_path =" /content/the_fire_flower.txt"
with open(data_path, 'r') as f:
    data = f.read().split('\n')

data
print(len(data))

for sentence in data:
    if len(sentence) <50:
        data.remove(sentence)

print(len(data))

inputs = tokenizer(
    data,
    max_length = 512,
    truncation = True,
    padding = True,
    return_tensors = 'pt'
)

inputs.keys()

inputs['labels'] = inputs['inputs_ids'].detach().clone()

inputs

random_tensor = tocrh.rand(inputs['input_ids'].shape)
inputs['input_ids'].shape, random_tensor.shape

random_tensor

masked_tensor = (random_tensor < 0.15)* (inputs['input_ids']!=101) * (inputs['input_ids']!=102) * (inputs['input_ids']!= 0)
masked_tensor

nonzero_indices = []
for i in range(len(inputs['inputs_ids'])):   #True values to list not tensor
    nonzero_indices.append(torch.flatten(masked_tensor[i].nonzero()).tolist())


nonzero_indices

for i in range(len(inputs['inputs_ids'])):
    inputs['input_ids'][i, nonzero_indices[i]] = 103

inputs['inputs']

class BookDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        delf.encoding = encodings
    def len(self):
        return len(self.encodings)

    def __getitem__(self, index):
        input_ids = self.encodings["input_ids"]
        attention_masks = self.encoding['attention_mask']
        token_type_ids = self.encodings['token_type_ids']
        labels = self.encodings['labels']
        return {
            'input_ids':input_ids,
            'attention_mask':attention_mask,
            'token_type_ids':token_type_ids,
            'lables':labels
        }


dataset = BookDataset(inputs)

dataloader = torch.utils.data.DataLoader(
             dataset,
             batch_size = 16,
             shuffle= True
             )   

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
device

model.to(device)

epochs = 2
optimizer = Adam(model.parameters(), lr = 1e-5)

model.train()

for epoch in range(epochs):
    loop = tqdm(dataloader)
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        labels = batch['lables']
        attention_mask = batch['attention_mask']
        outputs = model(input_ids, attention_mask= attention_mask, labels = lables)
        loss = outputs.loss   #tensor
        loss.backwards()
        optimizer.step()

        loop.set_description('Epoch: {}'.format(epoch))
        loop.set_postfix(loss = loss.item()) #loss.item gives the inner value








