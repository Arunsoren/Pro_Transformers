#!pip install transformers
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sample_text = "my name is Arun!!"

tokens = tokenizer.tokenize(sample_text)
tokens

token_ids = tokenizer.convert_tokens_to_ids(tokens)
token_ids

token1 = tokenizer.encode(sample_text)
token1    #direct tokens to ids

token2 = tokenizer.encode_plus(sample_text, max_length= 10, padding = 'max_length', return_attention_mask= True, return_tensors = "pt")
token2    #with input_id , token_type_id and attention_mask

sample_text_1 = ['i do not knwo my name', 'i love drawing', 'i love creating Art']

token2 = tokenizer.batch_encode_plus(sample_text_1)
token2    #for batches of data

token3 = tokenizer(sample_text)
token3














































