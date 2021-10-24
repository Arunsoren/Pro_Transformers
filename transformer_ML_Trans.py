import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30K
from torchtext.data import Field, BucketIterator

from utils import translate_sentence, bleu, save_checkpoints, load_checkpoints

"""
install spacy languages :
python -m spacy doenload en
python -m spacy download de

"""

spacy_ger = spacy.load("de")
spacy_en = spacy.load("en")

def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenize(text)]

german.build_vocab(train_data, max_size=10000, min_freq= 2)
english.build_vocab(train_data, max_size= 10000, min_freq= 2)

class Transformer(nn.Module):
    def __init__(self,
                embedding_size,
                src_vocal_size,
                trg_vocal_size,
                src_pad_idx,
                num_heads,
                num_encoder_layers,
                num_decoder_layers,
                forward_expansion,
                dropout,
                max_len,
                device,):
                super(Transformer, self).__init__()
                self.src_word_enbedding = nn.Embedding(src_vocal_size, embedding_size)
                self.src_position_embedding = nn.Embedding(max_len, embedding_size)
                self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
                self.trg_position_embedding = nn.Embedding(max_len, embedding_size)
                self.device = device
                self.transformer = nn.Transformer(
                    embedding_size,
                    num_heads,
                    num_encoder_layers,
                    num_decoder_layers,
                    forward_expansion,
                    dropout,
                )
                self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
                self.dropout = nn.Dropout(dropout)
                self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        #src shape:(src_lee, N)  
        src_mask = src.transpose(0,1) == self.src_pad_idx 
        #(N, src_len) to change transpose
        return src_mask

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape


        src_positions = (
            torch.arrange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arrange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src)+ self.src_position_embedding(src_positions))
        )

        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))

        )

        src_paddings_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask = src_padding_mask,
            trg_mask = trg_mask
        )

        out= self.fc_out(out)
        return out

#training phase
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

#training hyperparmeters
num_epochs = 5
learning_rate = 3e-4
batch_size = 32

#model hyperparameters
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layer = 3
dropout = 0.10
max_len = 100
forward_expansion = 4  #No of nodes we are ampping to only 4 nodes
src_pad_idx = english.vocab.stoi["<pad>"]


#Tensorboard for plots
Writer = SummaryWriter("runs/loss_plot")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = batch_size,
    sort_within_batch = True,
    sort_key = lambda x: len(x.src),
    device = device
)

model = Transformer(embedding_size,
                    src_vocal_size,
                    trg_vocal_size,
                    src_pad_idx,
                    num_heads,
                    num_encoder_layers,
                    num_decoder_layers,
                    forward_expansion,
                    dropout,
                    max_len,
                    device).to(device)

optimizer = optim.Adam(model.parameters(), lr= learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index= pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = "ein Ausritt unter einer Brücke neben einem Boot"
#english :"a horse walk under a bridge next to a boat"


for epoch in range(num_epochs):
    print(f"[Epoch {epoch}/{num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict" :model.state_dict(),
            "optimizer" :optimizer.state_dict(),
        }

        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_len = 100
                                        )

    print(f"Translated example sentence\n{translated_sentence}")
    model.train()

    for batcg_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        #forward prop
        output = model(inp_data, target[:-1])

        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        optimizer.zero_grad()

        loss = criterion(output, target)
        loss.backward()
        #for exploding grad prob
        torch.nn.utils.clip_grad_norm_(model.parameter(), max_norm=1)

        optimizer.step() 
        writer.add_scalar("Training loss", loss, global_step =step)
        step += 1

#take time 
score = blue(test_data, model ,german, english, device)
print(f"Blue score{score*100:.2f}")






