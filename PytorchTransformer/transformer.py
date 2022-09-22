from cgi import test
from json import load
import torch
import torch.optim as optim
import torch.nn as nn
import spacy

from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from utils import load_checkpoint, save_checkpoint, translate_sequence, bleu

spacy_german = spacy.load("de_core_news_sm")
spacy_english= spacy.load("en_core_web_sm")

def tokenize_german(text):
    return [token.text for token in spacy_german.tokenizer(text)]

def tokenize_english(text):
    return [token.text for token in spacy_english.tokenizer(text)]

german = Field(tokenize=tokenize_german, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(tokenize=tokenize_english, lower=True, init_token="<sos>", eos_token="<eos>")


train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)

german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_index,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len, 
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
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
        self.src_pad_index = src_pad_index
    
    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_index
        return src_mask

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device)
        )
        trg_positions = (
            torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_paddind_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask = src_paddind_mask,
            tgt_mask=trg_mask
        )

        out = self.fc_out(out)

        return out  

if __name__ == "__main__":

    print(f"GPU = {torch.cuda.get_device_name(0)}")
    print(f"GPU available = {torch.cuda.is_available()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_model = True
    load_model = False

    #Training hyperparams
    num_epochs = 50
    learning_rate = 3e-4
    batch_size=32

    #Model Hyperparams
    src_vocab_size = len(german.vocab)
    trg_vocab_size = len(english.vocab)
    embedding_size = 512
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dropout = 0.10
    max_len = 100
    forward_expansion = 4
    src_pad_idx = english.vocab.stoi["<pad>"]

    #Tensorboard
    writer = SummaryWriter("runs/loss_plot")
    step = 0

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        sort_within_batch = True,
        sort_key = lambda x: len(x.src),
        device=device,
    )

    #Model setup
    model = Transformer(
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = english.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    if load_model:
        load_checkpoint(torch.load("checkpoint.pth.tar"), model, optimizer)

    sentence = "ein pferd geht unter einer brücke neben einem boot."

    for epoch in range(num_epochs):
        print(f"[Epoch = {epoch}]")

        if save_model:
            checkpoint_dict = {
                "state_dict" : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
            }

            save_checkpoint(checkpoint_dict)

        model.eval()
        translated_sentence = translate_sequence(model, sentence, german, english, device, max_len=100)

        print("Translated example sentence:")
        print(translated_sentence)

        model.train()

        for batch_idx, batch in enumerate(train_iterator):
            #print(f"Batch index = {batch_idx}")

            input_data = batch.src.to(device)
            target = batch.trg.to(device)

            output = model(input_data, target[:-1])

            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)
            optimizer.zero_grad()

            loss = criterion(output, target)
            loss.backward()

            

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)

            optimizer.step()

            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1


    final_model_dict = {
        "state_dict" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        }

    save_checkpoint(final_model_dict, filename="final_model.pth.tar")
    print("Saved final model")


    score = bleu(test_data, model, german, english, device)
    print(f"Blue score = {score*100:.2f}")

