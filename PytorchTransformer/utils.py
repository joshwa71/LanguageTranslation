import spacy
import torch
import sys

from torchtext.data.metrics import bleu_score

def translate_sequence(model, sequence, german, english, device, max_len=50):

    spacy_german = spacy.load("de_core_news_sm")

    # create tokens and set lower case
    if type(sequence) == str:
        tokens = [token.text.lower() for token in spacy_german(sequence)]
    else:
        tokens = [token.lower() for token in sequence]

    # add eos and sos tokens to sentences
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # convert tokens to indices
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # convert input to tensor
    input_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # create empty output tensor to feed to transformer
    outputs = [english.vocab.stoi["<sos>"]]

    # generate next word, feed input and next word to model
    for i in range(max_len):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(input_tensor, trg_tensor)
        
        # find highest probability word
        highest_prob = output.argmax(2)[-1, :].item()
        #add highest probability word to outputs
        outputs.append(highest_prob)


        if highest_prob == english.vocab.stoi["<eos>"]:
            break

    translated_sequence = [english.vocab.itos[idx] for idx in outputs]

    return translated_sequence[1:]

def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sequence(model, src, german, english, device)
        prediction = prediction[:-1]

        targets.append([trg])
        outputs.append(prediction)

    score = bleu_score(outputs, targets)

    return score

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])    

        
