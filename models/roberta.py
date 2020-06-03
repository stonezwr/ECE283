import torch
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
import torch.nn.functional as F


class Roberta(nn.Module):
    def __init__(self):
        super(Roberta, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.length = 1024*20

    
    def forward(self, x):
        with torch.no_grad():
            y = []
            for q in x:
                token = torch.tensor([self.tokenizer.encode(q, add_special_tokens=True)]).to('cuda')
                out = self.model(token)[0]
                out = out.reshape(-1)
                tmp = torch.zeros(self.length).to('cuda')
                if out.shape[0] < self.length:
                    tmp[0:out.shape[0]] = out
                else:
                    tmp = out[0:self.length]
                y.append(tmp)
        y = torch.stack(y, dim=0)
        return y
        

