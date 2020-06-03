import torch
from transformers import TransfoXLTokenizer, TransfoXLModel
import torch.nn as nn
import torch.nn.functional as F


class TransXL(nn.Module):
    def __init__(self):
        super(TransXL, self).__init__()
        self.tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        self.model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
        self.length = 1024*20

    
    def forward(self, x):
        with torch.no_grad():
            y = []
            for q in x:
                token = torch.tensor(self.tokenizer.encode(q, add_special_tokens=True)).unsqueeze(0).to('cuda')
                outputs = self.model(token)
                out, mems = outputs[:2]
                out = out.reshape(-1)
                tmp = torch.zeros(self.length).to('cuda')
                if out.shape[0] < self.length:
                    tmp[0:out.shape[0]] = out
                else:
                    tmp = out[0:self.length]
                y.append(tmp)
            y = torch.stack(y, dim=0)
        return y
        

