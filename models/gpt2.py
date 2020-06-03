import torch
from transformers import GPT2Tokenizer, GPT2Model
import torch.nn as nn
import torch.nn.functional as F


class GPT2(nn.Module):
    def __init__(self):
        super(GPT2, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2Model.from_pretrained('gpt2')
        self.length = 1024*20

    
    def forward(self, x):
        with torch.no_grad():
            y = []
            for q in x:
                token = torch.tensor(self.tokenizer.encode(q, add_special_tokens=True)).unsqueeze(0).to('cuda')
                outputs = self.model(token)
                out = outputs[0]
                out = out.reshape(-1)
                tmp = torch.zeros(self.length).to('cuda')
                if out.shape[0] < self.length:
                    tmp[0:out.shape[0]] = out
                else:
                    tmp = out[0:self.length]
                y.append(tmp)
            y = torch.stack(y, dim=0)
        return y
        

