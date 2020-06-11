import torch
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
import torch.nn.functional as F


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        self.x = 20
        self.y = 768

    
    def forward(self, x):
        with torch.no_grad():
            y = []
            for q in x:
                token = torch.tensor([self.tokenizer.encode(q, add_special_tokens=True)]).to('cuda')
                out = self.model(token)[0][0]
                tmp = torch.zeros((self.x, self.y)).to('cuda')
                if out.shape[0] < self.x:
                    tmp[0:out.shape[0], :] = out
                else:
                    tmp = out[0:self.x, :] = out
                y.append(tmp)
        y = torch.stack(y, dim=0)
        return y
        

