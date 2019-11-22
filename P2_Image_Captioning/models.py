
import torch 
import torch.nn as nn
import torchvision.models as models 


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.embed_size = embed_size
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        resnet_modules = list(resnet.children())[:-1] 
        self.backbone = nn.Sequential(*resnet_modules)
        self.tail = nn.Linear(in_features=resnet.fc.in_features, 
                                    out_features=embed_size)
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.tail(x)
        return x


class DecoderRNN(nn.Module):
    def __init__(self,  embed_size, hidden_size, vocab_size, dropout_prob=.2, 
                 num_layers=2, 
                 ):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.caption_embeddings = nn.Embedding(vocab_size, embed_size)
        self.drop_prob = dropout_prob

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                            dropout= self.drop_prob, batch_first=True)
        self.dropout = nn.Dropout(self.drop_prob)

        self.fc = nn.Linear(in_features=hidden_size, 
                            out_features=vocab_size)
        
        self.init_weights()


    def forward(self, features, captions):
        captions = captions[:, :-1]
        captions = self.caption_embeddings(captions)
        
        #concatenate features extracted and captions 
        concat = torch.cat((features.unsqueeze(1), captions), 1)

        output, hidden = self.lstm(concat)
        output = self.dropout(output)
        output = self.fc(output)

        return output
    
    def sample(self, inputs, states=None, max_len=20):
          " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
          pass

      
    def init_weights(self):
        """init for fully connected layer"""
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-1, 1)


