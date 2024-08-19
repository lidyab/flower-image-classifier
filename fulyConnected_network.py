import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def pretrained_ntk():
    ''' To extract input size from selected pretrained network '''
    pre_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Freeze the convolutional layers
    for param in pre_model.features.parameters():
        param.requires_grad = False

    return pre_model


class Classifier(nn.Module):
    def __init__(self, n_hidden, n_output, device ='cpu', drop_p=0.5):
        super().__init__()

        n_features = pretrained_ntk().classifier[0].in_features
         
        self.hidden = nn.ModuleList([nn.Linear(n_features, n_hidden[0])])
         # variable number of more hidden layers
        layer_sizes = zip(n_hidden[:-1], n_hidden[1:])
        self.hidden.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(n_hidden[-1], n_output)
    
        self.dropout = nn.Dropout(p=drop_p)

        self.to(device)

    def forward(self, x):
        x = pretrained_ntk().features(x)
        x = x.view(x.size(0), -1)
        
        for each in self.hidden:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)

        return x
