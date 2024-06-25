import torch
import torch.nn as nn
import torchvision.models as models

class ZeroConvBlock(nn.Module):
    def __init__(self, input, output):
        super(ZeroConvBlock, self).__init__()
        self.conv = nn.Conv1d(input, output, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.constant_(self.conv.weight, 0)
        # nn.init.constant_(self.conv.bias, 0)
        # nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)

class ImageEmbedding(nn.Module):
    def __init__(self,device):
        super(ImageEmbedding, self).__init__()
        # You can replace this with any other suitable architecture
        self.device = device
        cachedResnet = models.resnet18(pretrained=True)
        for param in cachedResnet.parameters():
            param.requires_grad = False 
        cachedResnet.fc = nn.Identity()
        self.cnn = cachedResnet
        # self.cnn.fc = nn.Linear(self.cnn.fc.in_features, output_dim)
        
    def forward(self, x):
        try:
            x = torch.stack(x,axis=0)
        except:
            print('fed as img not tuple')
        x = x.to(self.device)
        with torch.no_grad(): # We don't need this remove it
            return self.cnn(x)


class ModifiedTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model ,nhead ,dim_feedforward ,dropout ,activation
):
        super(ModifiedTransformerEncoder, self).__init__()  
        self.nheads = nhead
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imageEmbedding = ImageEmbedding(self.device)
        

        self.inputConv = ZeroConvBlock(d_model, d_model)

        # self.inputConv = ZeroConvBlock(image_condition.shape()[-1],self.d_model)

        self.originalLayers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=self.d_model,
                                                              nhead=self.nheads,
                                                              dim_feedforward=self.dim_feedforward,
                                                              dropout=self.dropout,
                                                              activation=self.activation) for _ in range(num_layers)])
        self.trainableLayers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=self.d_model,
                                                              nhead=self.nheads,
                                                              dim_feedforward=self.dim_feedforward,
                                                              dropout=self.dropout,
                                                              activation=self.activation) for _ in range(num_layers)])
        
        self.zeroConvLayers = nn.ModuleList([ZeroConvBlock(self.d_model, self.d_model) for _ in range(num_layers)])

        # Check if GPU is available
        

        # Move the model to the chosen device
        # self.to(self.device)  

        # Set requires_grad to False for the parameters of the original layers
        for layer in self.originalLayers:
            for param in layer.parameters():
                param.requires_grad = False

    def loadCondition(self, condition):
        self.image_condition = condition.to(self.device)
        # self.image_condition = condition

    def forward(self, x, img_condition):
        # Initial processing of the condition
        # x = x.to(self.device)
        condition_embedding = self.imageEmbedding(img_condition)
        
        condition_embedding = condition_embedding.view(1, -1).repeat(x.size(0), 1).view(x.size(0), x.size(1), -1)
        # Apply the ZeroConvBlock
        condition_embedding = self.inputConv(condition_embedding.permute(1, 2, 0)).permute(2, 0, 1)

        trainableOutput = x + condition_embedding
        
        originalOutput = x
        
        for i in range(len(self.trainableLayers)):
            originalLayer = self.originalLayers[i]
            trainableLayer = self.trainableLayers[i]
            convBlock = self.zeroConvLayers[i]

            originalIntermediate = originalLayer(originalOutput)
            
            trainableOutput = trainableLayer(trainableOutput)
            
            convOutput = convBlock(trainableOutput.permute(1, 2, 0)).permute(2, 0, 1)
            
            originalOutput = originalIntermediate + convOutput
        
        return originalOutput
    
    def load_original_weights(self, state_dict):
        
        # Iterate over each layer in originalLayers and load the corresponding weights
        for i, layer in enumerate(self.originalLayers):
            # Construct the keys for the encoder layer's parameters
            layer_state_dict = {k.replace(f'seqTransEncoder.layers.{i}.', ''): v 
                                for k, v in state_dict.items() if f'seqTransEncoder.layers.{i}.' in k}
            layer.load_state_dict(layer_state_dict, strict=True)
            for param in layer.parameters():
                param.requires_grad = False
            
        # trainable layers should be a copy of the original and then finetuned
        for i, layer in enumerate(self.trainableLayers):
            # Construct the keys for the encoder layer's parameters
            layer_state_dict = {k.replace(f'seqTransEncoder.layers.{i}.', ''): v 
                                for k, v in state_dict.items() if f'seqTransEncoder.layers.{i}.' in k}
            layer.load_state_dict(layer_state_dict, strict=True)