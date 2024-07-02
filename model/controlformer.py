import torch
import torch.nn as nn
import torchvision.models as models
import clip
from transformers import CLIPModel, CLIPProcessor
import torch

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
    def __init__(self,device,output_dim=512):
        super(ImageEmbedding, self).__init__()
        # You can replace this with any other suitable architecture
        self.device = device
        cachedResnet = models.resnet18(pretrained=True)
        for param in cachedResnet.parameters():
            param.requires_grad = False
        cachedResnet.fc = nn.Linear(512, output_dim)
        for param in cachedResnet.fc.parameters():
            param.requires_grad = True
            
        # cachedResnet.fc = nn.Identity()
        self.cnn = cachedResnet
        # self.cnn.fc = nn.Linear(self.cnn.fc.in_features, output_dim)
        
    def forward(self, x):
        try:
            x = torch.stack(x,axis=0)
        except:
            x=x
        x = x.to(self.device)
        return self.cnn(x)
    
import torch.nn.functional as F

class SimpleCNN(nn.Module):
        def __init__(self, input_shape=(480, 480, 3), output_dim=512, device='cuda'):
            super(SimpleCNN, self).__init__()
            self.device = device
            # Convolutional layers with pooling and batch normalization
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3) # 480x480x3 -> 240x240x32
            self.bn1 = nn.BatchNorm2d(32)
            
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2) # 240x240x32 -> 120x120x64
            self.bn2 = nn.BatchNorm2d(64)
            
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) # 120x120x64 -> 60x60x128
            self.bn3 = nn.BatchNorm2d(128)
            
            self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1) # 60x60x128 -> 30x30x256
            self.bn4 = nn.BatchNorm2d(256)
            
            self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1) # 30x30x256 -> 15x15x512
            self.bn5 = nn.BatchNorm2d(512)
            
            # Global average pooling layer
            self.global_avg_pool = nn.AdaptiveAvgPool2d((3, 3))
            
            # Fully connected layer
            self.fc = nn.Linear(512 * 3 * 3, output_dim)
            
        def forward(self, x):
            x = x.to(self.device)
            # Convolutional layers with ReLU activations, batch normalization, and pooling
            x = F.relu(self.bn1(self.conv1(x)))
            
            x = F.relu(self.bn2(self.conv2(x)))
            
            x = F.relu(self.bn3(self.conv3(x)))
            
            x = F.relu(self.bn4(self.conv4(x)))
            
            x = F.relu(self.bn5(self.conv5(x)))
            x = self.global_avg_pool(x)

            # Flatten the output
            x = x.view(x.size(0), -1)
            
            # Fully connected layer
            x = self.fc(x)
            
            return x


class ImageEmbeddingCNN(nn.Module):
    def __init__(self, device, input_channels=3, embedding_size=512):
        super(ImageEmbeddingCNN, self).__init__()
        self.input_channels = input_channels
        self.embedding_size = embedding_size
        self.device = device
        self.features = nn.Sequential(
            # Convolutional Block 1
            nn.Conv2d(input_channels, 16, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolutional Block 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=4),
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(32 * 30 * 30, self.embedding_size),  # Adjust the input size based on your image dimensions after convolutions
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.to(self.device)
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten the feature map
        embedding = self.fc(features)
        return embedding


class ImageEmbeddingClip(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(ImageEmbeddingClip, self).__init__()
        self.device = device

        model_name = "openai/clip-vit-base-patch32"  # Choose a suitable model size
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)  # Explicitly move model to device
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def preprocess_image(self, image_tensor):
        inputs = self.processor(images=image_tensor, return_tensors="pt")
        inputs = inputs.to(self.device)  # Move preprocessed data to device for GPU usage
        return inputs

    def get_image_embeddings(self, preprocessed_images):
        with torch.no_grad():  # Disable gradient calculation for efficiency
            image_features = self.model.get_image_features(**preprocessed_images)
            return image_features # Access the image embedding tensor

    def forward(self, x):
        preprocessed_images = self.preprocess_image(x)
        image_embeddings = self.get_image_embeddings(preprocessed_images)
        return image_embeddings
        
        image = self.preprocessInput(x)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        
        return image_features

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
        # self.imageEmbedding = ImageEmbedding(self.device)
        self.imageEmbedding = ImageEmbeddingClip(self.device)
        self.conditioning_process = nn.Linear(512*3, d_model)

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


    # def forward(self....): # logic for CNN
    #     # x = x.to(self.device)
    #     img_condition = torch.stack(img_condition)
    #     batch_size = img_condition.size(0)

    #     img_conditions = img_condition.view(batch_size * 3, img_condition.size(2), img_condition.size(3), img_condition.size(4))  # Shape: (batch_size * 3, C, H, W)

        
        
    #     embeddings = self.imageEmbedding(img_conditions)  # Shape: (batch_size * 3, 512)

    #     concatenated_condition = embeddings.view(batch_size, -1)  # Shape: (batch_size, 3 * 512)

    #     condition_embedding = self.conditioning_process(concatenated_condition)
    #       ...rest of the code

    def forward(self, x, img_condition):
        # Initial processing of the condition
        # x = x.to(self.device)
        img_condition = torch.stack(img_condition)
        
        batch_size = img_condition.size(0)

        img_conditions = img_condition.view(batch_size * 3, img_condition.size(2), img_condition.size(3), img_condition.size(4))  # Shape: (batch_size * 3, C, H, W)

        
        
        embeddings = self.imageEmbedding(img_conditions)  # Shape: (batch_size * 3, 512)

        concatenated_condition = embeddings.view(batch_size, -1)  # Shape: (batch_size, 3 * 512)

        condition_embedding = self.conditioning_process(concatenated_condition)

        
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
            for param in layer.parameters():
                param.requires_grad = True