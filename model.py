import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchsummary import summary

hp = OmegaConf.load('./config/default.yaml')

class CNN(nn.Module):
    def __init__(self, model_char):
        super(CNN, self).__init__()
        '''
        [Model_char]
        A: Default
        B: 1x1 Conv
        C: Depthwise Conv
        '''
        self.use_SC = hp.model.use_SkipConnect
        #* Stage1 
        self.stage1 = nn.ModuleList()
        self.stage1.append(nn.Conv2d(3, 32, 5, stride = 1, padding = 1))

        if hp.model.use_BN:
            self.stage1.append(nn.BatchNorm2d(32))

        self.stage1.append(nn.ReLU())

        if hp.model.use_Dropout:
            self.stage1.append(nn.Dropout(hp.model.dropout))
        
        self.stage1.append(nn.MaxPool2d(2, stride = 2, padding = 0))

        #* Stage 2
        self.stage2 = nn.ModuleList()

        if model_char == 'A':
            self.stage2.append(nn.Conv2d(32, 64, 3, stride = 1, padding = 1))
            
            if hp.model.use_BN:
                self.stage2.append(nn.BatchNorm2d(64))

            self.stage2.append(nn.ReLU())

            if hp.model.use_Dropout:
                self.stage2.append(nn.Dropout(hp.model.dropout))
            
            self.stage2.append(nn.MaxPool2d(2, stride = 2, padding = 0))
        
        if model_char == 'B':
            channel_list = [16, 16, 64]
            ker_list = [1, 3, 1]
            for i in range(len(ker_list)):
                if i == 0:
                    self.stage2.append(nn.Conv2d(32, channel_list[i], ker_list[i]))
                else:
                    self.stage2.append(nn.Conv2d(channel_list[i-1], channel_list[i], ker_list[i]))
                
                if hp.model.use_BN:
                    self.stage2.append(nn.BatchNorm2d(channel_list[i]))

                self.stage2.append(nn.ReLU())

                if hp.model.use_Dropout:
                    self.stage2.append(nn.Dropout(hp.model.dropout))
            
            self.stage2.append(nn.MaxPool2d(2, stride = 2, padding = 0))

        if model_char == 'C':
            self.stage2.append(nn.Conv2d(32, 64, 3, stride = 1, padding = 1, groups = 32))
            self.stage2.append(nn.Conv2d(64, 64, 1))
            
            if hp.model.use_BN:
                self.stage2.append(nn.BatchNorm2d(64))

            self.stage2.append(nn.ReLU())

            if hp.model.use_Dropout:
                self.stage2.append(nn.Dropout(hp.model.dropout))
            
            self.stage2.append(nn.MaxPool2d(2, stride = 2, padding = 0))    

        
        #* Stage 3
        self.stage3 = nn.ModuleList()
       
        if model_char == 'A':
            self.stage3.append(nn.Conv2d(64, 128, 3, stride = 1, padding = 1))
            
            if hp.model.use_BN:
                self.stage3.append(nn.BatchNorm2d(128))

            self.stage3.append(nn.ReLU())

            if hp.model.use_Dropout:
                self.stage3.append(nn.Dropout(hp.model.dropout))
            
            self.stage3.append(nn.MaxPool2d(2, stride = 2, padding = 0))

        if model_char == 'B':
            channel_list = [32, 32, 128]
            ker_list = [1, 3, 1]
            
            for i in range(len(ker_list)):
                if i == 0:
                    self.stage3.append(nn.Conv2d(64, channel_list[i], ker_list[i]))
                else:
                    self.stage3.append(nn.Conv2d(channel_list[i-1], channel_list[i], ker_list[i]))

                if hp.model.use_BN:
                    self.stage3.append(nn.BatchNorm2d(channel_list[i]))

                self.stage3.append(nn.ReLU())

                if hp.model.use_Dropout:
                    self.stage3.append(nn.Dropout(hp.model.dropout))
                
            self.stage3.append(nn.MaxPool2d(2, stride = 2, padding = 0))

        if model_char == 'C':
            self.stage3.append(nn.Conv2d(64, 128, 3, stride = 1, padding = 1, groups=64))
            self.stage3.append(nn.Conv2d(128, 128, 1))
            
            if hp.model.use_BN:
                self.stage3.append(nn.BatchNorm2d(128))

            self.stage3.append(nn.ReLU())

            if hp.model.use_Dropout:
                self.stage3.append(nn.Dropout(hp.model.dropout))
            
            self.stage3.append(nn.MaxPool2d(2, stride = 2, padding = 0))        
        
        self.cnn_stages = [self.stage1, self.stage2, self.stage3]
        
        if model_char == 'A':
            self.stage4 = nn.Linear(1152, 500)
        if model_char == 'B':
            self.stage4 = nn.Linear(512, 500)
        if model_char == 'C':
            self.stage4 = nn.Linear(1152, 500)
        
        self.stage5 = nn.Linear(500, 10)

    
    def forward(self, x):

        for cnn_stage in self.cnn_stages:
            for layer in cnn_stage:
                if self.use_SC:
                    '''apply skip connection to CNN output'''
                    if "Conv" in str(layer):
                        x = x + layer(x)
                    else:
                        x = layer(x)

                else: 
                    x = layer(x)

        x = x.view(x.size(0), -1)

        x = self.stage4(x)
        x = self.stage5(x)

        return x

if __name__ == '__main__':
    
    hp = OmegaConf.load('./config/default.yaml')
    device = torch.device('cuda') if torch.cuda.is_available() else cpu
    net = CNN('A').to(device)
    
    summary(net, (3,32,32))

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(pytorch_total_params)