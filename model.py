import math
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable


class BasicModel(nn.Module):
    def __init__(self,h,w,in_channel,output_length):
        super(BasicModel,self).__init__()
        self.h = h
        self.w = w
        self.in_channel = in_channel
        self.hidden_channel = in_channel * 2
        self.output_length = output_length

        # we hope both h and w are the power of 2
        assert math.log2(h) == round(math.log2(h)), "spectrogram height must be a power of 2"
        assert math.log2(w) == round(math.log2(w)), "spectrogram width must be a power of 2"
        
        self.density = int(math.log2(max(h,w)))
        self.kernel_size = (int(4/(self.w/self.h))-1,4-1)
        self.padding_size = (int(1-(self.w != self.h)),1)

        self.backbone = nn.Sequential()
        self.backbone.add_module('input-conv2d',nn.Conv2d(self.in_channel,self.hidden_channel,self.kernel_size,2,self.padding_size,bias=False))
        self.backbone.add_module('input-bn',nn.BatchNorm2d(self.hidden_channel))
        self.backbone.add_module('input-maxPool',nn.MaxPool2d(self.kernel_size,2,self.padding_size))
        self.backbone.add_module('input-relu',nn.ReLU(True))
        
        for i in range((self.density-3)//2):
            self.backbone.add_module('{0}:inter-{1}-{2}-conv2d'.format(i, self.hidden_channel*(2**i), self.hidden_channel*(2**(i+1))), nn.Conv2d(self.hidden_channel*(2**i),
                                     self.hidden_channel*(2**(i+1)), self.kernel_size, 2, self.padding_size, bias=False))
            self.backbone.add_module(
                '{0}:inter-bn'.format(i), nn.BatchNorm2d(self.hidden_channel*(2**(i+1))))
            self.backbone.add_module(
                '{0}:inter-maxPool'.format(i), nn.MaxPool2d(self.kernel_size, 2, self.padding_size))
            self.backbone.add_module('{0}:inter-relu'.format(i), nn.ReLU(True))
        
        denses = (self.density-3)//2
        cur_h = (self.h // 4) // (4**denses) 
        cur_w = (self.w // 4) // (4**denses) 
      
        self.conv1 = nn.Conv2d(
            self.hidden_channel*(2**((self.density-3)//2)), self.output_length, (cur_h-1, cur_w-1), 2, 0)
        self.classifier = nn.Softmax(dim=1)
        self.loss = nn.MSELoss()
    
    def forward(self,input):
        output1 = self.backbone(input)
        output2 = self.conv1(output1)
        output2 = output2.reshape((int(output2.size()[0]),1,1,self.output_length))
        
        # final_output = self.classifier(torch.squeeze(output2))
        final_output = output2

        return final_output
        

