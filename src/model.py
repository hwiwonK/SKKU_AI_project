"""# resnet 최종"""
import torch.nn as nn

class Residual_block(nn.Module): 
  def __init__(self, input_channel, output_channel, stride = 1): 
    super(Residual_block,self).__init__() # Residual Block 

    self.residual_block = nn.Sequential( nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
                                    nn.BatchNorm2d(output_channel), 
                                    nn.ReLU(), 
                                    nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(output_channel)
                                    ) 

    self.relu = nn.ReLU() 
    self.shortcut = nn.Sequential() #input, output size 같아야 함

    if stride != 1:
      self.shortcut = nn.Sequential(
          nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(output_channel)
      )


  def forward(self, x): 
    result = self.residual_block(x) # F(x) 
    result = result + self.shortcut(x)
    result = self.relu(result) 
    return result

class ResNet(nn.Module):
    def __init__(self, input_channel=3 , num_classes=15):
        self.num_classes = num_classes
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Sequential(
                                        Residual_block(64, 64, stride=1),
                                        Residual_block(64, 64, stride=1)
        )
        
        self.conv3 = nn.Sequential(
                                        Residual_block(64, 128, stride=2),
                                        Residual_block(128, 128, stride=1),
        )

        self.conv4 = nn.Sequential(
                                        Residual_block(128, 256, stride=2),
                                        Residual_block(256, 256, stride=1)
        )
        
        self.conv5 = nn.Sequential(
                                        Residual_block(256, 512, stride=2),
                                        Residual_block(512, 512, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x