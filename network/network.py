import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


class AveragePoolClassifier(nn.Module):
    def __init__(self,channels,num_classess):
        super(AveragePoolClassifier,self).__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(channels,num_classess)
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias,val=0)
    def forward(self,x):
        output = self.avg(x)
        output = output.view(output.size(0),-1)
        output = nn.Dropout(p=0.5)(output)
        output = self.linear(output)
        return output

##  padding = (kernel-1)*dalidat
##
class DailatedBlock(nn.Module):
    def __init__(self,input_channel,output_channel,kernel):
        super(DailatedBlock,self).__init__()
        padding  = (kernel-1)//2
        self.conv1 = nn.Conv1d( input_channel,output_channel,kernel_size=kernel,stride=1,padding=padding)
        self.bn = nn.BatchNorm1d(output_channel)
        self.LeakReLu = nn.ReLU()
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias,val=0)
        nn.init.constant_(self.bn.weight,val=1)
        nn.init.constant_(self.bn.bias,val=0)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.LeakReLu(x)
        return x
    
#################
######
################
class DailatedBlockAdd(nn.Module):
    def __init__(self,input_channel,output_channel,kernel):
        super(DailatedBlockAdd,self).__init__()
        padding  = (kernel) //2
        self.conv1 = nn.Conv1d(input_channel,output_channel,kernel_size=kernel,stride=1,padding=padding)
        self.bn = nn.BatchNorm1d(output_channel)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias,val=0)
        nn.init.constant_(self.bn.weight,val=1)
        nn.init.constant_(self.bn.bias,val=0)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        return x

class ResidualBlock1(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(ResidualBlock1,self).__init__()
        self.block1 = DailatedBlock(input_channel,output_channel,8)
        self.block2 = DailatedBlock(output_channel,output_channel*2,5)
        self.block3 = DailatedBlockAdd(output_channel*2,output_channel,8)
        self.block4 = DailatedBlockAdd(input_channel,output_channel,1)
        self.LeakReLU = nn.ReLU()
    def forward(self,x):
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        add = self.block4(x)
        output = self.LeakReLU(output+add)
        return output
class ResidualNet(nn.Module):
    def __init__(self,input_channel,output_channel,num_classes):
        super(ResidualNet,self).__init__()
        self.block1 = ResidualBlock1(input_channel,output_channel)
        self.block2 = ResidualBlock1(output_channel,output_channel*2)
        self.block3 = ResidualBlock1(output_channel*2,output_channel*2)
        self.block4 = ResidualBlock1(output_channel*2,output_channel*2)
        self.avg =  nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(output_channel*2,128)
        self.linear2 = nn.Linear(128,num_classes)
        self.classifier1 = AveragePoolClassifier(output_channel,num_classes)
        self.classifier2 = AveragePoolClassifier(output_channel*2,num_classes)
        self.classifier3 = AveragePoolClassifier(output_channel*2,num_classes)
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias,val=0)
        nn.init.constant_(self.linear2.bias,val=0)
    def forward(self,x):
        results = []
        block1 = self.block1(x)
        classifer1 = self.classifier1(block1)
        block2 = self.block2(block1)
        classifier2 = self.classifier2(block2)
        block3 = self.block3(block2)
        classifier3 = self.classifier3(block3)
        block4 = self.block4(block3)
        results.append(classifer1)
        results.append(classifier2)
        results.append(classifier3)
        avg = self.avg(block4)
        ful1 = self.linear1(avg.view(avg.size(0),-1))
        drop = nn.Dropout(p=0.5)(ful1)
        res = self.linear2(drop)
        return res,results

class ResiualBase(nn.Module):
    def __init__(self,input_channel,inter_channel,output_channel):
        super(ResiualBase,self).__init__()
        self.conv1_1 = nn.Conv1d(input_channel,inter_channel,8,stride=1,padding=4)
        self.BN_1 = nn.BatchNorm1d(inter_channel)
        self.relu = nn.ReLU()
        self.conv1_2 = nn.Conv1d(inter_channel,inter_channel,5,stride=1,padding=2)
        self.BN_2 = nn.BatchNorm1d(inter_channel)
        self.conv1_3 = nn.Conv1d(inter_channel,output_channel,3,stride=1,padding=1)
        self.BN_3 = nn.BatchNorm1d(output_channel)
        self.conv_app = nn.Conv1d(input_channel,output_channel,2,stride=1,padding=1)
        nn.init.kaiming_normal_(self.conv1_1.weight)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        nn.init.kaiming_normal_(self.conv1_3.weight)
        nn.init.kaiming_normal_(self.conv_app.weight)
    def forward(self,x):
        x1 = self.conv1_1(x)
        bn1 = self.BN_1(x1)
        relu1 = self.relu(bn1) 
        x2 = self.conv1_2(relu1)
        bn2 = self.BN_2(x2)
        relu2 = self.relu(bn2)
        x3 = self.conv1_3(relu2)
        bn3 = self.BN_3(x3)
        x4 = self.conv_app(x)
        bn4 = self.BN_3(x4)
        #print(bn3.shape,bn4.shape)
        result = bn3+bn4
        result = self.relu(result)
        return result 

class ResNet(nn.Module):
    def __init__(self,input_channel,num_classes):
        super(ResNet,self).__init__()
        self.resblock1 = ResiualBase(input_channel,128,128)
        self.resblock2 = ResiualBase(128,128,128)
        self.resblock3 = ResiualBase(128,256,256)
        self.resblock4 = ResiualBase(256,256,256)
        self.classifer1 = AveragePoolClassifier(128,num_classes)
        self.classifer2 = AveragePoolClassifier(128,num_classes)
        self.classifer3 = AveragePoolClassifier(256,num_classes)
        self.classifer4 = AveragePoolClassifier(256,num_classes)
    def forward(self,x):
        infor = []
        x1 = self.resblock1(x)
        x2 = self.resblock2(x1)
        x3 = self.resblock3(x2)
        x4 = self.resblock4(x3)
        res1 = self.classifer1(x1)
        res2 = self.classifer2(x2)
        res3 = self.classifer3(x3)
        res4 = self.classifer4(x4)
        infor.append(res1)
        infor.append(res2)
        infor.append(res3)
        return res4,infor


class BaseInception(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(BaseInception,self).__init__()
        self.conv_1 = nn.Conv1d(input_channel,output_channel,11,stride=1,padding=5)
        self.conv_2 = nn.Conv1d(input_channel,output_channel,9,stride=1,padding=4)
        self.conv_3 = nn.Conv1d(input_channel,output_channel,17,stride=1,padding=8)
        self.conv_4 = nn.Conv1d(input_channel,output_channel,5,stride=1,padding=2)
        self.conv_5 = nn.Conv1d(input_channel,output_channel,3,stride=1,padding=1)
        self.head = nn.Conv1d(input_channel,output_channel,2,stride=1,padding=1)
        self.max1 = nn.MaxPool1d(2,stride=1,padding=0)
        self.bn = nn.BatchNorm1d( int( output_channel*6))
        self.relu = nn.ReLU()
    def forward(self,x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)
        x5 = self.conv_5(x)
        x6 = self.head(x)
        x6 = self.max1(x6)
        #print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape,x6.shape)
        res = torch.cat([x1,x2,x3,x4,x5,x6],axis=1)
        res = self.bn(res)
        res = self.relu(res)
        return res

    
class InceptionTime(nn.Module):
    def __init__(self,input_dims,num_classes):
        super(InceptionTime,self).__init__()
        self.incep_1 = BaseInception(1,32)
        self.incep_2 = BaseInception(192,32)
        self.incep_3 = BaseInception(192,64)
        self.incep_4 = BaseInception(384,64)
        self.classifier1 = AveragePoolClassifier(192,num_classes)
        self.classifier2 = AveragePoolClassifier(192,num_classes)
        self.classifier3 = AveragePoolClassifier(384,num_classes)
        self.classifier4 = AveragePoolClassifier(384,num_classes)
    def forward(self,x):
        lists = []
        x1 = self.incep_1(x)
        x2 = self.incep_2(x1)
        x3 = self.incep_3(x2)
        x4 = self.incep_4(x3)
        res1 = self.classifier1(x1)
        lists.append(res1)
        res2 = self.classifier2(x2)
        lists.append(res2)
        res3 = self.classifier3(x3)
        lists.append(res3)
        result = self.classifier4(x4)
        return result,lists




class FCN(nn.Module):
    def __init__(self,input_dims,num_classes):
        super(FCN,self).__init__()
        self.conv1 = nn.Conv1d(input_dims,128,11,stride=1,padding=5)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 =nn.Conv1d(128,256,13,1,padding=6)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256,256,11,1,5)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256,256,11,1,5)
        self.bn4 = nn.BatchNorm1d(256)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.ReLU = nn.ReLU()
        self.linear1 = nn.Linear(256,256)
        self.linear2 = nn.Linear(256,num_classes)
        self.classifier1 = AveragePoolClassifier(128,num_classes)
        self.classifier2 = AveragePoolClassifier(256,num_classes)
        self.classifier3 = AveragePoolClassifier(256,num_classes)
        
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.constant_(self.bn1.weight,val=1)
        nn.init.constant_(self.bn2.weight,1)
        nn.init.constant_(self.bn3.weight,val=1)
        nn.init.constant_(self.bn4.weight,val=1)
        nn.init.constant_(self.conv1.bias,val=0)
        nn.init.constant_(self.bn1.bias,val=0)
        nn.init.constant_(self.conv2.bias,val=0)
        nn.init.constant_(self.bn2.bias,val=0)
        nn.init.constant_(self.conv3.bias,val=0)
        nn.init.constant_(self.conv4.bias,val=0)
        nn.init.constant_(self.bn3.bias,val=0)
        nn.init.constant_(self.bn4.bias,val=0)
        nn.init.constant_(self.linear1.bias,val=0)
        nn.init.constant_(self.linear2.bias,val=0)
    def forward(self,x):
        result = []
        times = []
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.ReLU(bn1)
        classifier1 = self.classifier1(relu1)
        result.append(classifier1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.ReLU(bn2)
        classifier2 = self.classifier2(relu2)
        result.append(classifier2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.ReLU(bn3)
        classifier3 = self.classifier3(relu3)
        result.append(classifier3)
        conv4 = self.conv4(relu3)
        bn4 = self.bn4(conv4)
        relu4 = self.ReLU(bn4)
        avg = self.avg(relu4)
        ful1 = self.linear1(avg.view(avg.size(0),-1))
        drop = nn.Dropout(p=0.5)(ful1)
        res = self.linear2(drop)
        return res,result

'''
vs = torch.rand((12,1,34))
model = FCN(1,5)
ts,lists = model(vs)
print(ts.shape)
for vs in lists:
    print(vs.shape)
'''
