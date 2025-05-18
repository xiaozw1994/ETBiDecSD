from torch.utils.data import Subset,Dataset
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
import torch
import torchvision
import network.network as net
import network.decouple_loss  as losses
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import processing.config as cfg
import random
import numpy as np
from sklearn.metrics import f1_score
import json
import sys
import time

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class LoadData(Dataset):
    def __init__(self,train_x,train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.len = len(self.train_x)

    def __getitem__(self,index):
        return self.train_x[index],self.train_y[index]

    def __len__(self):
        return self.len

def getAverage(lists,output):
    sun_weaghts = 0
    for value in lists:
        value = value.detach()
        sun_weaghts += value
    sun_weaghts += output
    return sun_weaghts/ len(lists)

def getWeightedvalue(lists,output):
    sun_weaghts = 0
    for value in lists:
        value = value.detach()
        sun_weaghts += value
    pervalue = 0
    sun_weaghts += output
    for value in lists:
        pervalue += value * (value /sun_weaghts  )
    pervalue += output * (output/sun_weaghts)
    return pervalue

def count_parameters(model):
    """统计模型的可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input_shape, device):
    """统计模型的浮点运算次数(FLOPs)"""
    from thop import profile  # 需要安装thop包: pip install thop
    dummy_input = torch.rand(1, *input_shape).to(device)
    flops, _ = profile(model, inputs=(dummy_input,))
    return flops



r = cfg.loss_coefficient
Temperate = cfg.temperate
Beta = cfg.beta
lr = cfg.learning_rate 
input_num = cfg.univarite_dimension
epoches = cfg.epoch
weight_decay = cfg.weight_decay
Alph = cfg.alpha

def print_progress_bar(epoch, total_epochs, loss, length=50):
    """Prints a progress bar with training information"""
    progress = (epoch + 1) / total_epochs
    filled_length = int(length * progress)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\rEpoch {epoch + 1}/{total_epochs} |{bar}| {progress * 100:.1f}% | Loss: {loss:.4f}')
    sys.stdout.flush()

def train_and_test(class_name,train_loader,test_loader,num_classes,length,device):
    model = net.FCN(input_num,num_classes)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    
    # Initialize SOTA tracking
    SOTA = {
        'accuracy': 0.1,
        'f1': 0.0
    }
    
    start_time = time.time()
    
    for epoch in range(epoches):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            output,lists = model(images)
            
            #loss_classifier = losses.dkd_loss(lists[0],output,labels,Alph,Beta,Temperate) + \
            #                losses.dkd_loss(lists[1],output,labels,Alph,Beta,Temperate) + \
            #                losses.dkd_loss(lists[2],output,labels,Alph,Beta,Temperate) + \
            #                losses.dkd_loss(output,lists[0],labels,Alph,Beta,Temperate) + \
            #                losses.dkd_loss(output,lists[1],labels,Alph,Beta,Temperate) + \
            #                losses.dkd_loss(output,lists[2],labels,Alph,Beta,Temperate)
            #loss_Bied =  losses.dkd_loss(lists[0],output,labels,Alph,Beta,Temperate)+ losses.dkd_loss(lists[1],output,labels,Alph,Beta,Temperate)+ losses.dkd_loss(lists[2],output,labels,Alph,Beta,Temperate)+losses.dkd_loss(output,lists[0],labels,Alph,Beta,Temperate)+ losses.dkd_loss(output,lists[1],labels,Alph,Beta,Temperate)+ losses.dkd_loss(output,lists[2],labels,Alph,Beta,Temperate)
            
            trasive_loss_classifier = losses.dkd_loss(lists[2],output,labels,Alph,Beta,Temperate)+ losses.dkd_loss(output,lists[2],labels,Alph,Beta,Temperate)+ losses.dkd_loss(lists[2],lists[1],labels,Alph,Beta,Temperate)+losses.dkd_loss(lists[1],lists[2],labels,Alph,Beta,Temperate)+ losses.dkd_loss(lists[1],lists[0],labels,Alph,Beta,Temperate)+ losses.dkd_loss(lists[0],lists[1],labels,Alph,Beta,Temperate)
            weight_ensemble = getAverage(lists,output)
            loss_weight_ensemble_loss = losses.dkd_loss(lists[0],weight_ensemble,labels,Alph,Beta,Temperate)+ losses.dkd_loss(lists[1],weight_ensemble,labels,Alph,Beta,Temperate)+ losses.dkd_loss(lists[2],weight_ensemble,labels,Alph,Beta,Temperate)+losses.dkd_loss(weight_ensemble,lists[0],labels,Alph,Beta,Temperate)+ losses.dkd_loss(weight_ensemble,lists[1],labels,Alph,Beta,Temperate)+ losses.dkd_loss(weight_ensemble,lists[2],labels,Alph,Beta,Temperate)
            #loss_weight_output_loss =  losses.dkd_loss(output,weight_ensemble,labels,Alph,Beta,Temperate) +  losses.dkd_loss(weight_ensemble,output,labels,Alph,Beta,Temperate)
            #loss_weight_ensemble_loss += loss_weight_output_loss
            loss_trans = trasive_loss_classifier + loss_weight_ensemble_loss
            loss = loss_func(output, labels)
            loss_c1 = loss_func(lists[0],labels)
            loss_c2 = loss_func(lists[1],labels)
            loss_c3 = loss_func(lists[2],labels)
            loss_ensemble = loss_func(weight_ensemble,labels)
            loss = loss + loss_c1 + loss_c2 + loss_c3 + loss_ensemble


            #loss = loss_func(output, labels)
            #loss_c1 = loss_func(lists[0],labels)
            #loss_c2 = loss_func(lists[1],labels)
            #loss_c3 = loss_func(lists[2],labels)
            #loss = loss + loss_c1 + loss_c2 + loss_c3
            loss = (1-r)*loss + r*loss_trans
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)  
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # Print progress every 50 epochs or on the last epoch
        if (epoch + 1) % 50 == 0 or (epoch + 1) == epoches:
            print_progress_bar(epoch, epoches, avg_loss)
            if (epoch + 1) % 100 == 0 or (epoch + 1) == epoches:
                print()  # New line for better readability of the 100-epoch loss reports
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Final evaluation after training completes
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output,_ = model(images)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = sum(1 for x,y in zip(all_preds, all_labels) if x == y) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Update SOTA
    SOTA['accuracy'] = accuracy
    SOTA['f1'] = f1
    
    # Save model if it's the best so far
    if accuracy > SOTA['accuracy']:
        torch.save(model.state_dict(), "SaveModel/"+class_name+".pkl")
        print(f"\nThe {class_name} model achieved: Accuracy={100*accuracy:.2f}%, F1={100*f1:.2f}%")
    
    return json.dumps(SOTA)