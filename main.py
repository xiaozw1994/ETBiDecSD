import os 
import time 
import network.network as net
from random import shuffle
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import processing.config as cfg
import processing.data as data
import processing.augmentation as aug
from torch.utils.data import Subset,Dataset
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
import network.decouple_loss as losses
import random
import numpy as np
import argparse
from keras.utils import to_categorical
import json
import psutil
import pynvml  # For GPU metrics (need to install with pip install nvidia-ml-py3)
from network.torch_functions import  setup_seed, LoadData, train_and_test

# Initialize NVML for GPU monitoring
try:
    pynvml.nvmlInit()
    HAS_GPU_METRICS = True
except:
    HAS_GPU_METRICS = False

def get_system_metrics():
    """Get CPU, memory, and GPU metrics"""
    metrics = {
        'cpu_usage': psutil.cpu_percent(),
        'memory_usage': psutil.virtual_memory().percent,
        'gpu_usage': 0,
        'gpu_memory': 0
    }
    
    if HAS_GPU_METRICS:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics['gpu_usage'] = utilization.gpu
            metrics['gpu_memory'] = (memory_info.used / memory_info.total) * 100
        except:
            pass
    
    return metrics

GPU_number = cfg.GPU_number
class_name = cfg.dataset_name

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_number)
# set random seed 
setup_seed(123)
data_dir = os.path.join(cfg.store_local_lib, class_name)

## Load dataset
x_train, y_train = data.readucr(data_dir+"/"+class_name+"_TRAIN")
x_test, y_test = data.readucr(data_dir+"/"+class_name+"_TEST")
num_classes = len(np.unique(y_train))
x_train = data.NormalizationFeatures(x_train)
x_test = data.NormalizationFeatures(x_test)
y_train = data.NormalizationClassification(y_train, num_classes)
y_test = data.NormalizationClassification(y_test, num_classes)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test)
y_train = np.argmax(y_train, axis=-1)
y_test = np.argmax(y_test, axis=-1)

train_batch, train_length = x_train.shape
x_train = x_train.reshape((train_batch, 1, train_length)).astype(np.float32)
test_batch, train_leangth = x_test.shape
x_test = x_test.reshape((test_batch, 1, train_leangth)).astype(np.float32)

train_set = LoadData(x_train, y_train)
test_set = LoadData(x_test, y_test)
batch_size = train_batch // 4
test_batch = test_batch // 4
train_loader = dataloader.DataLoader(dataset=train_set, batch_size=train_batch, shuffle=True)
test_loader = dataloader.DataLoader(dataset=test_set, batch_size=test_batch, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Start monitoring
start_time = time.time()
metrics_samples = []
sample_interval = 10  # Sample metrics every 10 seconds

def sample_metrics():
    metrics = get_system_metrics()
    metrics_samples.append(metrics)
    return metrics

# Print initial metrics
#print("Initial system metrics:")
initial_metrics = sample_metrics()
#print(json.dumps(initial_metrics, indent=2))

# Train and test
sota = train_and_test(class_name, train_loader, test_loader, num_classes, train_length, device)

# Calculate training time
training_time = time.time() - start_time

# Calculate average metrics
avg_metrics = {
    'cpu_usage': np.mean([m['cpu_usage'] for m in metrics_samples]),
    'memory_usage': np.mean([m['memory_usage'] for m in metrics_samples]),
    'gpu_usage': np.mean([m['gpu_usage'] for m in metrics_samples]) if HAS_GPU_METRICS else 0,
    'gpu_memory': np.mean([m['gpu_memory'] for m in metrics_samples]) if HAS_GPU_METRICS else 0
}

# Create final result dictionary
result = {
    'training_time_seconds': training_time,
    'average_metrics': avg_metrics,
    'performance_metrics': json.loads(sota)
}

print("\nFinal Results:")
print(json.dumps(result, indent=2))

# Cleanup NVML if it was initialized
if HAS_GPU_METRICS:
    pynvml.nvmlShutdown()
