import os 

data_lib_url = "http://timeseriesclassification.com/dataset.php"
####dataset repo
univarite_dimension = 1
store_local_lib = "data/"
#### GPU index used
GPU_number = 0
#### dataset name
dataset_name = "Beef"
###  coeffecient for loss function
loss_coefficient = 0.1
### a scalarable temporate for distillation
temperate = 1.0
#####  the weight of dual non-target class loss
beta = 1.0
##### the weight of dual target class loss
alpha = 1.0
###### learning rate
learning_rate = 0.0001
##### weight_decay value
weight_decay = 0.0005
#### training epoch
epoch = 200