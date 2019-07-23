#!/usr/bin/python
#-*- encoding:utf-8 -*-

import numpy as np 
import time
import os
import random
import pandas as pd 
import prepare
import torch
import nets
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

print('\nreading datset, waiting ...... ')
start = time.time()
"""
data_type: 1 header only,2 payload only, 3 header and payload
"""
data_type = 3
header_payload = False
if data_type == 3:
	header_payload=True
train_data_type = prepare.TrainDataSetHeader(data_type=data_type)

x_train,y_train,x_test,y_test = train_data_type.get_item()
print('\nfinish reading dataset, cost time :',time.time() - start)

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
random.seed(1)

# GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

#super parameters

num_workers = 0
NUM_CLASSES = 12
batch_size = 256
workers = 0
lr = 1e-4
lr_decay = 5
weight_decay = 1e-4

stage = 0
start_epoch = 0
stage_epochs = [5,3,2]  
total_epochs = sum(stage_epochs)
best_precision = 0
lowest_loss = 100
print_freq = 1
evaluate = False
resume = False
train_val = False

model_type = 'CNN'

if not os.path.exists('../model/%s' %model_type):
	os.makedirs('../model/%s' %model_type)

if not os.path.exists('../result/%s' %model_type):
	os.makedirs('../result/%s' %model_type)

if not os.path.exists('../result/%s.txt' %model_type):
	with open('../result/%s.txt' %model_type,'w') as acc_file:
		pass
	
with open('../result/%s.txt' %model_type,'a') as acc_file:
	acc_file.write('\n%s %s\n' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), model_type))

#模型恢复训练
if resume:
	checkpoint_path = '../model/checkpoint.pth.tar' %model_type
	if os.path.isfile(checkpoint_path):
		print('loading checkpoint {}'.format(checkpoint_path))
		checkpoint = torch.load(checkpoint_path)
		start_epoch = checkpoint['epoch'] + 1
		best_precision = checkpoint['best_precision']
		lowest_loss = checkpoint['lowest_loss']
		stage = checkpoint['stage']
		lr = checkpoint['lr']
		model.load_state_dict(checkpoint['state_dict'])

		if start_epoch in np.cumsum(stage_epochs)[:-1]:
			stage += 1
			optimizer = prepare.adjust_learning_rate(model,weight_decay,lr,lr_decay)
			model.load_state_dict(torch.load('../model/checkpoint.pth.tar' %model_type)['state_dict'])
		print('loaded checkpoint (epoch {})'.format(checkpoint['epoch']))
	else:
		print('no checkpoint found at {}'.format(checkpoint_path))


print('\npreparing data, wait wait wait ...')
data_train,data_val,label_train,label_val = train_test_split(x_train,y_train,test_size=0.05,random_state=930802)
my_train_data = np.concatenate((data_train,label_train.reshape(len(label_train),1)),axis=1)
my_val_data = np.concatenate((data_val,label_val.reshape(len(label_val),1)),axis=1)
my_test_data = np.concatenate((x_test,y_test.reshape(len(y_test),1)),axis=1)

train_data = prepare.DealDataSet(my_train_data,header_payload=header_payload)
validate_data = prepare.DealDataSet(my_val_data,header_payload=header_payload)
test_data = prepare.DealDataSet(my_test_data,header_payload=header_payload)
print('trian dataset shape: ',train_data.xshape)
print('validate dataset shape: ',validate_data.xshape)
print('test dataset shape: ',test_data.xshape)
train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
validate_loader = DataLoader(validate_data,batch_size=batch_size*2,shuffle=False,num_workers=num_workers,pin_memory=True)
test_loader = DataLoader(test_data,batch_size=int(batch_size*2),shuffle=False,num_workers=num_workers,pin_memory=True)

# normalize = transforms.Normalize(mean=[0.488],std=[0.229])

# total models
my_models = {
	'FCNN':my_new_nets.CNN(num_class=NUM_CLASSES,head_payload=header_payload),
	'CNN_NORMAL':my_new_nets.CNN_NORMAL(num_class=NUM_CLASSES,head_payload=header_payload),
	'LSTM':my_new_nets.LSTM(num_class=NUM_CLASSES,head_payload=header_payload),
	'CNN_LSTM':my_new_nets.CNN_LSTM(num_class=NUM_CLASSES,head_payload=header_payload),
	'PARALLELNETS':my_new_nets.PARALLELNETS(num_class=NUM_CLASSES,head_payload=header_payload),
	'CROSS_CNN':my_new_nets.CROSS_CNN(num_class=NUM_CLASSES,head_payload=header_payload),
	'PARALLEL_CROSS_CNN':my_new_nets.PARALLEL_CROSS_CNN(num_class=NUM_CLASSES,head_payload=header_payload),
	'DPARALLEL_CROSS_NET':my_new_nets.DPARALLEL_CROSS_NET(num_class=NUM_CLASSES,head_payload=header_payload),
	'PARALLEL_CROSS_CNN_ADD':my_new_nets.PARALLEL_CROSS_CNN_ADD(num_class=NUM_CLASSES,head_payload=header_payload)
}


train_model = 'CNN_LSTM'
model = my_models['CNN_LSTM']
model = torch.nn.DataParallel(model).cuda()

# summary(model,(1,16,16))
print(model)


loss_type = nn.CrossEntropyLoss().cuda()
#
optimizer = optim.Adam(model.parameters(),lr,weight_decay=weight_decay,amsgrad=True)

optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=True)
train_start = time.time()
if evaluate:
	prepare.validate(validate_loader,model,loss_type,best_precision,lowest_loss)
else:
	for epoch in range(start_epoch,total_epochs):
		#train for one epoch
		prepare.train(train_loader,model,loss_type,optimizer,epoch)
		#evaluate on validate set
		accuracy,avg_loss = prepare.validate(validate_loader,model,loss_type,best_precision,lowest_loss)

		with open('../result/%s.txt' %model_type,'a') as acc_file:
			acc_file.write('Epoch: %2d, Accuracy: %.8f, Loss: %.8f\n' % (epoch, accuracy, avg_loss))

		is_best = accuracy > best_precision
		is_lowest_loss = avg_loss < lowest_loss
		best_precision = max(accuracy,best_precision)
		lowest_loss = min(avg_loss,lowest_loss)
		state = {
		 	'epoch':epoch,
		 	'state_dict':model.state_dict(),
		 	'best_precision':best_precision,
		 	'lowest_loss':lowest_loss,
		 	'stage':stage,
		 	'lr':lr
		}

		prepare.save_checkpoint(state,is_best,is_lowest_loss,model_type)

		if (epoch + 1) in np.cumsum(stage_epochs)[:-1]:
			stage += 1
			optimizer = prepare.adjust_learning_rate(model,weight_decay,lr,lr_decay)
			model.load_state_dict(torch.load('../model/%s/model_best.pth.tar' %model_type)['state_dict'])
			print('\n \nStep next stage .........\n \n')
			with open('../result/%s.txt' % model_type,'a') as acc_file:
				acc_file.write('\n---------------------Step next stage---------------------\n')
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("		finish training cost time: %ss" %(time.time() - train_start))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

with open('../result/%s.txt' % model_type,'a') as acc_file:
	acc_file.write("*** best accuracy: %.8f %s ***\n" %(best_precision,model_type))

with open('../result/best_acc.txt', 'a') as acc_file:
	acc_file.write('%s  * best acc: %.8f  %s\n' % (
	time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), best_precision,model_type))

test_start = time.time()
result,top1_metrics,topk_metrics = prepare.test(test_loader,model,num_class=NUM_CLASSES,topk=5)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("		finish testing cost time: %ss" %(time.time() - test_start))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

torch.cuda.empty_cache()

top1_prob,top1_pred_label,topk_prob,topk_pred_label,actual_label = result[0],result[1],result[2],result[3],result[4]


accuracy,recall,precision,F1_score = top1_metrics[0],top1_metrics[1],top1_metrics[2],top1_metrics[3]
topk_acc,topk_recall,topk_preci,topk_F1 = topk_metrics[0],topk_metrics[1],topk_metrics[2],topk_metrics[3]


conf_mtx= confusion_matrix(actual_label,top1_pred_label)
print('\nConfusion Matrix:')
print(conf_mtx)

def num2str(data):
	str_data = []
	for x in data:
		str_data.append(str(round(x,4)))
	my_str = " ".join(str_data)
	return my_str

total_num = [415,52246,4109,94932,1358,2108,3989,1972,1067,63928,5509,2109]
total_num = np.asarray(total_num)
correct_num = []
for r,w in enumerate(conf_mtx):
	correct_num.append(w[r])
correct_num = np.asarray(correct_num)
accuracy = accuracy_score(actual_label, top1_pred_label)
top1_precision = precision_score(actual_label, top1_pred_label, average=None)
top1_recall = recall_score(actual_label, top1_pred_label, average=None)
top1_f1_score = f1_score(actual_label, top1_pred_label, average=None)
pr = top1_precision * top1_recall
apr = (correct_num / total_num) * top1_precision * top1_recall
target_names = ['class 0', 'class 1', 'class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9','class 10','class 11']
report = classification_report(actual_label, top1_pred_label, target_names=target_names)
print(report)
print("accuracy            :",accuracy)
print('e_accuracy		   :',num2str(correct_num / total_num))
print("precision 		   :",num2str(top1_precision))
print("recall 			   :",num2str(top1_recall))
print('f1-socre 		   :',num2str(top1_f1_score))
print('precision_recall    :',num2str(pr))
print('acc_precision_recall:',num2str(apr))



with open(train_model + '.txt','w',encoding='utf-8') as f:
	f.write("\n****************************  " + train_model + "  ****************************\n")
	f.write(report)
	f.write("\naccuracy :\n")
	f.write(str(round(accuracy,4)))
	f.write("\neach accuracy:\n")
	f.write(num2str(correct_num / total_num))
	f.write('\nprecision :\n')
	f.write(num2str(top1_precision))
	f.write("\nrecall :\n")
	f.write(num2str(top1_recall))
	f.write('\nf1-socre :\n')
	f.write(num2str(top1_f1_score))
	f.write("\n*****************************************************************")

print("done!")
