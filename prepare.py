#-*- coding:utf-8 -*-
#!/usr/bin/python
#-*- encoding:utf-8 -*-

import numpy as np 
import time
import pandas as pd 
import torch
import tqdm
import shutil
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def train(train_loader,model,loss_type,optimizer,epoch):
	'''
	train_data: train data,include label in the last dimension
	model: net model 
	loss: type of loss to be used
	optimizer: training optimmizer
	epoch: current epoch
	return: None
	'''
	batch_time =AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	acc = AverageMeter()
	batch_size = 16
	num_class = 12
	# switch to train mode///why?
	model.train()

	end = time.time()

	for step,(feature,label) in enumerate(train_loader):
		feature = Variable(feature).cuda(async=True)
		label = Variable(label).cuda(async=True)


		y_pred = model(feature)
	
		loss = loss_type(y_pred,label.squeeze())
		losses.update(loss.item(),feature.size(0))

		pred_acc,pred_count = accuracy(y_pred.data,label,topk=(1,1))
		acc.update(pred_acc,pred_count)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


		batch_time.update(time.time() - end,1)
		end = time.time()


		if step % 10 == 0:
			print('opoch:[{0}][{1}/{2}]\t'
				  'Time:{batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data:{data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss:{loss.val:.4f} ({loss.avg:.4f})\t'
				  'Accuracy:{acc.val:.3f} ({acc.avg:.3f})'.format(
				  	epoch,step,len(train_loader),batch_time=batch_time,data_time=data_time,loss=losses,acc=acc
				  	)
				)

def validate(validate_loader,model,loss_type,best_precision,lowest_loss):
	batch_time = AverageMeter()
	losses = AverageMeter()
	acc = AverageMeter()

	#switch to evalute model
	model.eval()

	end = time.time()
	for step,(feature,label) in enumerate(validate_loader):
		# feature,label = data
		
		feature = Variable(feature).cuda(async=True)
		label = Variable(label).cuda(async=True)

		with torch.no_grad():
			y_pred = model(feature)
			loss = loss_type(y_pred,label.squeeze())

		#measure accuracy and record loss
		pred_acc,PRED_COUNT = accuracy(y_pred.data,label,topk=(1,1))
		losses.update(loss.item(),feature.size(0))
		acc.update(pred_acc,PRED_COUNT)

		
		batch_time.update(time.time(),1)
		end = time.time()

		if step % 10 == 0:
			print('TrainVal: [{0}/{1}]\t'
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				'Accuray {acc.val:.3f} ({acc.avg:.3f})'.format(
					step, len(validate_loader), batch_time=batch_time, loss=losses, acc=acc))

	print(' * Accuray {acc.avg:.3f}'.format(acc=acc), '(Previous Best Acc: %.3f)' % best_precision,
		' * Loss {loss.avg:.3f}'.format(loss=losses), 'Previous Lowest Loss: %.3f)' % lowest_loss)
	return acc.avg,losses.avg

def test(test_loader,model,num_class,topk=1):
	"""
	test_loader: test data, type of DataLoader
	model: pretrained model
	filename: the file used to save inference result
	"""
	top1_prob = []
	top1_pred_label = []
	topk_prob = []
	topk_pred_label = []
	actual_label = []
	correct = 0

	predict_num = torch.zeros((1,num_class))
	acc_num = torch.zeros((1,num_class))
	target_num = torch.zeros((1,num_class))
	topk_predict_num = torch.zeros((1,num_class))
	topk_acc_num = torch.zeros((1,num_class))
	topk_target_num = torch.zeros((1,num_class))

	model.eval()
	for step,(feature,label) in enumerate(test_loader):
		feature = Variable(feature)
		label = Variable(label)
		
		with torch.no_grad():
			y_pred = model(feature)
			#使用softmax预测结果
			smax = nn.Softmax(1)
			smax_out = smax(y_pred)

		probility,pred_label = torch.topk(smax_out,topk)
		p1,l1 = torch.topk(smax_out,1)

		top1_mask = torch.zeros(y_pred.size()).scatter_(1,l1.cpu().view(-1,1),1)
		topk_mask = torch.zeros(y_pred.size())
		topk_label_index = pred_label.view(1,-1)
		topk_label_row = np.array([[x]*topk for x in range(feature.size(0))]).reshape(1,-1).tolist()

		topk_mask[topk_label_row,topk_label_index] = 1 
		actual_mask = torch.zeros(y_pred.size()).scatter_(1,label.cpu().view(-1,1),1)
		top1_acc_mask = top1_mask * actual_mask
		topk_acc_mask = topk_mask * actual_mask

		acc_num += top1_acc_mask.sum(0)
		predict_num += top1_mask.sum(0)
		target_num += actual_mask.sum(0)
		topk_acc_num += topk_acc_mask.sum(0)
		topk_predict_num += topk_mask.sum(0)
		topk_target_num += actual_mask.sum(0)

		actual_label += label.squeeze().tolist()
		topk_prob += probility.tolist()
		topk_pred_label += pred_label.tolist()
		top1_prob += p1.tolist()
		top1_pred_label += l1.tolist()


	top1_prob = np.array(top1_prob)
	top1_pred_label = np.array(top1_pred_label)
	topk_prob = np.array(topk_prob)
	topk_pred_label = np.array(topk_pred_label)
	actual_label = np.array(actual_label).reshape(-1,1)

	recall = acc_num / target_num
	precision = acc_num / predict_num
	F1 = 2*recall*precision/(recall + precision)
	accuracy = acc_num.sum(1) / target_num.sum(1)
	# accuracys = acc_num / target_num
	topk_recall = topk_acc_num / topk_target_num
	topk_precision = topk_acc_num / topk_predict_num
	topk_F1 = 2*topk_recall*topk_precision/(topk_recall + topk_precision)
	topk_accuracy = topk_acc_num.sum(1) / topk_target_num.sum(1)
	# topk_accuracys = topk_acc_num / topk_target_num

	recall = (recall.numpy()*100).round(4)
	precision = (precision.numpy()*100).round(4)
	F1 = (F1.numpy()*100).round(4)
	accuracy = (accuracy.numpy()*100).round(4)
	# accuracys = (accuracys.numpy()*100).round(4)
	topk_recall = (topk_recall.numpy()*100).round(4)
	topk_precision = (topk_precision.numpy()*100).round(4)
	topk_F1 = (topk_F1.numpy()*100).round(4)
	topk_accuracy = (topk_accuracy.numpy()*100).round(4)
	# topk_accuracys = (topk_accuracys.numpy()*100).round(4)

	result = (top1_prob,top1_pred_label,topk_prob,topk_pred_label,actual_label)
	top1_metrics = (accuracy,recall,precision,F1)
	topk_metrics = (topk_accuracy,topk_recall,topk_precision,topk_F1)
	
	return result,top1_metrics,topk_metrics



def accuracy(y_pred,y_label,topk=(1,)):
	""""
	y_pred: the net predected label
	y_label: the actual label
	topk: the top k accuracy
	return: accuracy and data length
	""" 
	final_acc = 0
	maxk = max(topk)

	PRED_COUNT = y_label.size(0)
	PRED_CORRECT_COUNT = 0
	prob,pred_label = y_pred.topk(maxk,dim=1,largest=True,sorted=True)
	for x in range(pred_label.size(0)):
		if int(pred_label[x]) == y_label[x]:
			PRED_CORRECT_COUNT += 1
	
	if PRED_COUNT == 0:
		return final_acc

	final_acc = PRED_CORRECT_COUNT / PRED_COUNT
	return final_acc*100,PRED_COUNT


def adjust_learning_rate(model,weight_decay,base_lr,lr_decay):
	base_lr = base_lr / lr_decay
	return optim.Adam(model.parameters(),base_lr,weight_decay=weight_decay,amsgrad=True)

def save_checkpoint(state,is_best,is_lowest_loss,filename):
	s_filename = '../model/%s/checkpoint.pth.tar' %filename
	torch.save(state,s_filename)
	if is_best:
		shutil.copyfile(s_filename,'../model/%s/model_best.pth.tar' %filename)
	if is_lowest_loss:
		shutil.copyfile(s_filename,'../model/%s/lowest_loss.pth.tar' %filename)


class DealDataSet(Dataset):
	"""docstring for DealDataSet"""
	def __init__(self,data_list,header_payload=False):
		self.x = torch.from_numpy(data_list[:,:-1])
		self.x = self.x.type(torch.FloatTensor)
		if header_payload == True:
			self.x = self.x.view(self.x.shape[0],1,22,22)
		else:
			self.x = self.x.view(self.x.shape[0],1,16,16)
		self.y = torch.from_numpy(data_list[:,[-1]])
		self.y = self.y.type(torch.LongTensor)
		self.len = self.x.shape[0]
		self.xshape = self.x.shape
		self.yshape = self.y.shape


	def __getitem__(self,index):
		return self.x[index],self.y[index]

	def __len__(self):
		return self.len




class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self,val,n=1):
		self.val = val
		self.sum += val*n
		self.count += n
		self.avg = self.sum / self.count

class TrainDataSetHeader():
	def __init__(self,data_type=1):
		self.data_type = data_type
		super(TrainDataSetHeader, self).__init__()

	def read_csv(self):
		if self.data_type == 1:		
			mydata_botnet =  pd.read_csv('../flow_labeled/labeld_Botnet.csv')#2075
			mydata_DDoS =  pd.read_csv('../flow_labeled/labeld_DDoS.csv')#261226
			mydata_glodeneye =  pd.read_csv('../flow_labeled/labeld_DoS-GlodenEye.csv')#20543
			mydata_hulk =  pd.read_csv('../flow_labeled/labeld_DoS-Hulk.csv')#474656
			mydata_slowhttp =  pd.read_csv('../flow_labeled/labeld_DoS-Slowhttptest.csv')#6786
			mydata_slowloris =  pd.read_csv('../flow_labeled/labeld_DoS-Slowloris.csv')#10537
			mydata_ftppatator =  pd.read_csv('../flow_labeled/labeld_FTP-Patator.csv')#19941
			mydata_heartbleed =  pd.read_csv('../flow_labeled/labeld_Heartbleed-Port.csv')#9859
			mydata_infiltration_2 =  pd.read_csv('../flow_labeled/labeld_Infiltration-2.csv')#5126
			mydata_infiltration_4 =  pd.read_csv('../flow_labeled/labeld_Infiltration-4.csv')#168
			mydata_portscan_1 =  pd.read_csv('../flow_labeled/labeld_PortScan_1.csv')#755
			mydata_portscan_2 =  pd.read_csv('../flow_labeled/labeld_PortScan_2.csv')#318881
			mydata_sshpatator =  pd.read_csv('../flow_labeled/labeld_SSH-Patator.csv')#27545
			mydata_bruteforce =  pd.read_csv('../flow_labeled/labeld_WebAttack-BruteForce.csv')#7716
			mydata_sqlinjection =  pd.read_csv('../flow_labeled/labeld_WebAttack-SqlInjection.csv')#25
			mydata_xss =  pd.read_csv('../flow_labeled/labeld_WebAttack-XSS.csv')#2796
		elif self.data_type == 2:
			mydata_botnet =  pd.read_csv('../payload_labeled/labeld_Botnet_payload.csv')
			mydata_DDoS =  pd.read_csv('../payload_labeled/labeld_DDoS_payload.csv')
			mydata_glodeneye =  pd.read_csv('../payload_labeled/labeld_DoS-GlodenEye_payload.csv')
			mydata_hulk =  pd.read_csv('../payload_labeled/labeld_DoS-Hulk_payload.csv')
			mydata_slowhttp =  pd.read_csv('../payload_labeled/labeld_DoS-Slowhttptest_payload.csv')
			mydata_slowloris =  pd.read_csv('../payload_labeled/labeld_DoS-Slowloris_payload.csv')
			mydata_ftppatator =  pd.read_csv('../payload_labeled/labeld_FTP-Patator_payload.csv')
			mydata_heartbleed =  pd.read_csv('../payload_labeled/labeld_Heartbleed-Port_payload.csv')
			mydata_infiltration_2 =  pd.read_csv('../payload_labeled/labeld_Infiltration-2_payload.csv')
			mydata_infiltration_4 =  pd.read_csv('../payload_labeled/labeld_Infiltration-4_payload.csv')
			mydata_portscan_1 =  pd.read_csv('../payload_labeled/labeld_PortScan_1_payload.csv')
			mydata_portscan_2 =  pd.read_csv('../payload_labeled/labeld_PortScan_2_payload.csv')
			mydata_sshpatator =  pd.read_csv('../payload_labeled/labeld_SSH-Patator_payload.csv')
			mydata_bruteforce =  pd.read_csv('../payload_labeled/labeld_WebAttack-BruteForce_payload.csv')
			mydata_sqlinjection =  pd.read_csv('../payload_labeled/labeld_WebAttack-SqlInjection_payload.csv')
			mydata_xss = pd.read_csv('../payload_labeled/labeld_WebAttack-XSS_payload.csv')
		elif self.data_type == 3:
			mydata_botnet =  pd.read_csv('../head_payload_labeled/labeld_Botnet_head_payload.csv')
			mydata_DDoS =  pd.read_csv('../head_payload_labeled/labeld_DDoS_head_payload.csv')
			mydata_glodeneye =  pd.read_csv('../head_payload_labeled/labeld_DoS-GlodenEye_head_payload.csv')
			mydata_hulk =  pd.read_csv('../head_payload_labeled/labeld_DoS-Hulk_head_payload.csv')
			mydata_slowhttp =  pd.read_csv('../head_payload_labeled/labeld_DoS-Slowhttptest_head_payload.csv')
			mydata_slowloris =  pd.read_csv('../head_payload_labeled/labeld_DoS-Slowloris_head_payload.csv')
			mydata_ftppatator =  pd.read_csv('../head_payload_labeled/labeld_FTP-Patator_head_payload.csv')
			mydata_heartbleed =  pd.read_csv('../head_payload_labeled/labeld_Heartbleed-Port_head_payload.csv')
			mydata_infiltration_2 =  pd.read_csv('../head_payload_labeled/labeld_Infiltration-2_head_payload.csv')
			mydata_infiltration_4 =  pd.read_csv('../head_payload_labeled/labeld_Infiltration-4_head_payload.csv')
			mydata_portscan_1 =  pd.read_csv('../head_payload_labeled/labeld_PortScan_1_head_payload.csv')
			mydata_portscan_2 =  pd.read_csv('../head_payload_labeled/labeld_PortScan_2_head_payload.csv')
			mydata_sshpatator =  pd.read_csv('../head_payload_labeled/labeld_SSH-Patator_head_payload.csv')
			mydata_bruteforce =  pd.read_csv('../head_payload_labeled/labeld_WebAttack-BruteForce_head_payload.csv')
			mydata_sqlinjection =  pd.read_csv('../head_payload_labeled/labeld_WebAttack-SqlInjection_head_payload.csv')
			mydata_xss = pd.read_csv('../head_payload_labeled/labeld_WebAttack-XSS_head_payload.csv')

		botnet = mydata_botnet.values[:,1:]
		ddos = mydata_DDoS.values[:,1:]
		glodeneye = mydata_glodeneye.values[:,1:]
		hulk = mydata_hulk.values[:,1:]
		slowhttp = mydata_slowhttp.values[:,1:]
		slowloris = mydata_slowloris.values[:,1:]
		ftp_patator = mydata_ftppatator.values[:,1:]
		heartbleed = mydata_heartbleed.values[:,1:]
		infiltration_2 = mydata_infiltration_2.values[:,1:]
		infiltration_4 = mydata_infiltration_4.values[:,1:]
		portscan_1 = mydata_portscan_1.values[:,1:]
		portscan_2 = mydata_portscan_2.values[:,1:]
		ssh_patator = mydata_sshpatator.values[:,1:]
		bruteforce = mydata_bruteforce.values[:,1:]
		sqlinjection = mydata_sqlinjection.values[:,1:]
		xss = mydata_xss.values[:,1:]
		
		return botnet,ddos,glodeneye,hulk,slowhttp,slowloris,ftp_patator,heartbleed,infiltration_2,infiltration_4,portscan_1,portscan_2,ssh_patator,bruteforce,sqlinjection,xss

	def get_item(self):
		botnet,ddos,glodeneye,hulk,slowhttp,slowloris,ftp_patator,heartbleed,infiltration_2,infiltration_4,portscan_1,portscan_2,ssh_patator,bruteforce,sqlinjection,xss = self.read_csv() 

		print('shape of botnet: ',botnet.shape)
		print('shape of DDoS: ',ddos.shape)
		print('shape of glodeneye: ',glodeneye.shape)
		print('shape of hulk: ',hulk.shape)
		print('shape of slowhttp: ',slowhttp.shape)
		print('shape of slowloris: ',slowloris.shape)
		print('shape of ftppatator: ',ftp_patator.shape)
		print('shape of heartbleed: ',heartbleed.shape)
		print('shape of infiltration_2: ',infiltration_2.shape)
		print('shape of infiltration_4: ',infiltration_4.shape)
		print('shape of portscan_1: ',portscan_1.shape)
		print('shape of portscan_2: ',portscan_2.shape)
		print('shape of sshpatator: ',ssh_patator.shape)
		print('shape of brutefoece: ',bruteforce.shape)
		print('shape of sqlinjection: ',sqlinjection.shape)
		print('shape of xss: ',xss.shape)

		x_botnet = botnet[:,:-1]
		x_ddos = ddos[:,:-1]
		x_glodeneye = glodeneye[:,:-1]
		x_hulk = hulk[:,:-1]
		x_slowhttp = slowhttp[:,:-1]
		x_slowloris = slowloris[:,:-1]
		x_ftppatator = ftp_patator[:,:-1]
		x_heartbleed = heartbleed[:,:-1]
		x_infiltration_2 = infiltration_2[:,:-1]
		x_infiltration_4 = infiltration_4[:,:-1]
		x_portscan_1 = portscan_1[:,:-1]
		x_portscan_2 = portscan_2[:,:-1]
		x_sshpatator = ssh_patator[:,:-1]
		x_bruteforce = bruteforce[:,:-1]
		x_sqlinjection = sqlinjection[:,:-1]
		x_xss = xss[:,:-1]

		y_botnet = botnet[:,-1]
		y_ddos = ddos[:,-1]
		y_glodeneye = glodeneye[:,-1]
		y_hulk = hulk[:,-1]
		y_slowhttp = slowhttp[:,-1]
		y_slowloris = slowloris[:,-1]
		y_ftppatator = ftp_patator[:,-1]
		y_heartbleed = heartbleed[:,-1]
		y_infiltration_2 = infiltration_2[:,-1]
		y_infiltration_4 = infiltration_4[:,-1]
		y_portscan_1 = portscan_1[:,-1]
		y_portscan_2 = portscan_2[:,-1]
		y_sshpatator = ssh_patator[:,-1]
		y_bruteforce = bruteforce[:,-1]
		y_sqlinjection = sqlinjection[:,-1]
		y_xss = xss[:,-1]

		x_tr_botnet,x_te_botnet,y_tr_botnet,y_te_botnet = train_test_split(x_botnet,y_botnet,test_size=0.2,random_state=1)
		x_tr_ddos,x_te_ddos,y_tr_ddos,y_te_ddos = train_test_split(x_ddos,y_ddos,test_size=0.2,random_state=1)
		x_tr_glodeneye,x_te_glodeneye,y_tr_glodeneye,y_te_glodeneye = train_test_split(x_glodeneye,y_glodeneye,test_size=0.2,random_state=1)
		x_tr_hulk,x_te_hulk,y_tr_hulk,y_te_hulk = train_test_split(x_hulk,y_hulk,test_size=0.2,random_state=1)
		x_tr_slowhttp,x_te_slowhttp,y_tr_slowhttp,y_te_slowhttp = train_test_split(x_slowhttp,y_slowhttp,test_size=0.2,random_state=1)
		x_tr_slowloris,x_te_slowloris,y_tr_slowloris,y_te_slowloris = train_test_split(x_slowloris,y_slowloris,test_size=0.2,random_state=1)
		x_tr_ftppatator,x_te_ftppatator,y_tr_ftppatator,y_te_ftppatator = train_test_split(x_ftppatator,y_ftppatator,test_size=0.2,random_state=1)
		x_tr_heartbleed,x_te_heartbleed,y_tr_heartbleed,y_te_heartbleed = train_test_split(x_heartbleed,y_heartbleed,test_size=0.2,random_state=1)
		x_tr_infiltration_2,x_te_infiltration_2,y_tr_infiltration_2,y_te_infiltration_2 = train_test_split(x_infiltration_2,y_infiltration_2,test_size=0.2,random_state=1)
		x_tr_infiltration_4,x_te_infiltration_4,y_tr_infiltration_4,y_te_infiltration_4 = train_test_split(x_infiltration_4,y_infiltration_4,test_size=0.2,random_state=1)
		x_tr_portscan_1,x_te_portscan_1,y_tr_portscan_1,y_te_portscan_1 = train_test_split(x_portscan_1,y_portscan_1,test_size=0.2,random_state=1)
		x_tr_portscan_2,x_te_portscan_2,y_tr_portscan_2,y_te_portscan_2 = train_test_split(x_portscan_2,y_portscan_2,test_size=0.2,random_state=1)
		x_tr_sshpatator,x_te_sshpatator,y_tr_sshpatator,y_te_sshpatator = train_test_split(x_sshpatator,y_sshpatator,test_size=0.2,random_state=1)
		x_tr_bruteforce,x_te_bruteforce,y_tr_bruteforce,y_te_bruteforce = train_test_split(x_bruteforce,y_bruteforce,test_size=0.2,random_state=1)
		x_tr_sqlinjection,x_te_sqlinjection,y_tr_sqlinjection,y_te_sqlinjection = train_test_split(x_sqlinjection,y_sqlinjection,test_size=0.2,random_state=1)
		x_tr_xss,x_te_xss,y_tr_xss,y_te_xss = train_test_split(x_xss,y_xss,test_size=0.2,random_state=1)


		x_tr_infiltration = np.concatenate((x_tr_infiltration_2,x_tr_infiltration_4),axis=0)
		x_tr_portscan = np.concatenate((x_tr_portscan_1,x_tr_portscan_2),axis=0)
		x_tr_webattack = np.concatenate((x_tr_bruteforce,x_tr_sqlinjection,x_tr_xss),axis=0)

		y_tr_infiltration = np.concatenate((y_tr_infiltration_2,y_tr_infiltration_4))
		y_tr_portscan = np.concatenate((y_tr_portscan_1,y_tr_portscan_2))
		y_tr_webattack = np.concatenate((y_tr_bruteforce,y_tr_sqlinjection,y_tr_xss))

		x_te_infiltration = np.concatenate((x_te_infiltration_2,x_te_infiltration_4),axis=0)
		x_te_portscan = np.concatenate((x_te_portscan_1,x_te_portscan_2),axis=0)
		x_te_webattack = np.concatenate((x_te_bruteforce,x_te_sqlinjection,x_te_xss),axis=0)

		y_te_infiltration = np.concatenate((y_te_infiltration_2,y_te_infiltration_4))
		y_te_portscan = np.concatenate((y_te_portscan_1,y_te_portscan_2))
		y_te_webattack = np.concatenate((y_te_bruteforce,y_te_sqlinjection,y_te_xss))

		#play label
		y_tr_botnet = np.array([0]*len(y_tr_botnet))
		y_tr_ddos = np.array([1]*len(y_tr_ddos))
		y_tr_glodeneye = np.array([2]*len(y_tr_glodeneye))
		y_tr_hulk = np.array([3]*len(y_tr_hulk))
		y_tr_slowhttp = np.array([4]*len(y_tr_slowhttp))
		y_tr_slowloris = np.array([5]*len(y_tr_slowloris))
		y_tr_ftppatator = np.array([6]*len(y_tr_ftppatator))
		y_tr_heartbleed = np.array([7]*len(y_tr_heartbleed))
		y_tr_infiltration = np.array([8]*len(y_tr_infiltration))
		y_tr_portscan = np.array([9]*len(y_tr_portscan))
		y_tr_sshpatator = np.array([10]*len(y_tr_sshpatator))
		y_tr_webattack = np.array([11]*len(y_tr_webattack))

		y_te_botnet = np.array([0]*len(y_te_botnet))
		y_te_ddos = np.array([1]*len(y_te_ddos))
		y_te_glodeneye = np.array([2]*len(y_te_glodeneye))
		y_te_hulk = np.array([3]*len(y_te_hulk))
		y_te_slowhttp = np.array([4]*len(y_te_slowhttp))
		y_te_slowloris = np.array([5]*len(y_te_slowloris))
		y_te_ftppatator = np.array([6]*len(y_te_ftppatator))
		y_te_heartbleed = np.array([7]*len(y_te_heartbleed))
		y_te_infiltration = np.array([8]*len(y_te_infiltration))
		y_te_portscan = np.array([9]*len(y_te_portscan))
		y_te_sshpatator = np.array([10]*len(y_te_sshpatator))
		y_te_webattack = np.array([11]*len(y_te_webattack))

		x_train = np.concatenate((x_tr_botnet,x_tr_ddos,x_tr_glodeneye,x_tr_hulk,x_tr_slowhttp,x_tr_slowloris,x_tr_ftppatator,x_tr_heartbleed,x_tr_infiltration,x_tr_portscan,x_tr_sshpatator,x_tr_webattack))
		y_train = np.concatenate((y_tr_botnet,y_tr_ddos,y_tr_glodeneye,y_tr_hulk,y_tr_slowhttp,y_tr_slowloris,y_tr_ftppatator,y_tr_heartbleed,y_tr_infiltration,y_tr_portscan,y_tr_sshpatator,y_tr_webattack))
		
		x_test = np.concatenate((x_te_botnet,x_te_ddos,x_te_glodeneye,x_te_hulk,x_te_slowhttp,x_te_slowloris,x_te_ftppatator,x_te_heartbleed,x_te_infiltration,x_te_portscan,x_te_sshpatator,x_te_webattack))
		y_test = np.concatenate((y_te_botnet,y_te_ddos,y_te_glodeneye,y_te_hulk,y_te_slowhttp,y_te_slowloris,y_te_ftppatator,y_te_heartbleed,y_te_infiltration,y_te_portscan,y_te_sshpatator,y_te_webattack))

		return x_train,y_train,x_test,y_test


class TrainDataSetPayload():
	"""docstring for TrainDataSetPayload"""
	def __init__(self,):
		super(TrainDataSetPayload, self).__init__()
	
	def read_csv(self):
		payload_botnet =  pd.read_csv('../payload_labeled/labeld_Botnet_payload.csv')
		payload_DDoS =  pd.read_csv('../payload_labeled/labeld_DDoS_payload.csv')
		payload_glodeneye =  pd.read_csv('../payload_labeled/labeld_DoS-GlodenEye_payload.csv')
		payload_hulk =  pd.read_csv('../payload_labeled/labeld_DoS-Hulk_payload.csv')
		payload_slowhttp =  pd.read_csv('../payload_labeled/labeld_DoS-Slowhttptest_payload.csv')
		payload_slowloris =  pd.read_csv('../payload_labeled/labeld_DoS-Slowloris_payload.csv')
		payload_ftppatator =  pd.read_csv('../payload_labeled/labeld_FTP-Patator_payload.csv')
		payload_heartbleed =  pd.read_csv('../payload_labeled/labeld_Heartbleed-Port_payload.csv')
		payload_infiltration_2 =  pd.read_csv('../payload_labeled/labeld_Infiltration-2_payload.csv')
		payload_infiltration_4 =  pd.read_csv('../payload_labeled/labeld_Infiltration-4_payload.csv')
		payload_portscan_1 =  pd.read_csv('../payload_labeled/labeld_PortScan_1_payload.csv')
		payload_portscan_2 =  pd.read_csv('../payload_labeled/labeld_PortScan_2_payload.csv')
		payload_sshpatator =  pd.read_csv('../payload_labeled/labeld_SSH-Patator_payload.csv')
		payload_brutefoece =  pd.read_csv('../payload_labeled/labeld_WebAttack-BruteForce_payload.csv')
		payload_sqlinjection =  pd.read_csv('../payload_labeled/labeld_WebAttack-SqlInjection_payload.csv')
		payload_xss = pd.read_csv('../payload_labeled/labeld_WebAttack-XSS_payload.csv')

		# print('finish reading dataset, cost time :',time.time() - start)

		botnet = payload_botnet.values[:,1:]
		ddos = payload_DDoS.values[:,1:]
		glodeneye = payload_glodeneye.values[:,1:]
		hulk = payload_hulk.values[:,1:]
		slowhttp = payload_slowhttp.values[:,1:]
		slowloris = payload_slowloris.values[:,1:]
		ftp_patator = payload_ftppatator.values[:,1:]
		heartbleed = payload_heartbleed.values[:,1:]
		infiltration_2 = payload_infiltration_2.values[:,1:]
		infiltration_4 = payload_infiltration_4.values[:,1:]
		portscan_1 = payload_portscan_1.values[:,1:]
		portscan_2 = payload_portscan_2.values[:,1:]
		ssh_patator = payload_sshpatator.values[:,1:]
		bruteforce = payload_brutefoece.values[:,1:]
		sqlinjection = payload_sqlinjection.values[:,1:]
		xss = payload_xss.values[:,1:]

		return botnet,ddos,glodeneye,hulk,slowhttp,slowloris,ftp_patator,heartbleed,infiltration_2,infiltration_4,portscan_1,portscan_2,ssh_patator,bruteforce,sqlinjection,xss

	def get_item(self):
		botnet,ddos,glodeneye,hulk,slowhttp,slowloris,ftp_patator,heartbleed,infiltration_2,infiltration_4,portscan_1,portscan_2,ssh_patator,bruteforce,sqlinjection,xss = self.read_csv() 

		print('shape of botnet: ',botnet.shape)
		print('shape of DDoS: ',ddos.shape)
		print('shape of glodeneye: ',glodeneye.shape)
		print('shape of hulk: ',hulk.shape)
		print('shape of slowhttp: ',slowhttp.shape)
		print('shape of slowloris: ',slowloris.shape)
		print('shape of ftppatator: ',ftp_patator.shape)
		print('shape of heartbleed: ',heartbleed.shape)
		print('shape of infiltration_2: ',infiltration_2.shape)
		print('shape of infiltration_4: ',infiltration_4.shape)
		print('shape of portscan_1: ',portscan_1.shape)
		print('shape of portscan_2: ',portscan_2.shape)
		print('shape of sshpatator: ',ssh_patator.shape)
		print('shape of brutefoece: ',bruteforce.shape)
		print('shape of sqlinjection: ',sqlinjection.shape)
		print('shape of xss: ',xss.shape)

		x_botnet = botnet[:,:-1]
		x_ddos = ddos[:,:-1]
		x_glodeneye = glodeneye[:,:-1]
		x_hulk = hulk[:,:-1]
		x_slowhttp = slowhttp[:,:-1]
		x_slowloris = slowloris[:,:-1]
		x_ftppatator = ftp_patator[:,:-1]
		x_heartbleed = heartbleed[:,:-1]
		x_infiltration_2 = infiltration_2[:,:-1]
		x_infiltration_4 = infiltration_4[:,:-1]
		x_portscan_1 = portscan_1[:,:-1]
		x_portscan_2 = portscan_2[:,:-1]
		x_sshpatator = ssh_patator[:,:-1]
		x_bruteforce = bruteforce[:,:-1]
		x_sqlinjection = sqlinjection[:,:-1]
		x_xss = xss[:,:-1]

		y_botnet = botnet[:,-1]
		y_ddos = ddos[:,-1]
		y_glodeneye = glodeneye[:,-1]
		y_hulk = hulk[:,-1]
		y_slowhttp = slowhttp[:,-1]
		y_slowloris = slowloris[:,-1]
		y_ftppatator = ftp_patator[:,-1]
		y_heartbleed = heartbleed[:,-1]
		y_infiltration_2 = infiltration_2[:,-1]
		y_infiltration_4 = infiltration_4[:,-1]
		y_portscan_1 = portscan_1[:,-1]
		y_portscan_2 = portscan_2[:,-1]
		y_sshpatator = ssh_patator[:,-1]
		y_bruteforce = bruteforce[:,-1]
		y_sqlinjection = sqlinjection[:,-1]
		y_xss = xss[:,-1]

		x_tr_botnet,x_te_botnet,y_tr_botnet,y_te_botnet = train_test_split(x_botnet,y_botnet,test_size=0.2,random_state=1)
		x_tr_ddos,x_te_ddos,y_tr_ddos,y_te_ddos = train_test_split(x_ddos,y_ddos,test_size=0.2,random_state=1)
		x_tr_glodeneye,x_te_glodeneye,y_tr_glodeneye,y_te_glodeneye = train_test_split(x_glodeneye,y_glodeneye,test_size=0.2,random_state=1)
		x_tr_hulk,x_te_hulk,y_tr_hulk,y_te_hulk = train_test_split(x_hulk,y_hulk,test_size=0.2,random_state=1)
		x_tr_slowhttp,x_te_slowhttp,y_tr_slowhttp,y_te_slowhttp = train_test_split(x_slowhttp,y_slowhttp,test_size=0.2,random_state=1)
		x_tr_slowloris,x_te_slowloris,y_tr_slowloris,y_te_slowloris = train_test_split(x_slowloris,y_slowloris,test_size=0.2,random_state=1)
		x_tr_ftppatator,x_te_ftppatator,y_tr_ftppatator,y_te_ftppatator = train_test_split(x_ftppatator,y_ftppatator,test_size=0.2,random_state=1)
		x_tr_heartbleed,x_te_heartbleed,y_tr_heartbleed,y_te_heartbleed = train_test_split(x_heartbleed,y_heartbleed,test_size=0.2,random_state=1)
		x_tr_infiltration_2,x_te_infiltration_2,y_tr_infiltration_2,y_te_infiltration_2 = train_test_split(x_infiltration_2,y_infiltration_2,test_size=0.2,random_state=1)
		x_tr_infiltration_4,x_te_infiltration_4,y_tr_infiltration_4,y_te_infiltration_4 = train_test_split(x_infiltration_4,y_infiltration_4,test_size=0.2,random_state=1)
		x_tr_portscan_1,x_te_portscan_1,y_tr_portscan_1,y_te_portscan_1 = train_test_split(x_portscan_1,y_portscan_1,test_size=0.2,random_state=1)
		x_tr_portscan_2,x_te_portscan_2,y_tr_portscan_2,y_te_portscan_2 = train_test_split(x_portscan_2,y_portscan_2,test_size=0.2,random_state=1)
		x_tr_sshpatator,x_te_sshpatator,y_tr_sshpatator,y_te_sshpatator = train_test_split(x_sshpatator,y_sshpatator,test_size=0.2,random_state=1)
		x_tr_bruteforce,x_te_bruteforce,y_tr_bruteforce,y_te_bruteforce = train_test_split(x_bruteforce,y_bruteforce,test_size=0.2,random_state=1)
		x_tr_sqlinjection,x_te_sqlinjection,y_tr_sqlinjection,y_te_sqlinjection = train_test_split(x_sqlinjection,y_sqlinjection,test_size=0.2,random_state=1)
		x_tr_xss,x_te_xss,y_tr_xss,y_te_xss = train_test_split(x_xss,y_xss,test_size=0.2,random_state=1)


		x_tr_infiltration = np.concatenate((x_tr_infiltration_2,x_tr_infiltration_4),axis=0)
		x_tr_portscan = np.concatenate((x_tr_portscan_1,x_tr_portscan_2),axis=0)
		x_tr_webattack = np.concatenate((x_tr_bruteforce,x_tr_sqlinjection,x_tr_xss),axis=0)

		y_tr_infiltration = np.concatenate((y_tr_infiltration_2,y_tr_infiltration_4))
		y_tr_portscan = np.concatenate((y_tr_portscan_1,y_tr_portscan_2))
		y_tr_webattack = np.concatenate((y_tr_bruteforce,y_tr_sqlinjection,y_tr_xss))

		x_te_infiltration = np.concatenate((x_te_infiltration_2,x_te_infiltration_4),axis=0)
		x_te_portscan = np.concatenate((x_te_portscan_1,x_te_portscan_2),axis=0)
		x_te_webattack = np.concatenate((x_te_bruteforce,x_te_sqlinjection,x_te_xss),axis=0)

		y_te_infiltration = np.concatenate((y_te_infiltration_2,y_te_infiltration_4))
		y_te_portscan = np.concatenate((y_te_portscan_1,y_te_portscan_2))
		y_te_webattack = np.concatenate((y_te_bruteforce,y_te_sqlinjection,y_te_xss))

		#play label
		y_tr_botnet = np.array([0]*len(y_tr_botnet))
		y_tr_ddos = np.array([1]*len(y_tr_ddos))
		y_tr_glodeneye = np.array([2]*len(y_tr_glodeneye))
		y_tr_hulk = np.array([3]*len(y_tr_hulk))
		y_tr_slowhttp = np.array([4]*len(y_tr_slowhttp))
		y_tr_slowloris = np.array([5]*len(y_tr_slowloris))
		y_tr_ftppatator = np.array([6]*len(y_tr_ftppatator))
		y_tr_heartbleed = np.array([7]*len(y_tr_heartbleed))
		y_tr_infiltration = np.array([8]*len(y_tr_infiltration))
		y_tr_portscan = np.array([9]*len(y_tr_portscan))
		y_tr_sshpatator = np.array([10]*len(y_tr_sshpatator))
		y_tr_webattack = np.array([11]*len(y_tr_webattack))

		y_te_botnet = np.array([0]*len(y_te_botnet))
		y_te_ddos = np.array([1]*len(y_te_ddos))
		y_te_glodeneye = np.array([2]*len(y_te_glodeneye))
		y_te_hulk = np.array([3]*len(y_te_hulk))
		y_te_slowhttp = np.array([4]*len(y_te_slowhttp))
		y_te_slowloris = np.array([5]*len(y_te_slowloris))
		y_te_ftppatator = np.array([6]*len(y_te_ftppatator))
		y_te_heartbleed = np.array([7]*len(y_te_heartbleed))
		y_te_infiltration = np.array([8]*len(y_te_infiltration))
		y_te_portscan = np.array([9]*len(y_te_portscan))
		y_te_sshpatator = np.array([10]*len(y_te_sshpatator))
		y_te_webattack = np.array([11]*len(y_te_webattack))

		x_train = np.concatenate((x_tr_botnet,x_tr_ddos,x_tr_glodeneye,x_tr_hulk,x_tr_slowhttp,x_tr_slowloris,x_tr_ftppatator,x_tr_heartbleed,x_tr_infiltration,x_tr_portscan,x_tr_sshpatator,x_tr_webattack))
		y_train = np.concatenate((y_tr_botnet,y_tr_ddos,y_tr_glodeneye,y_tr_hulk,y_tr_slowhttp,y_tr_slowloris,y_tr_ftppatator,y_tr_heartbleed,y_tr_infiltration,y_tr_portscan,y_tr_sshpatator,y_tr_webattack))
		
		x_test = np.concatenate((x_te_botnet,x_te_ddos,x_te_glodeneye,x_te_hulk,x_te_slowhttp,x_te_slowloris,x_te_ftppatator,x_te_heartbleed,x_te_infiltration,x_te_portscan,x_te_sshpatator,x_te_webattack))
		y_test = np.concatenate((y_te_botnet,y_te_ddos,y_te_glodeneye,y_te_hulk,y_te_slowhttp,y_te_slowloris,y_te_ftppatator,y_te_heartbleed,y_te_infiltration,y_te_portscan,y_te_sshpatator,y_te_webattack))

		return x_train,y_train,x_test,y_test