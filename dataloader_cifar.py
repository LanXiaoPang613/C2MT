from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter

            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[], log='', clean_idx=[], test_form = None):
        
        self.r = r # noise ratio
        self.transform = transform
        self.test_form = test_form
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        self.noise_file = noise_file
     
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            self.clean_label = np.array(train_label)

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
            else:    #inject noise   
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r*50000)            
                noise_idx = idx[:num_noise]
                for i in range(50000):
                    if i in noise_idx:
                        if noise_mode=='sym':
                            if dataset=='cifar10': 
                                noiselabel = random.randint(0,9)
                            elif dataset=='cifar100':    
                                noiselabel = random.randint(0,99)
                            noise_label.append(noiselabel)
                        elif noise_mode=='asym':   
                            noiselabel = self.transition[train_label[i]]
                            noise_label.append(noiselabel)                    
                    else:    
                        noise_label.append(train_label[i])   
                print("save noisy labels to %s ..."%noise_file)        
                json.dump(noise_label,open(noise_file,"w"))       
            
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = np.array(noise_label).astype(np.int64)
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(noise_label)==np.array(train_label))                                                       
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()
                    clean_index = np.where(np.array(noise_label)[pred_idx.tolist()] == np.array(self.clean_label)[pred_idx.tolist()])[0]

                    num_per_class = []
                    for i in range(max(noise_label)):
                        temp = np.where(np.array(noise_label)[clean_index.tolist()] == i)[0]
                        num_per_class.append(len(temp))
                    num_per_class2 = []
                    for i in range(max(noise_label)):
                        temp = np.where(np.array(noise_label)[pred_idx.tolist()] == i)[0]
                        num_per_class2.append(len(temp))
                    print('clean num per class:', num_per_class, num_per_class2)

                    log.write('Numer of labeled samples:%d   AUC:%.3f   corrected clean num:%d, uncorrected noisy num:%d\n'
                              % (pred.sum(), auc, len(clean_index), len(pred_idx) - len(clean_index)))
                    log.flush()
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]
                    noise_index = np.where(np.array(noise_label)[pred_idx.tolist()] != np.array(self.clean_label)[pred_idx.tolist()])[0]
                    log.write('Numer of unlabeled samples:%d   corrected noisy num:%d, uncorrected clean num:%d\n'
                              % (pred.sum(), len(noise_index), len(pred_idx) - len(noise_index)))
                    log.flush()
                elif self.mode == 'boost':
                    pred_idx = clean_idx
                
                self.train_data = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))

    def if_noise(self, pred=None):
        if pred is None:
            noise_index = np.where(self.noise_label[:] != self.clean_label[:])[0]
            clean_index = np.where(self.noise_label[:] == self.clean_label[:])[0]
            return noise_index, clean_index
        else:
            pred_idx1 = pred.nonzero()[0].tolist()
            clean_index = np.where(np.array(self.noise_label)[pred_idx1] == np.array(self.clean_label)[pred_idx1])[0]
            pred_idx = (1 - pred).nonzero()[0].tolist()
            noise_index = np.where(np.array(self.noise_label)[pred_idx] != np.array(self.clean_label)[pred_idx])[0]
            print(
                f'选择的非mask样本中正确选取的干净标签数量{len(clean_index)}, 不正确选取的非干净数量{len(pred_idx1) - len(clean_index)}.\t '
                f'选择的mask样本中正确选取的不干净标签数量{len(noise_index)}, 不正确选取的干净数量{len(pred_idx) - len(noise_index)}')
            return len(clean_index), (len(pred_idx1) - len(clean_index)), len(noise_index), len(pred_idx) - len(
                noise_index)
    def print_noise_rate(self, new_y):
        temp_y = np.array(new_y.reshape(1, -1).squeeze())
        clean_index = np.where(temp_y[:] == np.array(self.clean_label)[:])
        print(f'clean rate is: {len(clean_index[0]) / len(self.clean_label)}')

    def load_train_label(self, new_y):
        temp_y = np.array(new_y.reshape(1, -1).squeeze()).astype(np.int64)
        self.noise_label[:] = np.array(temp_y)[:]
        if os.path.exists(self.noise_file):
            result = os.path.splitext(self.noise_file)
            noise_file_temp = result[0]+'_old'+result[1]
            if not os.path.exists(noise_file_temp):
                os.rename(self.noise_file, noise_file_temp)
        #   覆盖原来的noise_file
        json.dump(self.noise_label.tolist(), open(self.noise_file, "w"))

    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
        elif self.mode=='boost':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img_no_da = self.test_form(img)
            img = self.transform(img)
            return img, img_no_da, target, index
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        if self.dataset=='cifar10':
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])   
    def run(self,mode,pred=[],prob=[], clean_idx=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=pred, probability=prob,log=self.log)              
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=pred, log=self.log)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)     
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader
        elif mode=='boost':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode=mode, noise_file=self.noise_file, clean_idx=clean_idx, test_form=self.transform_test)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader