import os
import cv2
import torch
import numpy
import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import torch.utils.data as data
import argparse

# 使用与李珊等人相同的数据

def get_train_loader(args):
    train_domain = args.train_data.split(',')
    train_data = [ ]

    if "raf2" in train_domain:
        dataset0 = RAFTrainSet(args,  args.train_list0)
        dataloader0 = DataLoader(
            dataset0,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.workers,
            pin_memory=False
        )
        train_data.append(dataloader0)
    return train_data

def get_test_loader(args): # testing data used in the training stage(different ck+ and jaffe)
    test_domain = args.test_data.split(',')
    test_data = [ ]

    if 'raf' in test_domain:

        dataset1 = RAFTestSet(args,args.test_list1)
        data_loader1 = DataLoader(
            dataset1,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(data_loader1)
    if 'aff' in test_domain:

        dataset2 = RAFTestSet(args, args.test_list2)
        data_loader2 = DataLoader(
            dataset2,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(data_loader2)
    if 'fer' in test_domain:

        dataset3 = RAFTestSet(args, args.test_list3)
        data_loader3 = DataLoader(
            dataset3,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(data_loader3)
    if 'ck+' in test_domain:
        dataset4 = RAFTestSet(args,  args.test_list9)
        data_loader4 = DataLoader(
            dataset4,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False
        )
        test_data.append(data_loader4)
    if 'mmi' in test_domain:
        dataset5 = RAFTestSet(args,  args.test_list5)
        data_loader5 = DataLoader(
            dataset5,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(data_loader5)
    if 'jaf' in test_domain:
        dataset6 = RAFTestSet(args,  args.test_list10)
        data_loader6 = DataLoader(
            dataset6,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(data_loader6)
    if 'oul' in test_domain:
        dataset7 = RAFTestSet(args,  args.test_list7)
        data_loader7 = DataLoader(
            dataset7,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(data_loader7)
    if 'sfew' in test_domain:
        dataset8 = RAFTestSet(args,  args.test_list8)
        data_loader8 = DataLoader(
            dataset8,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(data_loader8)
    return test_data


def get_test_loader_pure_test(args):
    test_domain = args.test_data.split(',')
    test_data = [ ]

    if 'raf' in test_domain:

        dataset1 = RAFTestSet(args,args.test_list1)
        data_loader1 = DataLoader(
            dataset1,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(data_loader1)
    if 'aff' in test_domain:

        dataset2 = RAFTestSet(args, args.test_list2)
        data_loader2 = DataLoader(
            dataset2,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(data_loader2)
    if 'fer' in test_domain:

        dataset3 = RAFTestSet(args, args.test_list3)
        data_loader3 = DataLoader(
            dataset3,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(data_loader3)
    if 'ck+' in test_domain:
        dataset4 = RAFTestSet(args,  args.test_list4)
        data_loader4 = DataLoader(
            dataset4,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False
        )
        test_data.append(data_loader4)
    if 'mmi' in test_domain:
        dataset5 = RAFTestSet(args,  args.test_list5)
        data_loader5 = DataLoader(
            dataset5,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(data_loader5)
    if 'jaf' in test_domain:
        dataset6 = RAFTestSet(args,  args.test_list6)
        data_loader6 = DataLoader(
            dataset6,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(data_loader6)
    if 'oul' in test_domain:
        dataset7 = RAFTestSet(args,  args.test_list7)
        data_loader7 = DataLoader(
            dataset7,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(data_loader7)
    if 'sfew' in test_domain:
        dataset8 = RAFTestSet(args,  args.test_list8)
        data_loader8 = DataLoader(
            dataset8,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(data_loader8)

    if 'amazon' in test_domain:
        dataset3 = RAFTestSet(args,  args.train_list4)
        dataloader3 = DataLoader(
            dataset3,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(dataloader3)
    if 'dslr' in test_domain:
        dataset3 = RAFTestSet(args,  args.train_list5)
        dataloader3 = DataLoader(
            dataset3,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(dataloader3)
    if 'webcam' in test_domain:
        dataset3 = RAFTestSet(args,  args.train_list6)
        dataloader3 = DataLoader(
            dataset3,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.workers,
            pin_memory=True
        )
        test_data.append(dataloader3)

    return test_data

class RAFTrainSet(data.Dataset):
    def __init__(self,args,data_list):
        self.images = list()
        self.targets = list()
        self.args = args

        lines = open(data_list).readlines()
        for line in lines:
            path, label = line.strip().split(' ')
            self.images.append(path)
            self.targets.append(int(label))
        if self.args.aug == True:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((256,256)),
                transforms.RandomCrop((224,224)),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image,(self.args.size,self.args.size))
        image = self.transform(image)
        target = self.targets[index]
        return image,target


    def __len__(self):
        return len(self.targets)

class RAFTestSet(data.Dataset):
    def __init__(self,args,data_list):
        self.images = list()
        self.targets = list()
        self.args = args

        lines = open(data_list).readlines()
        for line in lines:
            path, label = line.strip().split(' ')
            self.images.append(path)
            self.targets.append(int(label))
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image,(self.args.size,self.args.size))
        image = self.transform(image)
        target = self.targets[index]
        return image,target

    def __len__(self):
        return len(self.targets)
