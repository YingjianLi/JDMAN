import torch.nn as nn
import torch.backends.cudnn as cudnn

from opts import args

from model.resnet import resnet50_with_adlayer,AdversarialNetwork
from model.vgg import vgg16_bn
from model.vgg2 import VGG_local_global
from datasets import get_train_loader, get_test_loader,get_test_loader_pure_test
from log import Logger
from train import Trainer
import os
import torch
import datetime
import csv

print("run date: " + str(datetime.datetime.now()))

os.environ["CUDA_VISIBLE_DEVICES"] = args.nGPU
def get_catalogue():
    model_creators = dict()
    model_creators['resnet50_with_adlayer_all'] = resnet50_with_adlayer
    model_creators['vgg16'] = vgg16_bn
    model_creators['vgg16_local_global'] = VGG_local_global
    return model_creators

def create_model(args):
    state = None

    model_creators = get_catalogue() # 获取可用模型目录

    assert args.model in model_creators # 确认参数中的模型在可用目录里

    model = model_creators[args.model](args) # model结构

    if args.MI == True:
        # optimize mutual information by adversarial learning according to
        # Self-supervised representation learning from multi-domain data
        adv_model = AdversarialNetwork(3072, 512, args.n_epochs * 112)

    if args.resume: # 模型恢复
        save_path = os.path.join(args.save_path)  # 模型保存目录
        checkpoint = torch.load(save_path)

        model.load_state_dict(checkpoint['model']) # 模型参数
        state = checkpoint['state'] # state参数

    cudnn.benchmark = True
    GPUs = args.nGPU.split(",")
    if len(GPUs) > 1:
        # 并行
        model = nn.DataParallel(model, device_ids=[i for i in range(len(GPUs))]).cuda()
        if args.MI == True:
            adv_model = nn.DataParallel(adv_model, device_ids=[i for i in range(len(GPUs))]).cuda()
            return model, state, adv_model
        else:
            return model, state
    else:
        model = model.cuda()
        if args.MI == True:
            adv_model = adv_model.cuda()
            return model, state, adv_model
        else:
            return model, state

def main():
    # Create Model, Criterion and State
    if args.MI:
        model, state, adv_model = create_model(args)
    else:
        model, state = create_model(args)

    print("=> Model and criterion are ready")

    # Create Dataloader
    if not args.test_only:
        train_loader = get_train_loader(args)  # 获取训练数据, [dataloader1,dataloader2,dataloader3]
    val_loader = get_test_loader(args)  # 获取测试数据 [data_loader1,data_loader2,data_loader3,data_loader4]
    val_loader_pure_test = get_test_loader_pure_test(args)
    print("=> Dataloaders are ready")

    # Create Logger
    logger = Logger(args, state)  #创建模型保存目录，记录state
    print("=> Logger is ready")

    # Create Trainer
    if args.MI:
        trainer = Trainer(args,model,adv_model)
    else:
        trainer = Trainer(args, model, None) # 训练模型
    print("=> Trainer is ready")
    print('=> super parameters: ' + str(vars(args)))

    if args.test_only: # 仅测试，前提是已经有训练好的模型了
       test_summary = trainer.test(0, val_loader)
       print("- Test:  Acc %6.3f " % (test_summary['acc']))
    else: # 训练模式
        print(args.print)
        start_epoch = logger.state['epoch'] + 1 # 开始于上一次训练的下一个epoch
        print("=> Start training")

        domains = ['epoch','raf', 'aff', 'fer', 'ck+', 'mmi','jaf','oul','sfew']
        log_file = args.log_path+'.csv'
        with open(log_file, "a+", encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not args.target==None :
                # 先写入columns_name
                writer.writerow(['epoch',args.target])
                csvfile.close()
                best_epoches = [0, 0]
                best_accs = [0.0, 0.0]
            else:
                # 先写入columns_name
                writer.writerow(domains)
                csvfile.close()
                best_epoches = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                best_accs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


        for epoch in range(start_epoch, args.n_epochs + 1): # 在规定的训练epoch期间
            train_start = datetime.datetime.now()
            train_summary = trainer.train(epoch, train_loader,val_loader)  # 训练一次, return the epoch num and acc
            test_start = datetime.datetime.now()
            test_summary = trainer.test(epoch=epoch, test_loader=val_loader_pure_test)  # 测试一次 return only the acc
            test_end = datetime.datetime.now()
            domain_acc = test_summary['domain_acc']
            if not args.target==None :
                if domain_acc[1]>best_accs[1]: # domain_acc[0] is the number of current epoch
                    best_accs[1] = domain_acc[1]
                    best_epoches[1] = epoch
                    best_accs[1] = domain_acc[1]
                    best_epoches[1] = epoch
                    best_model = model
                    best_epoch = epoch
                    best_test_summary = test_summary
                    best_train_summary = train_summary
                    logger.record(best_epoch, train_summary=best_train_summary, test_summary=best_test_summary,
                                  model=best_model)  # 记录当前最佳模型
            else:
                for index in range(8):
                    if domain_acc[index+1]>best_accs[index+1]: # domain_Acc 第一个是epoch
                        best_accs[index+1] = domain_acc[index+1]
                        best_epoches[index+1] = epoch
                logger.record(epoch, train_summary=train_summary, test_summary=test_summary,
                              model=model)
            with open(log_file, "a+", encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(test_summary['domain_acc'])
                csvfile.close()
            print("training time of this epoch: " + str(test_start - train_start))
            print("testing time of this epoch: " + str(test_end - test_start))


        with open(log_file, "a+", encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(best_epoches)
            writer.writerow(best_accs)
            csvfile.close()

        logger.final_print()


if __name__ == '__main__':
    for index in range(3):
        print('第'+str(index)+'次实验')
        main()
