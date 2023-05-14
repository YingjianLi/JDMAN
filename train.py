import torch
import torch.optim as optim
from torch.autograd import Variable
import datetime
from model import resnet
import cmmd
import mmd
from my_loss import MC_Loss_dis_1_center,CDAN
import numpy as np
class Trainer:
    def __init__(self,args,model,adv_model):
        self.args = args
        self.decay = 1
        self.model = model
        self.adv_model = adv_model
        self.criterion_class = torch.nn.CrossEntropyLoss().cuda()
        self.mmd = mmd.MMD_loss()
        self.cmmd = cmmd.CMMD_loss()
        if self.args.MI:
            parameter_list = model.get_parameters() + adv_model.get_parameters()
        else:
            parameter_list = model.get_parameters()
        if self.args.criterion == 'mc_loss_center':
            self.mc_loss = MC_Loss_dis_1_center(num_classes=self.args.output_classes, feat_dim=3072,
                                       lamda1=self.args.lam1).cuda()  # lamda1 for inter_class loss

        if 'mc_loss'in self.args.criterion:
            self.optimizer4center = optim.SGD(
                self.mc_loss.parameters(),
                lr=1.0,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=True)
        if args.opti=='Adam':
            self.optimizer4nn= optim.Adam(
                model.parameters(),
                args.learn_rate,
                weight_decay=args.weight_decay,
                amsgrad=True
            )
        else:
            self.optimizer4nn = optim.SGD(
                parameter_list,
                args.learn_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=True
            )
        self.nGPU = args.nGPU
        self.learn_rate = args.learn_rate

    def train(self, epoch, train_loader, test_loader):

        # train_start = datetime.datetime.now()
        model = self.model
        torch.cuda.empty_cache()

        model.train()
        if self.args.MI:
            adv_model = self.adv_model
            adv_model.train()

        if self.args.opti == 'SGD':
            self.learning_rate(epoch)

        if self.args.target== None:
            # no target domain
            n_batches = len(train_loader[0])
            acc_avg = 0
            loss_avg = 0
            total = 0
            for i, (input_tensor, target) in enumerate(train_loader[0]):
                input_tensor = input_tensor.cuda()
                target = target.cuda()
                batch_size = target.size(0)
                input_var = Variable(input_tensor)
                target_var = Variable(target)
                output, feature = model(input_var)
                loss = self.criterion_class(output, target_var)
                acc = self.accuracy(output, target, (1,))[0]
                acc_avg += acc * batch_size
                loss_avg += loss.item() * batch_size
                total += batch_size
                if i % 100 == 0:
                    print("| Epoch[%d] [%d/%d]  Loss %1.4f  Acc %6.3f   LR %1.8f" % (
                        epoch,
                        i + 1,
                        n_batches,
                        loss_avg / total,
                        acc_avg / total,
                        self.decay * self.learn_rate))
                self.optimizer4nn.zero_grad()
                loss.backward()
                self.optimizer4nn.step()
            loss_avg /= total
            acc_avg /= total
            print("\n=> Epoch[%d]  Loss: %1.4f  Acc %6.3f  \n" % (
                epoch,
                loss_avg,
                acc_avg))
            summary = dict()

            summary['acc'] = acc_avg
            summary['loss'] = loss_avg
        else:
            # with target data
            # the second one is the target domain
            len_dataloader1 = len(train_loader[0])
            len_dataloader2 = len(test_loader[0])
            data_1_iter = iter(train_loader[0])
            data_2_iter = iter(test_loader[0])

            i = 1
            j = 1
            while i < len_dataloader1:
                # Training model using 1st dataset
                data_1 = data_1_iter.next()
                data_1_img, data_1_label = data_1[0].cuda(), data_1[1].cuda()

                output_1, features_1 = model(data_1_img)
                train_acc_1 = self.accuracy(output_1.data, data_1_label.data, (1,))[0]
                loss_1 = self.criterion_class(output_1, data_1_label)

                # Training model using 2nd dataset
                if j<len_dataloader2:
                    data_2 = data_2_iter.next()
                    j += 1
                else:
                    data_2_iter = iter(test_loader[0])
                    j = 1
                    data_2 = data_2_iter.next()
                data_2_img, data_2_label = data_2[0].cuda(), data_2[1].cuda()
                output_2, features_2 = model(data_2_img)
                labels_target_fake = torch.max(torch.nn.Softmax(dim=1)(output_2), 1)[1]

                if self.args.MI:
                    features = torch.cat((features_1, features_2), dim=0)
                    labels = torch.cat((data_1_label, labels_target_fake), -1)
                    outputs = torch.cat((output_1, output_2), dim=0)
                    softmax_out = torch.nn.Softmax(dim=1)(outputs)
                    entropy = Entropy(softmax_out)
                    adv_loss = CDAN([features, softmax_out], self.adv_model, entropy=None, coeff=resnet.calc_coeff(epoch*112+i), random_layer=None)
                    if 'mc_loss' in self.args.criterion:
                        weight = 2.0 / (1 + np.exp(-10 * ((epoch*112+i) / (self.args.n_epochs*112)))) - 1.0 #112 iter each epoch
                        if self.args.model == 'resnet50_with_adlayer_all':
                            da_loss = self.mc_loss(features, labels)
                            loss = loss_1 + self.args.alpha * weight * da_loss + self.args.beta*adv_loss
                            self.optimizer4nn.zero_grad()
                            self.optimizer4center.zero_grad()
                            loss.backward()
                            self.optimizer4nn.step()
                            self.optimizer4center.step()

                            if (i - 1) % 30 == 0:
                                print(
                                    'Ep:[{}/{}], Ba:[{}/{}], ls1:{:.4f}, lmc:{:.4f}, ladv:{:.4}, ac1:{:.4f}, lr:{:.4f}'.format(
                                        epoch, self.args.n_epochs, i, len_dataloader1, loss_1.item(),
                                        da_loss.item(), adv_loss.item(), train_acc_1.item(), self.decay * self.learn_rate))

                        else:
                            da_loss = self.mc_loss(features, labels)
                            loss = loss_1 + self.args.alpha * weight * da_loss + self.args.beta*adv_loss
                            self.optimizer4nn.zero_grad()
                            self.optimizer4center.zero_grad()
                            loss.backward()
                            self.optimizer4nn.step()
                            self.optimizer4center.step()

                            if (i - 1) % 30 == 0:
                                print(
                                    'Ep:[{}/{}], Ba:[{}/{}], ls1:{:.4f}, lmc:{:.4f}, ladv:{:.4}, ac1:{:.4f}, lr:{:.4f}'.format(
                                        epoch, self.args.n_epochs, i, len_dataloader1, loss_1.item(),
                                        da_loss.item(), adv_loss.item(), train_acc_1.item(), self.decay * self.learn_rate))
                    else:
                        loss = loss_1 + self.args.beta * adv_loss
                        self.optimizer4nn.zero_grad()
                        loss.backward()
                        self.optimizer4nn.step()
                        if (i - 1) % 30 == 0:
                            print(
                                'Ep:[{}/{}], Ba:[{}/{}], ls1:{:.4f}, ladv:{:.4}, ac1:{:.4f}, lr:{:.4f}'.format(
                                    epoch, self.args.n_epochs, i, len_dataloader1, loss_1.item(),
                                   adv_loss.item(), train_acc_1.item(), self.decay * self.learn_rate))

                else:
                    if 'mc_loss' in self.args.criterion:
                        labels = torch.cat((data_1_label, labels_target_fake), -1)
                        weight = 2.0 / (1 + np.exp(-10 * ((epoch*112+i) / (self.args.n_epochs*112)))) - 1.0 #112 iter each epoch
                        features = torch.cat((features_1, features_2), 0)
                        da_loss = self.mc_loss(features, labels)
                        loss = loss_1 + self.args.alpha * weight * da_loss
                        self.optimizer4nn.zero_grad()
                        self.optimizer4center.zero_grad()
                        loss.backward()
                        self.optimizer4nn.step()
                        self.optimizer4center.step()
                    else:
                        loss = loss_1
                        self.optimizer4nn.zero_grad()
                        loss.backward()
                        self.optimizer4nn.step()

                    if (i-1) % 100 == 0:
                        print(
                            'Ep:[{}/{}], Ba:[{}/{}], ls1:{:.4f}, lsd:{:.4f}, ac1:{:.4f}, lr:{:.4f}'.format(
                                epoch, self.args.n_epochs, i, len_dataloader1, loss_1.item(),
                                da_loss.item(),
                                train_acc_1.item(), self.decay * self.learn_rate))
                i += 1
            summary = dict()
            summary['acc'] = 0.0  # no use at all
        return summary

    def test(self, epoch=0, test_loader=None):
        with torch.no_grad():
            domain_acc = []
            domain_acc.append(epoch)
            if self.args.target==None: # testing on all datasets
                domains =['raf', 'aff', 'fer', 'ck+', 'mmi','jaf','oul','sfew']

                model = self.model
                model.eval()
                for dom in range(len(test_loader)):
                    acc_avg = 0
                    total = 0
                    predicted_list = []
                    target_list = []
                    for i, (input_tensor, target) in enumerate(test_loader[dom]):

                        if not self.nGPU==None:
                            input_tensor = input_tensor.cuda()
                            target = target.cuda()

                        batch_size = target.size(0)
                        input_var = Variable(input_tensor)
                        output,feature = model(input_var)

                        acc= self.accuracy(output.data, target,(1,))[0]
                        acc_avg += acc * batch_size

                        _, predicted = torch.max(output.data,1)
                        predicted_list.extend(list(predicted.data))
                        target_list.extend(list(target.data))
                        total += batch_size

                    acc_avg /= total
                    print("ACC on :  " +domains[dom]+'\n')
                    #print(self.get_confuse_matrix(predicted_list,target_list))
                    print("\n=> Test[%d]  Acc %6.3f\n" % (epoch, acc_avg))
                    domain_acc.append(acc_avg.item())

                torch.cuda.empty_cache()

                summary = dict()

                summary['acc'] = 0.0  # nouse
                summary['domain_acc']= domain_acc
            else:
                acc_avg = 0
                total = 0
                predicted_list = []
                target_list = []
                model = self.model
                model.eval()
                for i, (input_tensor, target) in enumerate(test_loader[0]):

                    if not self.nGPU==None:
                        input_tensor = input_tensor.cuda()
                        target = target.cuda()

                    batch_size = target.size(0)
                    input_var = Variable(input_tensor)

                    output,feature = model(input_var)

                    acc= self.accuracy(output.data, target,(1,))[0]
                    acc_avg += acc * batch_size

                    _, predicted = torch.max(output.data,1)
                    predicted_list.extend(list(predicted.data))
                    target_list.extend(list(target.data))

                    total += batch_size

                acc_avg /= total
                print("ACC on :  " +self.args.target+'\n')
                # print(self.get_confuse_matrix(predicted_list,target_list))
                print("\n=> Test[%d]  Acc %6.3f\n" % (epoch, acc_avg))
                domain_acc.append(acc_avg.item())

        torch.cuda.empty_cache()

        summary = dict()

        summary['acc'] = 0.0  # nouse
        summary['domain_acc']= domain_acc
        return summary

    def get_test_features(self, epoch, test_loader):

        n_batches = len(test_loader[0])

        acc_avg = 0.0
        total = 0

        model = self.model
        model.eval()
        predicted_list = []
        target_list = []
        feature_list = []
        for i, (input_tensor, target) in enumerate(test_loader[0]):

            if not self.nGPU==None :
                input_tensor = input_tensor.cuda()
                target = target.cuda()

            batch_size = target.size(0)
            input_var = Variable(input_tensor)
            # target_var = Variable(target)

            if self.args.criterion == 'DLP_LOSS':
                output, feature = model(input_var)
            else:
                output, feature = model(input_var)
            feature_list.extend(list(feature.cpu().detach().numpy()))

            acc = self.accuracy(output.data, target, (1,))[0]
            acc_avg += acc * batch_size

            _, predicted = torch.max(output.data, 1)
            predicted_list.extend(list(predicted.data))
            target_list.extend(list(target.cpu().detach().numpy().tolist()))

            total += batch_size
            if i % 100 == 0:
                print("| Test[%d] [%d/%d]   Acc %6.3f" % (
                    epoch,
                    i + 1,
                    n_batches,
                    acc_avg / total))

        acc_avg /= total
        print("\n=> Test[%d]  Acc %6.3f\n" % (
            epoch,
            acc_avg))

        summary = dict()

        summary['acc'] = acc_avg  # testset
        return summary, feature_list, target_list

    def get_confuse_matrix(self, predicted, target):
        np.set_printoptions(suppress=True,precision=4)
        num = int(self.args.output_classes)
        con_mat = np.zeros((num, num), np.float)
        for index in range(len(target)):
            con_mat[target[index]][predicted[index]] += 1
        for i in range(num):
            a = np.sum(con_mat, axis=1)  # sum of every row according to ECAN
            for j in range(num):
                con_mat[i][j] /= a[i]
        return con_mat

    def accuracy(self,output,target,topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _,pred = output.topk(maxk,1,True,True)
        pred = pred.t()
        correcct = pred.eq(target.view(1,-1).expand_as(pred))

        res = []

        for k in topk:
            correcct_k = correcct[:k].view(-1).float().sum(0,keepdim=True)

            res.append(correcct_k.mul_(100.0 / batch_size)[0])
        return res # the largest

    def learning_rate(self,epoch):
        self.decay = 0.1 **((epoch - 1) // self.args.decay)
        learn_rate = self.learn_rate * self.decay
        if learn_rate < 1e-7:
            learn_rate = 1e-7
        for param_group in self.optimizer4nn.param_groups:
            param_group['lr'] = learn_rate




