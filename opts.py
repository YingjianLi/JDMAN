import argparse

parser = argparse.ArgumentParser(description='Parser for all the training options')

# General options
parser.add_argument('-shuffle', action='store_true', help='Reshuffle data at each epoch')
parser.add_argument('-train_record', action='store_true', help='Path to save train record')
parser.add_argument('-save_best_model_only', action='store_true', help='only save the best model')
parser.add_argument('-save_every_model', action='store_true', help='only save all models')
parser.add_argument('-test_only', action='store_true', help='Only conduct test on the validation set')
parser.add_argument('-aug', action='store_true', help='augmentation the training images')


parser.add_argument('-model', required=True, help='Model type when we create a new one')
parser.add_argument('-MI', action='store_true', help='using Mutual Information')
parser.add_argument('-pretrained', required=True, help='Model type when we create a new one')
parser.add_argument('-train_list0', required=True, help='Path to rafdb2.0 ')
parser.add_argument('-train_list1', required=True, help='Path to rafdb')
parser.add_argument('-test_list1', required=True, help='Path to data directory')
parser.add_argument('-test_list2', required=True, help='Path to data directory')
parser.add_argument('-test_list3', required=True, help='Path to data directory')
parser.add_argument('-test_list4', required=True, help='Path to data directory')
parser.add_argument('-test_list5', required=True, help='Path to data directory')
parser.add_argument('-test_list6', required=True, help='Path to data directory')
parser.add_argument('-test_list7', required=True, help='Path to data directory')
parser.add_argument('-test_list8', required=True, help='Path to data directory')
parser.add_argument('-test_list9', required=True, help='Path to data directory')
parser.add_argument('-test_list10', required=False, help='Path to data directory')
parser.add_argument('-print', required=True, help='information of the training hyper parameters')
parser.add_argument('-target', default=None, required=False,type=str, help='target domain')
parser.add_argument('-get_features', default='source', required=False,type=str, help='get source or target features')
parser.add_argument('-train_data', required=True, help='training data')
parser.add_argument('-test_data', required=True, help='testing data')
parser.add_argument('-save_path', required=True, help='train:the dir to save train record,'
                                                      ' test_only: the model pth file path')
parser.add_argument('-log_path', required=True, help='path to save csv file')
parser.add_argument('-output_classes', required=True, type=int, help='Num of emo classes')

# Training options
parser.add_argument('-learn_rate', default=1e-2, type=float, help='Base learning rate of training')
parser.add_argument('-momentum', default=0.9, type=float, help='Momentum for training')
parser.add_argument('-weight_decay', default=0.0005, type=float, help='Weight decay for training')
parser.add_argument('-alpha', default=0.01, type=float, help='Weight of feature loss')
parser.add_argument('-beta', default=0.01, type=float, help='Weight of feature loss')
parser.add_argument('-lam1', default=1.0, type=float, help='Weight of cross domain loss')
parser.add_argument('-n_epochs', default=20, type=int, help='Training epochs')
parser.add_argument('-batch_size', default=64, type=int, help='Size of mini-batches for each iteration')
parser.add_argument('-criterion', default='CrossEntropy', help='focal_loss,my_loss,or none')
parser.add_argument('-opti', default='SGD', help='optimizer,SGD,Adam')

# Model options
parser.add_argument('-resume', action='store_true', help='Whether continue to train from a previous checkpoint')
parser.add_argument('-nGPU', default="0,1,2,3",type=str, help='which GPUs for training')
parser.add_argument('-workers', default=4, type=int, help='Number of subprocesses to to load data')
parser.add_argument('-decay', default=8, type=int, help='LR decay')
parser.add_argument('-size', default=224, type=int)
parser.add_argument('-save_result', action='store_true', default=False, help='save result when evaluating')
args = parser.parse_args()
