import os
import argparse
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from train import train
from train import eval
from model import MobileNetV3
from center_loss import CenterLoss
from dataset_process import dataprocess
import moxing as mx

parser = argparse.ArgumentParser()
parser.add_argument('--data_url', default='obs', help='dir of dataset')  # 数据集存储路径
parser.add_argument('--train_url', default='obs', help='dir of output')  # 模型输出路径
parser.add_argument('--homepath', type=str, default=os.environ['HOME'])
parser.add_argument('--train_label_txt', type=str, default='./train_label.txt')
parser.add_argument('--test_label_txt', type=str, default='./test_label.txt')
parser.add_argument('--num_epoch', type=int, default=100, help='the number of training epoches')  # 训练次数，默认为10
parser.add_argument('--batch_size', type=int, default=256,
                    help='the number of samples one batch')  # 一个batch的数据量，默认32
parser.add_argument('--lr_model',type=int,default=0.001)
parser.add_argument('--lr_centloss',type=int,default=0.5)
parser.add_argument('--gamma',type=int,default=0.5,help='learning rate decay')
parser.add_argument('--step',type=int,default=20)
parser.add_argument('--weight',type=int,default=1)
parser.add_argument('--num_gpus', type=int, default=1, help='the num of gpu')  # 使用的GPU数量
parser.add_argument('--app_url', help='dir of app_url',default='s3://my-obs-cn-north1/aifood-train')  # main.py的上级路径
parser.add_argument('--boot_file', help='dir of boot_file',default='s3://my-obs-cn-north1/aifood-train')  # mian.py的路径
parser.add_argument('--log_file', help='dir of log_file',default='s3://my-obs-cn-north1/aifood-train')  # 输出日志的路径
parser.add_argument('--init_method', help='init_method')  # init_method
args = parser.parse_args()


def main():
    homepath=os.environ['HOME']
    datapath=os.path.join(homepath,'data') 
    mx.file.copy_parallel(args.data_url, datapath)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MobileNetV3().to(device)
    centerloss = CenterLoss(num_classes=75, feat_dim=1280,use_gpu=True)
    cross_entropy = nn.CrossEntropyLoss()
    optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
    optimizer_centloss = torch.optim.SGD(centerloss.parameters(), lr=args.lr_centloss)
    train_iterator, test_iterator = dataprocess(train_label_path=args.train_label_txt,
                                                data_dirtory=datapath,
                                                test_label_path=args.test_label_txt,
                                                batch_size=args.batch_size)

    if args.step > 0:
        scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.step, gamma=args.gamma)

    if not (os.path.isdir(os.path.join(args.homepath, 'model'))):
        os.makedirs(os.path.join(args.homepath, 'model'))
    tmp_accuracy = 0

    for epoch in range(args.num_epoch):
        if args.step > 0:
            scheduler.step()
        train_loss, train_acc = train(model=model, device=device, train_iterator=train_iterator,
                                      optimizer_model=optimizer_model, optimizer_centloss=optimizer_centloss,
                                      criterion1=cross_entropy, criterion2=centerloss,
                                      weight_centloss=args.weight)
        test_loss, test_acc = eval(model=model, device=device, test_iterator=test_iterator,
                                   criterion1=cross_entropy, criterion2=centerloss,
                                   weight_centloss=args.weight_centloss)
        print('|Epoch:', epoch + 1, '|Train loss', train_loss.item(), '|Train acc:', train_acc.item(),
              '|Test loss', test_loss.item(), '|Test acc', test_acc.item())
        if test_acc > tmp_accuracy:
            MODEL_SAVE_PATH = os.path.join(args.homepath, 'model', 'mymodel_{}.pth'.format(epoch))
            torch.save(model.save_dict(), MODEL_SAVE_PATH)
            tmp_accuracy = test_acc
    mox.file.copy(MODEL_SAVE_PATH,os.path.join(args.train_url,'model/mymodel.pth'))


if __name__ == '__main__':
    main()
