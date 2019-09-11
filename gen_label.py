# -*- coding: UTF-8 -*-
import os
import random


def gen_label(input_dir='D:\\资料\\培训\\aifood\\aifood\\aifood\\images',
              small_label_txt='.\\small_labels_25c.txt'):
    cls_large = 0
    cls_small = 0
    small_label_list = []
    with open(small_label_txt, encoding='UTF-8') as small_labels:
        for line in small_labels.readlines():
            small_label_list.append(line.split('\n')[0])
    with open('.\\small_label.txt', 'a') as small_label_write:
        with open('.\\large_label.txt', 'a') as large_label_write:
            for i in os.listdir(input_dir):
                for j in os.listdir(os.path.join(input_dir, i)):
                    if j in small_label_list:
                        for k in os.listdir(os.path.join(input_dir, i, j)):
                            small_label_write.write('images/' + i + '/' + j + '/' + k + ' ' + str(cls_small) + '\n')
                        cls_small = cls_small + 1
                    else:
                        for k in os.listdir(os.path.join(input_dir, i, j)):
                            large_label_write.write('images/' + i + '/' + j + '/' + k + ' ' + str(cls_large) + '\n')
                        cls_large = cls_large + 1
            large_label_write.close()
        small_label_write.close()


def gen_random_num(num_count, min, max):
    tmp_num = random.randint(min, max)
    num_list = []
    while len(num_list) < num_count:
        if tmp_num not in num_list:
            num_list.append(tmp_num)
    return num_list


def split_train_test(label_path='.\\large_label.txt',
                     train_label='.\\train_label.txt',
                     test_label='.\\test_label.txt'):
    i = 0
    with open(label_path) as f1:
        with open(train_label,'a') as f2:
            with open(test_label,'a') as f3:
                lines = f1.readlines()
                while i < int(len(lines) / 4):
                    f3.write(lines[random.randint(0, len(lines) - 1)])
                    lines.remove(lines[random.randint(0, len(lines) - 1)])
                    i += 1
                for j in lines:
                    f2.write(j)


if __name__ == '__main__':
    split_train_test()
