from matplotlib import pyplot as plt
import json
import math
import argparse

parser = argparse.ArgumentParser(description='Plot for lr range test log')
parser.add_argument("--low",default=1e-5,type=float)
parser.add_argument('--high',default=50,type=float)
parser.add_argument('--filename',type=str,default='./lr_range_test_data/sgd_lr_range_find_minibatch.json')

args = parser.parse_args()

low = math.log2(args.low)
high = math.log2(args.high)

trainloss = json.load(open(args.filename))
x = [2**(low+(high-low)*i/len(trainloss)) for i in range(int(len(trainloss)))]
y = trainloss
plt.plot(x,y)
plt.xscale('log')
plt.xlabel('learning rate', fontsize=12)
plt.ylabel('loss')
plt.title(args.filename.split('/')[-1].split('_')[0])
plt.show()