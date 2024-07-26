from utils import deduplicate_in_time,calc_lMAD
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
device = torch.device('cuda:0')
#x = torch.arange(20).to(device)
#y = torch.Tensor([.1,9,.1,9,2,6,9,1,20,1,3,5,10,1,22,5,12,122,6,8]).to(device)
x = torch.arange(20).to(device)+20
y = torch.randn(20).to(device)
print(x)
print(y)
t = time.time()
z = deduplicate_in_time(x,y,1,device=device)
print(time.time()-t)

x = torch.Tensor([[1,2,3],[4,2,5]])
y = torch.where(torch.sum(x,dim=0)>6)
print(y)
df = pd.read_csv('/home/lilab/jzhu/data/Work/PLAN/Huoshan/LOC-FLOW/Match_Locate_Py/experiments/20240725T093150/catalog/20210426/raw_ctlg.txt',
	sep=' ')
print(df.describe())
df.hist(column=['Times_of_local_MAD'])
plt.savefig('lmad.pdf')

t0 = time.time()
x = torch.arange(2000000).to(device)+20
t = calc_lMAD(x,1000)
print(time.time()-t0)