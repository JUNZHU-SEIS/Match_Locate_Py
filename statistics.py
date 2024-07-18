import os
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
root = 'experiments'
folders = glob.glob(os.path.join(root,'*'))
for folder in folders:
	fdays = glob.glob(os.path.join(folder,'catalog','*'))
	for fday in fdays:
		print(fday)
		ctlg = pd.read_csv(os.path.join(fday,'uniq.txt'),sep=' ')
		ctlg['Times_of_MAD'].hist(bins=np.arange(20,200))
		plt.tight_layout()
		plt.savefig(os.path.join(fday,'hist.pdf'))
		plt.close()
