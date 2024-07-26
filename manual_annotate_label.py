import pandas as pd
import os
root = '/home/lilab/jzhu/data/Work/PLAN/Huoshan/LOC-FLOW/Match_Locate_Py/debug/check_mad_threshold/catalog/20210426/'
df = pd.read_csv(os.path.join(root,'raw_ctlg.txt'),sep=' ',dtype={'ot':str})
annotation = []
if __name__=="__main__":
	for i in range(len(df)):
		etry = df.iloc[i]
		print(etry)
		annotation.append(int(input('Is this event true or false detection?')))
	df['manual_label'] = annotation
	df.to_csv(os.path.join(root,'raw_ctlg.label'),index=False,sep=' ')
	print(df)