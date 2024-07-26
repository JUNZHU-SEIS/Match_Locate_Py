import pandas as pd
import os
from manual_annotate_label import root
df = pd.read_csv(os.path.join(root,'raw_ctlg.label'),sep=' ')
column = ['Times_of_MAD','Times_of_local_MAD','peakedness_cc','n_high_cc_channels','n_high_cc_stations']
for k in column:df[k] = df[k].abs()
print(df[df['manual_label']==1][column].describe())
print(df[df['manual_label']==0][column].describe())
print(df[((df['Times_of_MAD']>11)&(df['Times_of_local_MAD']>11))][column+['manual_label','template_id']])