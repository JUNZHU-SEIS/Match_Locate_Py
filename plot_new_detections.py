import matplotlib.pyplot as plt
import pandas as pd
from obspy import UTCDateTime,read
from obspy.geodetics.base import locations2degrees as loc2deg
import os,yaml
from utils import create_velocity_model,read_growclust
import numpy as np
R = 6371
deg2km = np.pi*R/180
tag = '20240718T011147'
waveform_root = '../Data/waveform_sac_filtered'
root = os.path.join('experiments',tag)
with open(os.path.join(root,'config.yml'),'r') as f:config=yaml.load(f,Loader=yaml.Loader)
savedir = os.path.join(os.path.join(root,'post_pdf'))
if not os.path.exists(savedir):os.makedirs(savedir)
else:
	for x in os.listdir(savedir):os.unlink(os.path.join(savedir,x))
vmodel = create_velocity_model(config['fvelo'],'eg.tvel')
tplt = read_growclust(config['ctlg_path'])
print(tplt)

def plot_waveform():
	df = pd.read_csv(os.path.join(root,'new.ctlg'),sep=' ')
	for time,lat,lon,depth,i in zip(df['ot'],df['lat'],df['lon'],df['dep'],df['template_id']):
		print('='*10+'\n',time,lat,lon,depth)
		ot = UTCDateTime(time)
		date = '%d%02d%02d'%(ot.year,ot.month,ot.day)
		st = read(os.path.join(waveform_root,date,'*E'))
		st = st.trim(ot,ot+100,pad=True,fill_value=0)
		P,S,G = [],[],[]
		for tr in st:
			tr.stats.coordinates = {}
			tr.stats.coordinates['latitude'] = tr.stats.sac['stla']
			tr.stats.coordinates['longitude'] = tr.stats.sac['stlo']
			gcarc = loc2deg(lon,lat,tr.stats.sac.stla,tr.stats.sac.stlo)
			G.append(gcarc)
			focal = depth + tr.stats.sac.stel/1e3
			tp = vmodel.get_travel_times(focal,distance_in_degree=gcarc,phase_list=['p','P'])
			ts = vmodel.get_travel_times(focal,distance_in_degree=gcarc,phase_list=['s','S'])
			if len(tp):P.append(tp[0].time)
			if len(ts):S.append(ts[0].time)
		P,S,G = np.array(P),np.array(S),np.array(G)
		start = max(0,min(P)-3)
		diff = np.median(np.diff(np.sort(G)))
		offset_min,offset_max = np.sort(G)[1]-diff,np.sort(G)[-2]+diff
		print(offset_max,offset_min,np.max(G),np.min(G),diff)
		fig = st.plot(type='section',ev_coord=(lon,lat),dist_degree=True,handle=True,
			orientation='horizontal',recordstart=start,recordlength=(min(100,max(S)+5)-start),
			offset_min=offset_min,offset_max=offset_max)
		plt.show()
		plt.close()
plot_waveform()