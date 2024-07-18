import matplotlib.pyplot as plt
import pandas as pd

def plot_waveform():
	cmap = {x:y for x,y in zip('PS','kr')}
	df = read_growclust()
	df0 = pd.read_csv('../pyocto/events_50Hz.csv')
	ph = pd.read_csv('../pyocto/phases_50Hz.csv')
	savedir = os.path.join(vis,'pdf')
	if not os.path.exists(savedir):os.makedirs(savedir)
	else:
		for x in glob.glob(os.path.join(savedir,'*')):os.unlink(x)
	vm = create_velocity_model()
	for time,lat,lon,depth,i in zip(df['time'],df['lat'],df['lon'],df['depth'],df['id']):
		print('='*10+'\n',time,lat,lon,depth,i)
		etry = df0[df0.idx==i].iloc[0]
		lat0,lon0,depth0 = etry.latitude,etry.longitude,etry.depth
		picks = ph[ph['event_idx']==i]
		start,end = UTCDateTime(picks['time'].min())-10,UTCDateTime(picks['time'].max())+10
		ot = UTCDateTime(str(time))
		date = '%d%02d%02d'%(ot.year,ot.month,ot.day)
		st = read(os.path.join(waveform_root,date,'*E'))
		st = st.trim(start,end,pad=True,fill_value=0)
		st.detrend('demean').detrend('linear').taper(.05).filter(
			'bandpass',freqmin=2,freqmax=15)
		plot_phase_and_stream(st,picks,ot,i,lat,lon,depth,lat0,lon0,depth0,vm,cmap,savedir)
