import pandas as pd
import matplotlib as mpl
import numpy as np
import os,glob,h5py,yaml
for axis in ['right','left','top']:mpl.rcParams['axes.spines.%s'%axis] = True
from obspy import UTCDateTime,read
from obspy.geodetics.base import locations2degrees as loc2deg
import matplotlib.pyplot as plt
from utils import read_growclust,create_velocity_model
DAYLENGTH = 24*3600
root = 'experiments'

def read_template_arrival(i,hdf):
	tplt = hdf[str(i)]
	return {k:{'P':tplt[k].attrs['tp'],'S':tplt[k].attrs['ts']} for k in tplt.keys()}

def common_station_pick_time_difference(it,iarrival,jt,jarrival):
	dt = []
	delta_i_j = it-jt
	for station in iarrival.keys():
		if station not in jarrival.keys():continue
		if iarrival[station]['P']>0 and jarrival[station]['P']>0:
			dt.append(abs(delta_i_j+iarrival[station]['P']-jarrival[station]['P']))
		if iarrival[station]['S']>0 and jarrival[station]['S']>0:
			dt.append(abs(delta_i_j+iarrival[station]['S']-jarrival[station]['S']))
	return dt

def remove_based_on_phase_pick(df,attr,hdf,interval_threhold=4):
	n = len(df)
	remove_idx = set()
	columns = ['relative_t','template_id',attr]
	arrivals = {}
	for i in range(n-1):
		if i in remove_idx:continue
		it,itplt,iattr = [df.loc[i,key] for key in columns]
		if i in arrivals:iarrival = arrivals[i]
		else:
			iarrival = read_template_arrival(itplt,hdf)
			arrivals[i] = iarrival
		for j in range(i+1,n):
			if j in remove_idx:continue
			jt,jtplt,jattr = [df.loc[j,key] for key in columns]
			if j in arrivals:jarrival = arrivals[j]
			else:
				jarrival = read_template_arrival(jtplt,hdf)
				arrivals[j] = jarrival
			dt = common_station_pick_time_difference(it,iarrival,jt,jarrival)
			if len(dt)==0 or min(dt)>interval_threhold:continue
			if iattr-jattr>=0:remove_idx.add(j)
			else:
				remove_idx.add(i)
				continue
	return list(remove_idx)

def remove_existing_templates(date,df,tplt,hdf,interval_threshold=4):
	arrivals = {}
	remove_idx = set()
	tplt = tplt[((tplt['time']>=date)&(tplt['time']<date+DAYLENGTH))]
	if len(tplt)==0:return list(remove_idx)
	for idx,i,it in zip(df.index,df['template_id'],df['relative_t']):
		if i in arrivals:iarrival = arrivals[i]
		else:
			iarrival = read_template_arrival(i,hdf)
			arrivals[i] = iarrival
		for j,jt in zip(tplt['evid'],tplt['relative_t']):
			if j in arrivals:jarrival = arrivals[j]
			else:
				jarrival = read_template_arrival(j,hdf)
				arrivals[j] = jarrival
			dt = common_station_pick_time_difference(it,iarrival,jt,jarrival)
			if len(dt)==0 or min(dt)>interval_threshold:continue
			remove_idx.add(idx)
	return list(remove_idx)

def write_ctlg(d,p,i):
	header = 'ot relative_t template_id lon lat dep Times_of_MAD Times_of_local_MAD CC n_high_cc_channels peakedness_cc prominence width\n'
	if i>=1:f = open(p,'a')
	else:
		f = open(p,'w');f.write(header)
	for ot,relative_t,template_id,lon,lat,dep,Times_of_MAD,\
		Times_of_local_MAD,CC,n_high_cc_channels,peakedness,prominence,width in zip(
			d['ot'],d['relative_t'],d['template_id'],d['lon'],d['lat'],
			d['dep'],d['Times_of_MAD'],
			d['Times_of_local_MAD'],d['CC'],
			d['n_high_cc_channels'],
			d['peakedness_cc'],d['prominence'],d['width']):
		line = '%s %.2f %d %f %f %f %f %f %f %.2f %d %f %f'%(ot,
		relative_t,template_id,lon,lat,dep,Times_of_MAD,
		Times_of_local_MAD,CC,n_high_cc_channels,peakedness,prominence,width)
		print(line)
		f.write(line+'\n')
	f.close()

class Deduplicate():
	def __init__(self,plot=False,components='ENZ'):
		tags = sorted(glob.glob(os.path.join(root,'*')))
		message = 'Please select which folder (experiment) you want to deduplicate:\n%s\n'%(
			'\n'.join(['%d: %s'%(i,tag) for i,tag in enumerate(tags)]))
		idx = int(input(message))
		root_dir = tags[idx]
		with open(os.path.join(root_dir,'config.yml'),'r') as f:config=yaml.load(f,Loader=yaml.Loader)
		with open('config_postprocess.yml','r') as f:self.threshold=yaml.load(f,Loader=yaml.Loader)
		with open(os.path.join(root_dir,'config_postprocess.yml'),'w') as f:yaml.dump(self.threshold,f,default_flow_style=False)
		self.root_dir = root_dir
		self.hdf = h5py.File(config['tplt_path'],'r')['waveform']
		ctlg = read_growclust(config['ctlg_path'],nrows=config['ctlg_nrows'])
		ctlg['relative_t'] = [hr*3600+min*60+sec for hr,min,sec in zip(ctlg['hr'],ctlg['min'],ctlg['sec'])]
		self.ctlg = ctlg
		count = self.run()
		print(count,'detections after deduplicate.')
		if plot:
			path = os.path.join(self.root_dir,'new.ctlg')
			if os.path.exists(path):
				self.vmodel = create_velocity_model(config['fvelo'],'eg.tvel')
				self.config = config
				self.components = components
				savedir = os.path.join(os.path.join(root_dir,'post_pdf'))
				if not os.path.exists(savedir):os.makedirs(savedir)
				else:
					for x in os.listdir(savedir):os.unlink(os.path.join(savedir,x))
				self.savedir = savedir
				self.plot_waveform(path)
	def plot_one_event(self,time,lon,lat,depth,offset_min=None,offset_max=None,start=None,duration=None,comp='E'):
		print('='*10+'\n',time,lon,lat,depth)
		ot = UTCDateTime(time)
		date = '%d%02d%02d'%(ot.year,ot.month,ot.day)
		st = read(os.path.join(self.config['root'],date,'*%s'%comp))
		st = st.trim(ot,ot+100,pad=True,fill_value=0)
		if offset_min==None:P,S,G = [],[],[]
		for tr in st:
			tr.stats.coordinates = {}
			tr.stats.coordinates['latitude'] = tr.stats.sac['stla']
			tr.stats.coordinates['longitude'] = tr.stats.sac['stlo']
			if offset_min==None:
				gcarc = loc2deg(lat,lon,tr.stats.sac.stla,tr.stats.sac.stlo)
				G.append(gcarc)
				focal = max(0,depth + tr.stats.sac.stel/1e3)
				tp = self.vmodel.get_travel_times(focal,distance_in_degree=gcarc,phase_list=['p','P'])
				ts = self.vmodel.get_travel_times(focal,distance_in_degree=gcarc,phase_list=['s','S'])
				if len(tp):P.append(tp[0].time)
				if len(ts):S.append(ts[0].time)
		if offset_min==None:
			P,S,G = np.array(P),np.array(S),np.array(G)
			diff = np.median(np.diff(np.sort(G)))
			start = max(0,np.sort(P)[1]-3)
			duration = min(100,np.sort(S)[-2]+5)-start
			offset_min = max(np.sort(G)[1]-diff,0)
			offset_max = np.sort(G)[-2]+diff
		print(start,duration)
		fig = st.plot(type='section',ev_coord=(lat,lon),dist_degree=True,handle=True,
			orientation='horizontal',recordstart=start,recordlength=duration,
			offset_min=offset_min,offset_max=offset_max)
		plt.tight_layout()
		return fig,offset_min,offset_max,start,duration
	def plot_waveform(self,path):
		df = pd.read_csv(path,sep=' ')
		for time,lat,lon,depth,i in zip(df['ot'],df['lat'],df['lon'],df['dep'],df['template_id']):
			offset_min0 = -1
			for comp in self.components:
				if offset_min0<0:fig,offset_min,offset_max,start,duration = self.plot_one_event(time,lon,lat,depth,comp=comp)
				else:fig,offset_min,offset_max,start,duration = self.plot_one_event(time,lon,lat,depth,comp=comp,
					offset_min=offset_min0,offset_max=offset_max0,start=start0,duration=duration0)
				fig.savefig(os.path.join(self.savedir,'%d_%s_%s_candidate.pdf'%(i,time,comp)))
				plt.close()
				evtp = self.ctlg[self.ctlg['evid']==i].iloc[0]
				fig,offset_min0,offset_max0,start0,duration0 = self.plot_one_event(evtp['time'],evtp['LON'],evtp['LAT'],evtp['DEPTH'],
					offset_min=offset_min,offset_max=offset_max,start=start,duration=duration,comp=comp)
				fig.savefig(os.path.join(self.savedir,'%d_%s_%s_template.pdf'%(i,time,comp)))
				plt.close()
		return
	def statistics(self,df,fday):
		_,ax = plt.subplots(2,2)
		df[['Times_of_MAD']].hist(bins=np.arange(8,30),ax=ax[0,0])
		df[['Times_of_local_MAD']].hist(bins=np.arange(8,30),ax=ax[0,1])
		df[['n_high_cc_channels']].hist(bins=np.arange(3,30),ax=ax[1,0])
		df[['peakedness_cc']].hist(bins=np.arange(0,5000,50),ax=ax[1,1])
		plt.tight_layout()
		plt.savefig(os.path.join(fday,'hist.pdf'))
		plt.close()
		plt.scatter(df['n_high_cc_channels'],df['Times_of_local_MAD'],s=.5)
		plt.tight_layout()
		plt.savefig(os.path.join(fday,'nchan_vs_peak.pdf'))
		plt.close()
	def run(self,priority=['relative_t','Times_of_MAD']):
		count = 0
		catalog_dir = os.path.join(self.root_dir,'catalog')
		fdays = sorted(glob.glob(os.path.join(catalog_dir,'*')))
		for i,fday in enumerate(fdays):
			print('Deduplicating:',fday)
			date = os.path.basename(fday)
			reference = UTCDateTime(date)
			df = pd.read_csv(os.path.join(fday,'raw_ctlg.txt'),sep=' ')
			self.statistics(df,fday)
			df['relative_t'] = [UTCDateTime(x)-reference for x in df['ot']]
			# remove events with relatively low local MAD, low peakedness, etc.
			df = df[((df['Times_of_local_MAD']>self.threshold['local_mad_threshold'])&(
				df['peakedness_cc']>self.threshold['peakedness_cc_threshold']))]
			df.sort_values(priority,inplace=True)
			df = df.reset_index(drop=True)
			# remove duplicate detection
			remove_idx = remove_based_on_phase_pick(df,'Times_of_MAD',self.hdf)
			duplicate = df.iloc[remove_idx].sort_values(priority)
			unique = df.loc[~df.index.isin(remove_idx)].sort_values(priority)
			duplicate.to_csv(os.path.join(fday,'duplicate.txt'),index=False,sep=' ')
			unique.to_csv(os.path.join(fday,'unique.txt'),index=False,sep=' ')
			# remove existing self/cross detection
			df = pd.read_csv(os.path.join(fday,'unique.txt'),sep=' ')
			remove_idx = remove_existing_templates(reference,df,self.ctlg,self.hdf)
			exist = df.iloc[remove_idx].sort_values(priority)
			new = df.loc[~df.index.isin(remove_idx)].sort_values(priority)
			exist.to_csv(os.path.join(fday,'exist.txt'),index=False,sep=' ')
			new.to_csv(os.path.join(fday,'new.txt'),index=False,sep=' ')
			count += len(new)
			write_ctlg(new,os.path.join(self.root_dir,'new.ctlg'),i)
		return count

d = Deduplicate(plot=True,components='ENZ')