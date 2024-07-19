import pandas as pd
import os,glob,h5py,yaml
from obspy import UTCDateTime
from utils import read_growclust
from obspy.geodetics.base import locations2degrees as loc2deg
import numpy as np

def read_template_arrival(i):
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

def remove_existing_templates(date,df,tplt,interval_threshold=4):
	arrivals = {}
	remove_idx = set()
	tplt = tplt[((tplt['time']>=date)&(tplt['time']<date+DAYLENGTH))]
	if len(tplt)==0:return list(remove_idx)
	for idx,i,it in zip(df.index,df['template_id'],df['relative_t']):
		if i in arrivals:iarrival = arrivals[i]
		else:
			iarrival = read_template_arrival(i)
			arrivals[i] = iarrival
		for j,jt in zip(tplt['evid'],tplt['relative_t']):
			if j in arrivals:jarrival = arrivals[j]
			else:
				jarrival = read_template_arrival(j)
				arrivals[j] = jarrival
			dt = common_station_pick_time_difference(it,iarrival,jt,jarrival)
			if len(dt)==0 or min(dt)>interval_threshold:continue
			remove_idx.add(idx)
	return list(remove_idx)

def remove_based_on_phase_pick(df,attr,interval_threshold=4):
	n = len(df)
	remove_idx = set()
	columns = ['relative_t','template_id',attr]
	arrivals = {}
	for i in range(n-1):
		if i in remove_idx:continue
		it,itplt,iattr = [df.loc[i,key] for key in columns]
		if i in arrivals:iarrival = arrivals[i]
		else:
			iarrival = read_template_arrival(itplt)
			arrivals[i] = iarrival
		for j in range(i+1,n):
			if j in remove_idx:continue
			jt,jtplt,jattr = [df.loc[j,key] for key in columns]
			if j in arrivals:jarrival = arrivals[j]
			else:
				jarrival = read_template_arrival(jtplt)
				arrivals[j] = jarrival
			dt = common_station_pick_time_difference(it,iarrival,jt,jarrival)
			if len(dt)==0 or min(dt)>interval_threshold:continue
			if iattr-jattr>=0:remove_idx.add(j)
			else:
				remove_idx.add(i)
				continue
	return list(remove_idx)

def remove_based_on_phase_pick_and_source_location(df,attr,dist_threhsold=50,interval_threshold=1):
	n = len(df)
	remove_idx = set()
	columns = ['lat','lon','dep','relative_t','template_id',attr]
	for i in range(n-1):
		if i in remove_idx:continue
		ilat,ilon,idep,it,itplt,iattr = [df.loc[i,key] for key in columns]
		iarrival = read_template_arrival(itplt)
		for j in range(i+1,n):
			if j in remove_idx:continue
			jlat,jlon,jdep,jt,jtplt,jattr = [df.loc[j,key] for key in columns]
			jarrival = read_template_arrival(jtplt)
			dt = common_station_pick_time_difference(it,iarrival,jt,jarrival)
			if len(dt)==0 or min(dt)>interval_threshold:continue
			gcarc = loc2deg(ilat,ilon,jlat,jlon)
			z = idep-jdep
			dist = np.sqrt((gcarc*deg2km)**2+z**2)
			if dist>dist_threhsold:continue
			if iattr-jattr>=0:remove_idx.add(j)
			else:
				remove_idx.add(i)
				continue
	return list(remove_idx)

def write_ctlg(d,p,i):
	header = 'ot relative_t template_id lon lat dep Times_of_MAD Times_of_local_MAD CC n_high_cc_channels\n'
	if i>=1:f = open(p,'a')
	else:
		f = open(p,'w');f.write(header)
	for ot,relative_t,template_id,lon,lat,dep,Times_of_MAD,\
		Times_of_local_MAD,CC,n_high_cc_channels in zip(
			d['ot'],d['relative_t'],d['template_id'],d['lon'],d['lat'],
			d['dep'],d['Times_of_MAD'],
			d['Times_of_local_MAD'],d['CC'],
			d['n_high_cc_channels']):
		line = '%s %.2f %d %f %f %f %f %f %f %d'%(ot,
		relative_t,template_id,lon,lat,dep,Times_of_MAD,
		Times_of_local_MAD,CC,n_high_cc_channels)
		print(line)
		f.write(line+'\n')
	f.close()

if __name__ == "__main__":
	R = 6371
	deg2km = np.pi*R/180
	DAYLENGTH = 24*3600
	root = 'experiments'
	tags = sorted(glob.glob(os.path.join(root,'*')))
	message = 'Please select which folder you want to deduplicate:\n%s\n'%(
	'\n'.join(['%d: %s'%(i,tag) for i,tag in enumerate(tags)]))
	idx = int(input(message))
	root_dir = tags[idx]
	with open(os.path.join(root_dir,'config.yml'),'r') as f:config=yaml.load(f,Loader=yaml.Loader)
	catalog_dir = os.path.join(root_dir,'catalog')
	fdays = sorted(glob.glob(os.path.join(catalog_dir,'*')))
	hdf = h5py.File(config['tplt_path'],'r')['waveform']
	ctlg = read_growclust(config['ctlg_path'],nrows=config['ctlg_nrows'])
	ctlg['time'] = [UTCDateTime('%d%02d%02dT%02d:%02d:%06.3f'%(yr,mon,day,hr,min,sec)) for yr,mon,day,hr,min,sec in zip(ctlg['yr'],ctlg['mon'],ctlg['day'],ctlg['hr'],ctlg['min'],ctlg['sec'])]
	ctlg['relative_t'] = [hr*3600+min*60+sec for hr,min,sec in zip(ctlg['hr'],ctlg['min'],ctlg['sec'])]
	priority = ['relative_t','Times_of_MAD']
	count = 0
	for i,fday in enumerate(fdays):
		print('Deduplicating existing templates:',fday)
		date = os.path.basename(fday)
		reference = UTCDateTime(date)
		df = pd.read_csv(os.path.join(fday,'unique.txt'),sep=' ')
		remove_idx = remove_existing_templates(reference,df,ctlg)
		exist = df.iloc[remove_idx].sort_values(priority)
		new = df.loc[~df.index.isin(remove_idx)].sort_values(priority)
		exist.to_csv(os.path.join(fday,'exist.txt'),index=False,sep=' ')
		new.to_csv(os.path.join(fday,'new.txt'),index=False,sep=' ')
		count += len(new)
		write_ctlg(new,os.path.join(root_dir,'new.ctlg'),i)
	print(count,'new events after deduplicate')