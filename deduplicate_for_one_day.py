import pandas as pd
import os,glob,h5py,yaml
from obspy import UTCDateTime
from obspy.geodetics.base import locations2degrees as loc2deg
import numpy as np
R = 6371
deg2km = np.pi*R/180
root_dir = 'experiments/20240718T011147'
catalog_dir = os.path.join(root_dir,'catalog')
with open(os.path.join(root_dir,'config.yml'),'r') as f:config=yaml.load(f,Loader=yaml.Loader)
fdays = glob.glob(os.path.join(catalog_dir,'*'))
hdf = h5py.File(config['tplt_path'],'r')['waveform']

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

def remove_based_on_phase_pick(df,attr,interval_threhold=4):
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
			if len(dt)==0 or min(dt)>interval_threhold:continue
			if iattr-jattr>=0:remove_idx.add(j)
			else:
				remove_idx.add(i)
				continue
	return list(remove_idx)

def remove_based_on_phase_pick_and_source_location(df,attr,dist_threhsold=50,interval_threhold=1):
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
			if len(dt)==0 or min(dt)>interval_threhold:continue
			gcarc = loc2deg(ilat,ilon,jlat,jlon)
			z = idep-jdep
			dist = np.sqrt((gcarc*deg2km)**2+z**2)
			if dist>dist_threhsold:continue
			if iattr-jattr>=0:remove_idx.add(j)
			else:
				remove_idx.add(i)
				continue
	return list(remove_idx)

priority = ['relative_t','Times_of_MAD']
count = 0
for fday in fdays:
	print('Deduplicating:',fday)
	date = os.path.basename(fday)
	reference = UTCDateTime(date)
	df = pd.read_csv(os.path.join(fday,'raw_ctlg.txt'),sep=' ')
	df['relative_t'] = [UTCDateTime(x)-reference for x in df['ot']]
	df.sort_values(priority,inplace=True)
	df = df.reset_index(drop=True)
	remove_idx = remove_based_on_phase_pick(df,'Times_of_MAD')
	duplicate = df.iloc[remove_idx].sort_values(priority)
	unique = df.loc[~df.index.isin(remove_idx)].sort_values(priority)
	duplicate.to_csv(os.path.join(fday,'duplicate.txt'),index=False,sep=' ')
	unique.to_csv(os.path.join(fday,'unique.txt'),index=False,sep=' ')
	count += len(unique)
print(count,'unique events after deduplicate')