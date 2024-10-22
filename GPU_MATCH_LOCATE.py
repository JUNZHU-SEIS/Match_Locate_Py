from utils import (
	mark_templates,read_growclust,initialize,read_args,
	stack_all_nodes_of_CC_and_calculate_MAD,
	detect_peaks_on_all_nodes,reweight_by_common_station,
	deduplicate_in_space,deduplicate_in_time,write_catalog)
import pandas as pd
import os,torch,h5py,glob,yaml
torch.cuda.empty_cache()
import numpy as np
from obspy import read
from obspy.geodetics.base import locations2degrees as loc2deg
with open("config.yml", 'r') as stream:config = yaml.safe_load(stream)

def cc(x,y,eps,device):
	N,kernel = y.shape
	conv = torch.nn.Conv1d(N,N,kernel,bias=False).to(device)
	shift = torch.mean(y,axis=-1,keepdims=True)
	x -= shift
	y -= shift
	with torch.no_grad():
		conv.weight.data = torch.zeros_like(conv.weight.data)
		for i in range(N):conv.weight.data[i,i] = y[i]
		xy = conv(x)
		for i in range(N):conv.weight.data[i,i] = 1
		xx = conv(x*x)
		yy = torch.sum(y*y,axis=-1).unsqueeze(-1)
		CC = xy/torch.sqrt(xx*yy)
		CC[xx*yy<eps] = 0
	return CC,xx*yy<eps

class TemplateMatching(torch.utils.data.Dataset):
	def __init__(self,args):
		self.ctlg = read_growclust(args['ctlg_path'],nrows=args['ctlg_nrows'])
		continued = args['continue_previous_experiment']
		candidate_folders = sorted(glob.glob(os.path.join(args['root'],'*')))
		skip = -1
		if continued:
			exist_folders = [os.path.basename(x) for x in sorted(glob.glob(os.path.join(args['detection_folder'],'*')))]
			candidate_folders = [x for x in candidate_folders if (os.path.basename(x) not in exist_folders) or (exist_folders[-1]==os.path.basename(x))]
			if len(exist_folders):
				skip_ctlg = pd.read_csv(os.path.join(args['detection_folder'],exist_folders[-1],'raw_ctlg.txt'),sep='\s+')
				skip = max(skip_ctlg['template_id'])
				print('Skip until %d template ID for the first day.'%skip)
		self.skip = skip
		self.candidate_folders = candidate_folders
		self.tplt = h5py.File(args['tplt_path'],'r')['waveform']
		self.snr_threshold = args['snr_threshold']
		self.sampling_rate = args['sampling_rate']
		self.win0 = args['win0']
		self.duration = int((args['win0']+args['win1'])*self.sampling_rate)
		self.number_stations_threshold = args['number_stations_threshold']
		radius = args['radius']
		self.grids = self.generate_grid(radius['nx'],radius['ny'],radius['nz'],radius['dx'],radius['dy'],radius['dz'])
		self.max_number_channels = args['max_number_channels']
		self.max_number_chan_in_one_station = args['max_number_chan_in_one_station']
	def __len__(self):
		return len(self.tplt)*len(self.candidate_folders)
	def generate_grid(self,nx,ny,nz,dx,dy,dz):
		count = 0
		DX,DY,DZ = [],[],[]
		for i in (np.arange(2*nx+1)-nx)*dx:
			for j in (np.arange(2*ny+1)-ny)*dy:
				for k in (np.arange(2*nz+1)-nz)*dz:
					count += 1
					DX.append(i);DY.append(j);DZ.append(k)
		return pd.DataFrame({'dx':DX,'dy':DY,'dz':DZ})
	def filter_channels(self,d0,d1):
		k0 = set(d0.keys())
		folder = self.candidate_folders[d1]
		active_stations_on_that_day = glob.glob(os.path.join(folder,'*Z'))
		k1 = set(['.'.join(os.path.basename(x).split('.')[:-1]+['*']) for x in active_stations_on_that_day])
		common = list(k0 & k1)
		STATION,CHAN,T,PHASE,SNR,V,H = [],[],[],[],[],[],[]
		LON,LAT,ELE,DIST = [],[],[],[]
		ARRAY = []
		for k in common:
			data = d0[k]
			reference = data.attrs['reference']
			snr_p,snr_s = np.array(data.attrs['snr_p'])-self.snr_threshold,np.array(data.attrs['snr_s'])-self.snr_threshold
			ncomp = len(snr_p)
			flag_p,flag_s = np.sum(snr_p>=0)>=ncomp/2,np.sum(snr_s>=0)>=ncomp/2
			if flag_s:
				ts = data.attrs['ts']
				b = int(((ts+reference)-self.win0)*self.sampling_rate)
				e = b+self.duration
				if e<data[0].shape[1]:
					for ichan in np.argsort(-snr_s)[:self.max_number_chan_in_one_station]:
						STATION.append(k)
						CHAN.append(data.attrs['channels'][ichan])
						T.append(ts)
						PHASE.append('S')
						SNR.append(data.attrs['snr_s'][ichan])
						V.append(data.attrs['s_slowness_v'])
						H.append(data.attrs['s_slowness_h'])
						ARRAY.append(data[0][ichan,b:e])
						LON.append(data.attrs['stlo'])
						LAT.append(data.attrs['stla'])
						ELE.append(data.attrs['stel'])
						DIST.append(data.attrs['dist'])
			if flag_p:
				tp = data.attrs['tp']
				b = int(((tp+reference)-self.win0)*self.sampling_rate)
				e = b+self.duration
				if e<data[0].shape[1]:
					for ichan in np.argsort(-snr_p)[:self.max_number_chan_in_one_station]:
						STATION.append(k)
						CHAN.append(data.attrs['channels'][ichan])
						T.append(tp)
						PHASE.append('P')
						SNR.append(data.attrs['snr_p'][ichan])
						V.append(data.attrs['p_slowness_v'])
						H.append(data.attrs['p_slowness_h'])
						ARRAY.append(data[0][ichan,b:e])
						LON.append(data.attrs['stlo'])
						LAT.append(data.attrs['stla'])
						ELE.append(data.attrs['stel'])
						DIST.append(data.attrs['dist'])
		return pd.DataFrame({'station':STATION,'chan':CHAN,'t':T,'phase':PHASE,'snr':SNR,'vslowness':V,'hslowness':H,'stla':LAT,'stlo':LON,'stel':ELE,'dist':DIST}),ARRAY,folder

	def extract_continuous_waveforms(self,df,i):
		tmp_root = self.candidate_folders[i]
		data = []
		for station,chan in zip(df['station'],df['chan']):
			fname = os.path.join(tmp_root,station).replace('*',chan)
			tr = read(fname)[0]
			data.append(tr.data)
		return np.stack(data)

	def recalc_travel_times(self,evlo,evla,grids,stations):
		t_grids = []
		for i in range(len(grids)):
			grid = grids.iloc[i]
			t_grid = []
			for stla,stlo,v,h,t in zip(stations['stla'],stations['stlo'],stations['vslowness'],stations['hslowness'],stations['t']):
				delta_gcarc = loc2deg(evla+grid['dy'],evlo+grid['dx'],stla,stlo)-loc2deg(evla,evlo,stla,stlo)
				new_t = int((t+delta_gcarc*h+v*grid['dz'])*self.sampling_rate)
				t_grid.append(new_t)
			t_grid = np.array(t_grid)
			t_grids.append(t_grid)
		return np.stack(t_grids)
	def __getitem__(self,idx):
		ci,ti = idx//len(self.tplt),idx%len(self.tplt)
#		ci,ti = 3,6
		print('candidate %d (%s); template %d:'%(ci,self.candidate_folders[ci],ti))
		if ci==0 and ti<self.skip:return {'flag':0} # skip existing catalog
		event = self.ctlg.iloc[ti]
		ot = '%d%02d%02dT%02d:%02d:%06.3f'%(event['yr'],event['mon'],event['day'],event['hr'],event['min'],event['sec'])
		tp = self.tplt[str(ti)]
		stations,array,folder = self.filter_channels(tp,ci)
		if len(stations['station'].unique())<self.number_stations_threshold:return {'flag':0}
		x = np.stack(array)
		if len(stations)>self.max_number_channels:
			stations = stations.nlargest(self.max_number_channels,'snr')
			idx_nlargest = np.array(stations.index) # the array is shuffled
			x = x[idx_nlargest]
		reweight = reweight_by_common_station(list(stations['station']))
		y = self.extract_continuous_waveforms(stations,ci)
		t_grids = self.recalc_travel_times(event['LON'],event['LAT'],self.grids,stations)
		chan = list([x[:-1]+str(y) for x,y in zip(stations['station'],stations['chan'])])
		return {'flag':1,'template':x,'continuous':y,'t':np.array(stations['t']),
			'snr':np.array(stations['snr']),'vslowness':np.array(stations['vslowness']),
			'hslowness':np.array(stations['hslowness']),'template_idx':ti,'candidate_folder':folder,
			'stla':np.array(stations['stla']),'stlo':np.array(stations['stlo']),'stel':np.array(stations['stel']),
			't_grids':t_grids,'ot':ot,'evla':event['LAT'],'evlo':event['LON'],'evdp':event['DEPTH'],
			'grids':np.array(self.grids),'dist':np.array(stations['dist']),'chan':chan,'reweight':reweight}

conf = read_args()
args = initialize(config,conf)
print(args)
radius = args['radius']
distance = int(args['too_close_detections_to_remain']*args['sampling_rate'])
# if the template file (hdf5) alreay exists, comment this line to avoid marking templates agian.
#mark_templates(args)
ds = TemplateMatching(args)
pipe = torch.utils.data.DataLoader(ds,batch_size=1,num_workers=1,shuffle=False)
for x in pipe:
	if x['flag']==0:continue
	evla,evlo,evdp,folder,evid,ot = [x[key][0] for key in ['evla','evlo','evdp','candidate_folder','template_idx','ot']]
	print(folder,ot)
	template,continuous,shifts,snr,grids,dists,reweight = [x[key].to(args['device'])[0] for key in ['template','continuous','t_grids','snr','grids','dist','reweight']]
	channels = x['chan']
	evloc = {'evla':evla,'evlo':evlo,'evdp':evdp,'evid':evid,'ot':ot}
	CC,mask_zero_frac = cc(continuous,template,args['eps'],args['device'])
#	print('CC shape:',CC.shape)
	weight = ((snr/torch.sum(snr))+1/(dists*torch.sum(1/dists)))/2
	STACK,MAD,N,MASK,LEFTS = stack_all_nodes_of_CC_and_calculate_MAD(CC,mask_zero_frac,shifts,weight,reweight,args['minimum_cc'])
	time_idx,grid_idx,local_MADs = detect_peaks_on_all_nodes(STACK,MAD,N,MASK,args)
	time_idx,grid_idx,local_MADs,max_stack_over_all_nodes = deduplicate_in_space(time_idx,grid_idx,STACK,local_MADs)
	D = deduplicate_in_time(time_idx,max_stack_over_all_nodes,distance)
	if len(D)>0:write_catalog(D,time_idx,grid_idx,local_MADs,folder,LEFTS,evloc,grids,MAD,STACK,N,shifts,template,continuous,mask_zero_frac,channels,CC,dists,args)