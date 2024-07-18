import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os,h5py,glob,yaml,argparse,torch
from datetime import datetime
from obspy import read,UTCDateTime
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
import numpy as np
#from obspy.signal.trigger import trigger_onset
from obspy.geodetics.base import locations2degrees as loc2deg
from math import log10
from scipy.signal import detrend,find_peaks
R = 6371
deg2km = np.pi*R/180
experiment_dir = 'experiments'

def plot_waveform(data,components,p,s,snr_p,snr_s,tag,sampling_rate,folder):
	_,ax = plt.subplots(len(components),1,sharex=True,gridspec_kw={'hspace':0})
	t = np.arange(data.shape[-1])/sampling_rate
	if p>0 and s>0:
		e = int(sampling_rate*(2*(s-p)+s))
	elif p>0:e = int(sampling_rate*(2*p))
	elif s>0:e = int(sampling_rate*(2*s))
	else:e = len(t)
	e = min(e,len(t))
	for a,x,rp,rs,comp in zip(ax,data,snr_p,snr_s,components):
		if p>0:
			a.axvline(x=p,color='b',ls='--',label='predict P')
			if p<t[e-1]:a.text(p,min(x[:e]),'SNR=%.1f'%rp,ha='right',va='bottom')
		if s>0:
			a.axvline(x=s,color='r',ls='--',label='predict S')
			if s<t[e-1]:a.text(s,min(x[:e]),'SNR=%.1f'%rs,va='bottom')
		a.plot(t[:e],x[:e],lw=.8,color='gray');a.set_ylabel(comp);a.set_yticks([])
	a.set_xlim(t[0],t[e-1])
	a.set_xlabel('Seconds since the origin time')
	ax[0].set_title(tag)
	ax[0].legend()
	plt.tight_layout()
	plt.savefig(os.path.join(folder,'%s.pdf'%str(tag)))
	plt.close()

def normalize(x):
	x -= np.mean(x,axis=1,keepdims=True)
	x = detrend(x,axis=1)
	return x

def get_snr(d,tp,ts,sampling_rate,win=3,snr=-999):
	x = d.copy()
	win = int(win*sampling_rate)
	tp = int(tp*sampling_rate)
	ts = int(ts*sampling_rate)
	snr_p,snr_s = [snr]*len(d),[snr]*len(d)
	npts = x.shape[-1]
	w = tp
	win = min(w,win,npts-w)
	if win>0:
		signal = normalize(x[:,w:w+win])
		noise = normalize(x[:,w-win:w])
		numerator = np.sum(signal**2,axis=1)
		denominator = np.sum(noise**2,axis=1)
		for i,(a,b) in enumerate(zip(numerator,denominator)):
			snr_p[i]=(10*log10(a/b) if a*b>0 else snr)
	w = ts
	win = min(w,win,npts-w)
	if win>0:
		signal = normalize(x[:,w:w+win])
		noise = normalize(x[:,w-win:w])
		numerator = np.sum(signal**2,axis=1)
		denominator = np.sum(noise**2,axis=1)
		for i,(a,b) in enumerate(zip(numerator,denominator)):
			snr_s[i]=(10*log10(a/b) if a*b>0 else snr)
	return snr_p,snr_s

def slowness(arr):
	dt_dgc = arr.ray_param*np.pi/180
	vslowness_for_source_depth_offset = -arr.ray_param/(R*np.tan(arr.takeoff_angle*np.pi/180))
	return dt_dgc,vslowness_for_source_depth_offset

def calc_travel_time_and_slowness(lat,lon,dep,stla,stlo,stel,vmodel):
	focal = max(dep+stel/1e3,0)
	gcarc = loc2deg(lat,lon,stla,stlo)
	tp = vmodel.get_travel_times(focal,distance_in_degree=gcarc,phase_list=['p','P'])
	ts = vmodel.get_travel_times(focal,distance_in_degree=gcarc,phase_list=['s','S'])
	p,p_h,p_v = -1,-1,-1
	s,s_h,s_v = -1,-1,-1
	if len(tp):
		p = tp[0].time
		p_h,p_v = slowness(tp[0])
	if len(ts):
		s = ts[0].time
		s_h,s_v = slowness(ts[0])
	dist = np.sqrt((gcarc*deg2km)**2+focal**2)
	return p,p_h,p_v,s,s_h,s_v,dist

def pseudo_travel_time(lat,lon,dep,stla,stlo,stel,vmodel):
	lon += .01;lat += 0;dep += 0
	focal = max(dep+stel/1e3,0)
	gcarc = loc2deg(lat,lon,stla,stlo)
	tp = vmodel.get_travel_times(focal,distance_in_degree=gcarc,phase_list=['p','P'])
	ts = vmodel.get_travel_times(focal,distance_in_degree=gcarc,phase_list=['s','S'])
	p,p_h,p_v = -1,-1,-1
	s,s_h,s_v = -1,-1,-1
	if len(tp):
		p = tp[0].time
		p_h,p_v = slowness(tp[0])
	if len(ts):
		s = ts[0].time
		s_h,s_v = slowness(ts[0])
	dist = np.sqrt((gcarc*deg2km)**2+focal**2)
	return p,s

def read_growclust(p,nrows=None):
	df = pd.read_csv(p,sep='\s+',names=['yr','mon',
		'day','hr','min','sec','evid','LAT','LON',
		'DEPTH','mag','qID','cID','nbranch',
		'qnpair','qndiffP','qndiffS','rmsP','rmsS',
		'eh','ez','et','latC','lonC','depC'])
	return df if nrows==None else df.iloc[:nrows]

def mark_templates(args,rows=None):
	df = read_growclust(args['ctlg_path'],nrows=args['ctlg_nrows'])
	print(df)
	tplt_path = args['tplt_path']
	root = args['root']
	vmodel = create_velocity_model(args['fvelo'],'eg.tvel')
	sampling_rate = args['sampling_rate']
	win0,win1 = args['win0_save_tplt'],args['win1_save_tplt']
	log = args['tplt_log']
	if os.path.exists(tplt_path):os.unlink(tplt_path)
	if args['plot']:
		for x in glob.glob(os.path.join(log,'*')):os.unlink(x)
	f = h5py.File(tplt_path,mode='w');grp = f.create_group('waveform')
	for i in range(len(df) if rows==None else rows):
		etry = df.iloc[i]
		t = UTCDateTime('%d%02d%02dT%02d:%02d:%06.3f'%(etry['yr'],etry['mon'],etry['day'],etry['hr'],etry['min'],etry['sec']))
		lat,lon,dep = etry['LAT'],etry['LON'],etry['DEPTH']
		print(t,lat,lon,dep,i)
		date = '%04d%02d%02d'%(t.year,t.month,t.day)
		flst = glob.glob(os.path.join(root,date,'*Z'))
		for fname in flst:
			fname = '.'.join(fname.split('.')[:-1]+['*'])
			print(fname,t)
			st = read(fname,starttime=t-win0,endtime=t+win1).sort()
			components = [tr.stats.component for tr in st]
			channels = [tr.stats.channel for tr in st]
			st_original = st.copy()
			st_original.trim(starttime=t-win0,endtime=t+win1,pad=True,fill_value=0)
			st.detrend('demean').detrend('linear').taper(.05,max_length=5).filter('bandpass',freqmin=2,freqmax=15)
			st.trim(starttime=t-win0,endtime=t+win1,pad=True,fill_value=0)
			meta = st[-1].stats.sac
			stlo,stla,stel = meta.stlo,meta.stla,meta.stel
			data = np.vstack([tr.data for tr in st_original])
			tag = '%d/%s'%(i,fname.split('/')[-1])
			p,p_h,p_v,s,s_h,s_v,dist = calc_travel_time_and_slowness(lat,lon,dep,stla,stlo,stel,vmodel)
			snr_p,snr_s = get_snr(data,p+win0,s+win0,sampling_rate)
			if args['plot']:plot_waveform(data,components,p+win0,s+win0,snr_p,snr_s,tag.replace('/','.'),sampling_rate,log)
			dset = grp.create_dataset(tag,(1,len(components),data.shape[-1]),dtype=np.float32)
			dset[0] = data
			dset.attrs['otime'] = str(t)
			dset.attrs['reference'] = win0
			dset.attrs['channels'] = channels
			dset.attrs['tp'] = p
			dset.attrs['p_slowness_h'] = p_h
			dset.attrs['p_slowness_v'] = p_v
			dset.attrs['snr_p'] = snr_p
			dset.attrs['ts'] = s
			dset.attrs['s_slowness_h'] = s_h
			dset.attrs['s_slowness_v'] = s_v
			dset.attrs['snr_s'] = snr_s
			dset.attrs['stlo'] = stlo
			dset.attrs['stla'] = stla
			dset.attrs['stel'] = stel
			dset.attrs['dist'] = dist
	f.close()
	return

def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--eps", default=None, type=float, help="Non-zero denominator")
	parser.add_argument("--sampling_rate", default=None, type=int, help="How many grids on the x axis")
	parser.add_argument("--number_stations_threshold", default=None, type=int, help="How many stations to validate a template")
	parser.add_argument("--number_high_cc_channels", default=None, type=int, help="How many channels of high CC to validate a template")
	parser.add_argument("--max_number_channels", default=None, type=int, help="At most how many channels to make a template")
	parser.add_argument("--snr_threshold", default=None, type=float, help="SNR threhold for templates")
	parser.add_argument("--minimum_cc", default=None, type=float, help="High CC threshold")
	parser.add_argument("--mad_threshold", default=None, type=float, help="MAD threhold for templates")
	parser.add_argument("--local_mad_threshold", default=None, type=float, help="MAD threhold for templates")
	parser.add_argument("--too_close_detections_to_remain", default=None, type=int, help="Threshold for close occurrences")
	parser.add_argument("--nx", default=None, type=int, help="How many grids on the x axis")
	parser.add_argument("--ny", default=None, type=int, help="How many grids on the y axis")
	parser.add_argument("--nz", default=None, type=int, help="How many grids on the z axis")
	parser.add_argument("--duplicate_nx", default=None, type=int, help="Deduplicate on x axis")
	parser.add_argument("--duplicate_ny", default=None, type=int, help="Deduplicate on the y axis")
	parser.add_argument("--duplicate_nz", default=None, type=int, help="Deduplicate on the z axis")
	parser.add_argument("--dx", default=None, type=float, help="Grid cell size on the x axis")
	parser.add_argument("--dy", default=None, type=float, help="Grid cell size on the y axis")
	parser.add_argument("--dz", default=None, type=float, help="Grid cell size on the z axis")
	parser.add_argument("--cuda", default=3, type=int, help="CUDA device available")
	parser.add_argument("--win0", default=None, type=int, help="Seconds before the template phase")
	parser.add_argument("--win1", default=None, type=int, help="Seconds after the template phase")
	parser.add_argument("--win0_save_tplt", default=None, type=int, help="Seconds before the origin time of the template")
	parser.add_argument("--win1_save_tplt", default=None, type=int, help="Seconds after the origin time of the template")
	parser.add_argument("--halfwin_for_local_mad", default=None, type=int, help="Seconds of the half win to calculate the local MAD")
	parser.add_argument("--plot", default=None, type=int, help="Whether plot the detection")
	parser.add_argument("--continue_previous_experiment", default=None, type=int, help="Whether continue previous experiment")
	parser.add_argument("--ctlg_nrows", default=None, type=int, help="For mini test, use a small number of templates to debug")
	args = parser.parse_args()
	return args

def create_velocity_model(path,vmout):
	vm = pd.read_csv(path)
	f = open(vmout,'w')
	f.write('eg\n'*2)
	for dep,vp,vs in zip(vm['depth'],vm['vp'],vm['vs']):
		f.write('%f %f %f 1\n'%(dep,vp,vs))
	f.close();build_taup_model(vmout)
	return TauPyModel(model=vmout.split('.')[0])

def initialize(config,conf):
	args = {}
	args['continue_previous_experiment'] = conf.continue_previous_experiment if conf.continue_previous_experiment else config['continue_previous_experiment']
	if args['continue_previous_experiment']==0:
		now = datetime.now()
		experiment = os.path.join(experiment_dir,'%d%02d%02dT%02d%02d%02d'%(now.year,now.month,now.day,now.hour,now.minute,now.second))
	else:
		previous_experiments = sorted(glob.glob(os.path.join(experiment_dir,'*')))
		if args['continue_previous_experiment']<0:
			try:experiment = previous_experiments[args['continue_previous_experiment']]
			except:exit('The experiment you chose does NOT exist. Please select again.')
		elif args['continue_previous_experiment']==1:
			message = 'Please select which experiment below you would continue by enter index:\n'
			lines = ''.join(['%s %d\n'%(x,i) for i,x in enumerate(previous_experiments)])
			idx = input(message+lines)
			try:experiment = previous_experiments[int(idx)]
			except:exit('The experiment you chose does NOT exist. Please select again.')
		else:exit('The number is NOT valid. Please choose another number between %d and 1.'%(-len(previous_experiments)))
	args['experiment'] = experiment
	args['plot'] = conf.plot if conf.plot else config['plot']
	args['eps'] = conf.eps if conf.eps else config['eps']
	args['sampling_rate'] = conf.sampling_rate if conf.sampling_rate else config['sampling_rate']
	args['snr_threshold'] = conf.snr_threshold if conf.snr_threshold else config['snr_threshold']
	args['minimum_cc'] = conf.minimum_cc if conf.minimum_cc else config['minimum_cc']
	args['number_high_cc_channels'] = conf.number_high_cc_channels if conf.number_high_cc_channels else config['number_high_cc_channels']
	args['max_number_channels'] = conf.max_number_channels if conf.max_number_channels else config['max_number_channels']
	args['win0'],args['win1'] = (conf.win0 if conf.win0 else config['win0']),(conf.win1 if conf.win1 else config['win1'])
	args['win0_save_tplt'] = conf.win0_save_tplt if conf.win0_save_tplt else config['win0_save_tplt']
	args['win1_save_tplt'] = conf.win1_save_tplt if conf.win1_save_tplt else config['win1_save_tplt']
	args['halfwin_for_local_mad'] = conf.halfwin_for_local_mad if conf.halfwin_for_local_mad else config['halfwin_for_local_mad']
	args['number_stations_threshold'] = conf.number_stations_threshold if conf.number_stations_threshold else config['number_stations_threshold']
	args['mad_threshold'] = conf.mad_threshold if conf.mad_threshold else config['mad_threshold']
	args['local_mad_threshold'] = conf.local_mad_threshold if conf.local_mad_threshold else config['local_mad_threshold']
	args['too_close_detections_to_remain'] = conf.too_close_detections_to_remain if conf.too_close_detections_to_remain else config['too_close_detections_to_remain']
	args['device'] = torch.device('cuda:%d'%conf.cuda if torch.cuda.is_available() else 'cpu')
	if torch.cuda.is_available():print('CUDA:%d is used'%conf.cuda)
	else:print('CPU is used')
	args['ctlg_path'] = config['ctlg_path']
	args['ctlg_nrows'] = conf.ctlg_nrows if conf.ctlg_nrows else config['ctlg_nrows']
	args['root'] = config['root']
	args['fvelo'] = config['fvelo']
	args['tplt_folder'] = config['tplt_folder']
	args['tplt_path'] = os.path.join(config['tplt_folder'],config['tplt_path'])
	args['tplt_log'] = os.path.join(config['tplt_folder'],config['tplt_log'])
	if args['plot'] and (not os.path.exists(args['tplt_log'])):os.makedirs(args['tplt_log'])
	args['detection_folder'] = os.path.join(experiment,config['detection_folder'])
	args['log'] = os.path.join(experiment,config['log'])
	radius = config['radius']
	if conf.nx:radius['nx'] = conf.nx
	if conf.ny:radius['ny'] = conf.ny
	if conf.nz:radius['nz'] = conf.nz
	if conf.dx:radius['dx'] = conf.dx
	if conf.dy:radius['dy'] = conf.dy
	if conf.dz:radius['dz'] = conf.dz
	if conf.duplicate_nx:radius['duplicate_nx'] = conf.duplicate_nx
	if conf.duplicate_ny:radius['duplicate_ny'] = conf.duplicate_ny
	if conf.duplicate_nz:radius['duplicate_nz'] = conf.duplicate_nz
	args['radius'] = radius
	if not os.path.exists(experiment):os.makedirs(experiment)
	if not os.path.exists(args['detection_folder']):os.makedirs(args['detection_folder'])
	if not os.path.exists(args['log']):os.makedirs(args['log'])
	with open(os.path.join(experiment,'config.yml'),'w') as outfile:yaml.dump(args,outfile,default_flow_style=False)
	return args

def generate_catalog(ctlg,folder,out_folder):
	date = os.path.basename(folder)
	fdate = os.path.join(out_folder,date)
	fname = os.path.join(fdate,'raw_ctlg.txt')
	header = 'ot template_id lon lat dep Times_of_MAD Times_of_local_MAD CC n_high_cc_channels\n'
	if len(ctlg)==0:return
	if not os.path.exists(fdate):os.makedirs(fdate)
	if not os.path.exists(fname):
		with open(fname,'w') as f:
			f.write(header)
	with open(fname,'a') as f:
		for _,ki in ctlg.items():
			for _,kj in ki.items():
				for _,kk in kj.items():
					for _,kl in kk.items():
						if 'mad' in kl:
							kl = kl['info']
							line = '%s %d %f %f %f %f %f %f %d\n'%(str(kl['ot']),kl['evid'],kl['evlo'],kl['evla'],kl['evdp'],kl['mad'],kl['local_mad'],kl['cc'],kl['n_high_cc_channels'])
							f.write(line)

def calc_MAD_times(x):
	median = torch.median(x)
	mad = torch.median(torch.abs(x-median))
	times_mad = (x-median)/mad
	return times_mad

def detect_events_on_all_nodes(CC,STACK,MAD,N,MASK_ZERO,MASK,LEFTS,SHIFTS,template,continuous,channels,ded,grids,folder,evloc,args):
	mad_threshold,sampling_rate,win0,win1 = args['mad_threshold'],args['sampling_rate'],args['win0'],args['win1']
	win = int((win0+win1)*sampling_rate)
	date = os.path.basename(folder)
	fdate = os.path.join(args['log'],date)
	if not os.path.exists(fdate):os.makedirs(fdate)
	candidate_t = UTCDateTime(date)
	peak_distance = int(args['too_close_detections_to_remain']*args['sampling_rate'])
	for stack,mad,n_high_cc_channels,left,grid,mask,SHIFT in zip(STACK,MAD,N,LEFTS,grids,MASK,SHIFTS):
#		print(stack.max(),grid,mad.max())
#		triggers = trigger_onset(mad.cpu(),mad_threshold,mad_threshold/2)
#		for s0,s1 in triggers:
#			idx = s0 + torch.argmax(mad[s0:s1+1])
		peaks,_ = find_peaks(mad.cpu(),height=mad_threshold,distance=peak_distance)
		for idx in peaks:
			if n_high_cc_channels[idx]<args['number_high_cc_channels'] or mask[idx]==0:continue
			local_mad = calc_local_mad(stack,idx,int(args['halfwin_for_local_mad']*args['sampling_rate']))
#			print(stack.max(),local_mad)
			if local_mad<args['local_mad_threshold']:continue
			peak_mad = mad[idx].cpu()
			detect_ot = candidate_t+float(idx-left)/sampling_rate+win0
			loc = {'evlo':evloc['evlo']+grid[0],'evla':evloc['evla']+grid[1],'evdp':evloc['evdp']+grid[2]}
			info = {'ot':detect_ot,'evid':evloc['evid'],'evlo':loc['evlo'],'evla':loc['evla'],'evdp':loc['evdp'],'mad':mad[idx],'cc':stack[idx],'n_high_cc_channels':n_high_cc_channels[idx],'local_mad':local_mad}
			flag = ded.run(grid[0],grid[1],grid[2],idx-left+int(win0*sampling_rate),peak_mad,info)
			if args['plot'] and flag:
				print(grid,info)
				plot_detection_overlaped(idx,left,win0,sampling_rate,SHIFT,win1,template,continuous,win,MASK_ZERO,channels,CC,stack,info,peak_mad,local_mad,evloc,fdate)

def torch_normalize_by_std(x):
	x = x-torch.mean(x,axis=1,keepdims=True)
	return x/torch.std(x,axis=1,keepdims=True)

def torch_normalize_by_max(x):
	x = x-torch.mean(x,axis=1,keepdims=True)
	return x/torch.max(torch.abs(x),axis=1,keepdims=True)[0]

def torch_normalize_continuous_and_template(c,t):
	t_mean = torch.mean(t,axis=1,keepdims=True)
	t = t-t_mean #to avoid mutating data, do NOT use t -=t_mean
	t_max = torch.max(torch.abs(t),axis=1,keepdims=True)[0]
	t = t/t_max
	c = (c-t_mean)/t_max
	return c,t

def plot_detection_overlaped(idx,left,win0,sampling_rate,SHIFT,win1,template,continuous,win,MASK_ZERO,channels,CC,stack,info,peak_mad,local_mad,evloc,fdate,scale=6):
	start_continuous = int(idx-left+win0*sampling_rate)
	shifts = torch.sort(SHIFT)
	b,e = shifts.values[0]/sampling_rate-win0,shifts.values[-1]/sampling_rate+win1
	b,e = b.cpu().numpy(),e.cpu().numpy()
	B,E = int(b*sampling_rate),int(e*sampling_rate)
	offset = int(shifts.values[0]-(B+win0*sampling_rate))
	half = int((E-B)/2)
	continuous = continuous[:,start_continuous+B:start_continuous+E]
	nC,nT = torch_normalize_by_max(continuous),torch_normalize_by_max(template)
	for i,con in enumerate(nC):
		plt.plot(np.arange(E-B)/sampling_rate+b,con.cpu()+i*2,color='gray',lw=1)
	for i,tem in enumerate(nT):
		plt.plot((np.arange(win)+int(SHIFT[i])+offset)/sampling_rate-win0,tem.cpu()+i*2,color='k',lw=.5)
	for i in range(len(SHIFT)):
		plt.text(max(min(int(SHIFT[i])/sampling_rate+win1,e),b),i*2,'%.2f'%CC[i,idx-left+SHIFT[i]],color=('r' if MASK_ZERO[i,idx-left+SHIFT[i]]==False else 'b'))
	if len(SHIFT)<=36:
		for i in range(len(SHIFT)):plt.text(b,i*2,channels[i][0],color='k',ha='right')
	cc = stack[idx-half:idx+half]
	cc_range = torch.max(cc)-torch.min(cc)
	cc_modified = (cc-torch.min(cc))/cc_range*scale
	cc_peak = (cc[half]-torch.min(cc))/cc_range*scale
	plt.scatter(x=[b+half/sampling_rate],y=[cc_peak+2*i+1],color='r')
	plt.plot(np.arange(2*half)/sampling_rate+b,cc_modified+i*2+1,color='gray',lw=.5)
	plt.text(b,i*2+1+scale/2,'Stack CC',color='k')
	plt.yticks([])
	plt.xlabel('Seconds since %s'%str(info['ot']))
	plt.title('%d MAD;%d local MAD;%.2f CC\n(Template %d: %s)'%(peak_mad,local_mad,stack[idx],evloc['evid'],str(evloc['ot'])))
	plt.xlim(b,e)
	plt.ylim(-1,i*2+1+scale+.1)
	plt.tight_layout()
	for axis in ['right','left','top']:mpl.rcParams['axes.spines.%s'%axis] = False
	plt.savefig(os.path.join(fdate,'%d_%s.pdf'%(evloc['evid'],str(info['ot']))))
	plt.close()

def calc_local_mad(x,i,halfwin):
	b = max(i-halfwin,0)
	e = b+halfwin*2
	relative_idx = i-b
	sub = x[b:e]
	median = torch.median(sub)
	mad = torch.median(torch.abs(sub-median))
	times_mad = (sub-median)/mad
	return times_mad[relative_idx]

def meanCC(x,mask,shift,weight,cc_threshold):
	c = torch.zeros_like(x)
	m = torch.zeros_like(mask)
	w = torch.ones_like(x)*torch.unsqueeze(weight,1)
	left,right = torch.min(shift),torch.max(shift)
	for i,s in enumerate(shift):
		s = int(s-left)
		c[i] = torch.roll(x[i],-s)
		m[i] = torch.roll(mask[i],-s)
	n_high_cc_channels = torch.sum(c>cc_threshold,0)
	c*=w;w[m]=0
	stack = torch.sum(c,0)/torch.sum(w,0)
	return stack[:-(right-left)],left,n_high_cc_channels[:-(right-left)]

def reweight_by_common_station(l):
	C = []
	for x in l:C.append(1/l.count(x))
	return np.array(C)

def stack_all_nodes_of_CC_and_calculate_MAD(x,mask,shifts,weight,reweight,cc_threshold):
	lefts,rights = torch.min(shifts,1)[0],torch.max(shifts,1)[0]
	n_node,_ = shifts.shape
	_,nt = x.shape
	STACK = torch.zeros((n_node,nt))
	N = torch.zeros_like(STACK)
	MASK = torch.ones_like(STACK)
	MAD = torch.zeros_like(STACK)
	for node,shift in enumerate(shifts):
		c = torch.zeros_like(x)
		m = torch.zeros_like(mask)
		w = torch.ones_like(x)*torch.unsqueeze(weight,1)
		for i,s in enumerate(shift):
			s = int(s-lefts[node])
			c[i] = torch.roll(x[i],-s)
			m[i] = torch.roll(mask[i],-s)
		n_high_cc_channels = torch.sum((torch.abs(c)>cc_threshold)*torch.unsqueeze(reweight,1),0)
		c*=w;w[m]=0
		stack = torch.sum(c,0)/torch.sum(w,0)
		STACK[node] = stack
		N[node] = n_high_cc_channels
		MASK[node,-(rights[node]-lefts[node]):] = 0
		MAD[node,:-(rights[node]-lefts[node])] = calc_MAD_times(stack[:-(rights[node]-lefts[node])])
	return STACK,MAD,N,MASK,lefts