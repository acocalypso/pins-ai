"""Machine-readable catalog transcribed from nitr57/pins commit 14b50c4, NINA.Mcp/Tools/Catalog.

IMPORTANT: provenance is a WIP commit, not a live tools/list discovery.
Refresh/check against deployed MCP tools/list before training or inference.
"""
from __future__ import annotations

COMMIT = '14b50c4e09ef3904f5c8db5a29f05eb266e863a5'
REPO = f'https://github.com/nitr57/pins/blob/{COMMIT}/'
DEVICES = ['camera','mount','focuser','filter_wheel','guider','rotator','flat_device','dome','switch','weather','safety_monitor']

def prop(t: str, desc: str, **kw):
    return {'type':t,'description':desc,**kw}

def num(desc,lo=None,hi=None,integer=False,**kw):
    d=prop('integer' if integer else 'number',desc,**kw)
    if lo is not None: d['minimum']=lo
    if hi is not None: d['maximum']=hi
    return d

def txt(desc,maxlen=200,**kw): return prop('string',desc,minLength=1,maxLength=maxlen,**kw)
def boo(desc,default=None): return prop('boolean',desc,**({'default':default} if default is not None else {}))
def coords_ra(): return {'type':['number','string'],'description':'J2000 right ascension in HOURS (0 <= ra < 24); number or sexagesimal string, NOT degrees.','minimum':0,'exclusiveMaximum':24}
def coords_dec(): return {'type':['number','string'],'description':'J2000 declination in DEGREES (-90..90); number or sexagesimal string.','minimum':-90,'maximum':90}
def schema(fields=None,required=()):
    d={'type':'object','properties':dict(fields or {}),'additionalProperties':False}
    if required:d['required']=list(required)
    return d

TOOLS={}
def add(name, title, description, fields=None, required=(), *,kind='query', source='',job=False,guard=None, approval=False, must_confirm=False):
    fields=dict(fields or {})
    if job: fields['wait_seconds']=num('Seconds to wait, 0-120; if still running poll get_job.',0,120,default=30)
    if guard is None:guard='none' if kind=='query' or name in ('stop_all','cancel_job','stop_sequence') else 'forceable'
    if guard=='forceable':fields['force']=boo('Override active-sequence guard only when user explicitly authorizes it.',False)
    TOOLS[name]={'type':'function','function':{'name':name,'description':description,'parameters':schema(fields,required)},'title':title,'safety':{'kind':kind,'guard':guard,'requiresGatewayApproval':approval,'confirmUser':must_confirm},'source':REPO+source}

s='NINA.Mcp/Tools/Catalog/'
add('get_status','Get observatory status','Read real-time device connection and properties, current sequence and running jobs.',source=s+'StatusTools.cs')
add('get_recent_images','Get recent images','Read recent captured images and HFR/star statistics.',{'count':num('1-50 newest image records',1,50,integer=True,default=10)},source=s+'StatusTools.cs')
add('get_sequence','Get advanced sequence','Read the sequence tree and whether it is running.',{'max_depth':num('Sequence tree depth, 1-10',1,10,integer=True,default=4)},source=s+'SequenceTools.cs')
add('list_saved_sequences','List saved sequences','List saved sequence names in profile folder.',source=s+'SequenceTools.cs')
add('list_target_templates','List target templates','List available saved target templates for add_target.',source=s+'SequenceTools.cs')
add('get_job','Get background job','Read and optionally wait for a job, including success or failure.',{'job_id':txt('Job id, e.g. job-7f3a2c',64),'wait_seconds':num('Wait 0-120 seconds, default 0',0,120,default=0)},required=('job_id',),source=s+'JobTools.cs')
add('list_jobs','List background jobs','Read all running and recent background jobs.',source=s+'JobTools.cs')
add('stop_all','Emergency stop','Request sequence stop, cancel jobs, abort exposure, stop slew and focuser; then verify.',kind='stop',source=s+'StopTools.cs',guard='none')
add('cancel_job','Cancel background job','Request cancellation for one job.',{'job_id':txt('Job id',64),'wait_seconds':num('Wait 0-120 seconds, default 10',0,120,default=10)},required=('job_id',),kind='stop',source=s+'JobTools.cs',guard='none')
add('stop_sequence','Stop sequence','Request cancellation of active advanced sequence; does not park or warm camera.',kind='stop',source=s+'SequenceTools.cs',guard='none')
add('stop_guiding','Stop guiding','Stop guider while mount tracking remains on.',kind='stop',source=s+'GuiderTools.cs',approval=True)
add('connect_device','Connect device','Connect equipment slot selected in active PINS profile.',{'device':prop('string','Device slot',enum=DEVICES)},required=('device',),kind='hardware',job=True,source=s+'ConnectionTools.cs')
add('disconnect_device','Disconnect device','Disconnect one device; ensure camera warmed and mount parked when appropriate.',{'device':prop('string','Device slot',enum=DEVICES)},required=('device',),kind='hardware',job=True,source=s+'ConnectionTools.cs',approval=True)
add('slew_mount','Slew mount','Move mount to known J2000 coordinates; RA hours and Dec degrees.',{'ra':coords_ra(),'dec':coords_dec()},required=('ra','dec'),kind='hardware',job=True,source=s+'MountTools.cs',approval=True)
for name,title,desc in [('park_mount','Park mount','Park mount and stop tracking.'),('unpark_mount','Unpark mount','Unpark mount so slewing/tracking can start.'),('home_mount','Home mount','Find the physical home position using mount driver.')]:
    add(name,title,desc,kind='hardware',job=True,source=s+'MountTools.cs',approval=True)
add('set_tracking','Set mount tracking','Set sidereal tracking on or off.',{'enabled':boo('True to enable sidereal tracking, false to disable')},required=('enabled',),kind='hardware',source=s+'MountTools.cs',approval=True)
add('capture_image','Capture single image','Take one test exposure; save=false means analyzed but not saved.',{'exposure_seconds':num('Exposure seconds, 0.0001-3600',0.0001,3600),'filter':txt('Configured filter name',100),'gain':num('Camera driver gain',0,100000,integer=True),'offset':num('Camera driver offset',0,100000,integer=True),'binning':num('Symmetric binning 1-4',1,4,integer=True),'save':boo('Save frame, default false',False)},required=('exposure_seconds',),kind='hardware',job=True,source=s+'CameraTools.cs',approval=True)
add('cool_camera','Cool camera','Ramp camera cooler to requested sensor temperature.',{'target_celsius':num('Target sensor temperature °C',-50,30),'duration_minutes':num('Ramp time minutes 0-60',0,60,default=10)},required=('target_celsius',),kind='hardware',job=True,source=s+'CameraTools.cs',approval=True)
add('warm_camera','Warm camera','Ramp sensor to ambient before disconnect.',{'duration_minutes':num('Ramp time minutes 0-60',0,60,default=10)},kind='hardware',job=True,source=s+'CameraTools.cs',approval=True)
add('move_focuser','Move focuser','Move to absolute focuser position in steps.',{'position':num('Absolute focuser steps >= 0',0,integer=True)},required=('position',),kind='hardware',job=True,source=s+'FocusTools.cs',approval=True)
add('run_autofocus','Run autofocus','Run configured NINA autofocus routine; wait for job completion.',kind='hardware',job=True,source=s+'FocusTools.cs',approval=True)
add('change_filter','Change filter','Move filter wheel by either configured name OR zero-based position.',{'name':txt('Filter name',100),'position':num('Zero-based filter slot 0-100',0,100,integer=True)},kind='hardware',job=True,source=s+'FocusTools.cs',approval=True)
add('start_guiding','Start guiding','Start guiding; set force_calibration only if requested.',{'force_calibration':boo('True forces a fresh calibration',False)},kind='hardware',job=True,source=s+'GuiderTools.cs',approval=True)
add('set_switch','Set switch','Set connected power or heater switch to value within device-provided range.',{'index':num('Zero-based switch index',0,1000,integer=True),'value':num('Switch value: consult actual device range')},required=('index','value'),kind='hardware',job=True,source=s+'GuiderTools.cs',approval=True)
add('load_sequence','Load sequence','Load saved sequence (replaces current unsaved edits); does not start.',{'name':txt('Saved sequence name',200)},required=('name',),kind='change',source=s+'SequenceTools.cs',guard='strict',approval=True)
exposure=schema({'exposure_seconds':num('Exposure length seconds',0.0001,3600),'count':num('Number of exposures 1-1000',1,1000,integer=True),'filter':txt('Configured filter name',100),'gain':num('Gain',0,100000,integer=True),'offset':num('Offset',0,100000,integer=True),'binning':num('Binning 1-4',1,4,integer=True)},('exposure_seconds','count'))
add('add_target','Add target','Add target to current advanced sequence (no movement). Requires NAME and known J2000 RA/Dec; choose exactly one template or exposures. Never guess coordinates.',{'name':txt('Target name',100),'ra':coords_ra(),'dec':coords_dec(),'rotation_degrees':num('Sky position angle degrees 0<=x<360',0,exclusiveMaximum=360),'template':txt('Saved target template name',200),'exposures':{'type':'array','description':'Exposure plan, 1-20 blocks','items':exposure,'minItems':1,'maxItems':20},'slew_and_center':boo('Default true',True),'start_guiding':boo('Default true',True),'autofocus_at_start':boo('Default false',False),'dither_every':num('Dither every N frames; 0=never',0,100,integer=True,default=0)},required=('name','ra','dec'),kind='change',source=s+'SequenceTools.cs',approval=True)
add('clear_targets','Clear targets','Remove all targets from sequence; unsaved work is lost.',kind='change',source=s+'SequenceTools.cs',guard='strict',approval=True)
add('save_sequence','Save sequence','Save current advanced sequence to profile folder.',{'name':txt('File name without .json, max 100 chars',105),'overwrite':boo('True replaces an existing saved sequence',False)},required=('name',),kind='change',source=s+'SequenceTools.cs',approval=True)
add('start_sequence','Start sequence','Start loaded sequence after validation; skip_validation bypasses checks and should remain false.',{'skip_validation':boo('Bypass validation, not recommended',False)},kind='hardware',source=s+'SequenceTools.cs',guard='strict',approval=True)
add('reset_sequence','Reset sequence','Reset progress of sequence so it runs from beginning (only while idle).',kind='change',source=s+'SequenceTools.cs',guard='strict',approval=True)
assert len(TOOLS)==32, len(TOOLS)

# Some cross-field constraints are enforced by MCP code separately from JSON Schema.
CROSS_RULES={'add_target':'exactly one of template/exposures; template forbids slew_and_center/start_guiding/autofocus_at_start/dither_every',
             'change_filter':'exactly one of name/position', 'save_sequence':'filename safety and reserved suffix checks',
             'set_switch':'actual range comes from live device status',
             'start_sequence':'skip_validation must not be auto-set by the model'}
