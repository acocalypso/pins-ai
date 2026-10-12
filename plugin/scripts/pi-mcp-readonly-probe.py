"""Conservative Pi LoRA diagnostic with an enforced get_status-only MCP gateway.

The existing PINS plugin owns llama.cpp; no AI daemon or MCP server is created.
This trusted SSH diagnostic accesses its private raw completion endpoint. The
public informational chat API is unchanged and is not a production MCP chat loop.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def approve_readonly(call, exposed):
    # Server read-only mode also exposes stop/cancel actions. Model text is never authority.
    if call != {'name': 'get_status', 'arguments': {}} or 'get_status' not in exposed:
        raise ValueError('Probe permits only the exposed get_status tool with empty arguments')
    return {'name': 'get_status', 'arguments': {}}


REMOTE = r'''
import json,pathlib,subprocess,urllib.request,threading,time
home=pathlib.Path.home();root=home/'.local/share/pins-ai'
item=PAYLOAD
def api(path,post=False):
 key=(root/'api.key').read_text().strip()
 config=json.loads((root/'config.json').read_text())
 req=urllib.request.Request('http://127.0.0.1:'+str(config['apiPort'])+'/'+path,
  data=b'' if post else None,headers={'Authorization':'Bearer '+key})
 with urllib.request.urlopen(req,timeout=35) as r:return json.load(r)
def temp():return int(pathlib.Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000
if item['action']=='api':
 if item['path'] not in ('load','unload','status','diagnostics'):raise ValueError('Forbidden plugin operation')
 result=api(item['path'],item.get('post',False))
elif item['action'] in ('mcp','close-session'):
 if item['action']=='close-session':item['body']={'method':'DELETE'}
 body=item['body'];method=body['method']
 if method not in ('initialize','notifications/initialized','tools/list','tools/call','DELETE'):raise ValueError('Forbidden RPC method')
 if method=='tools/call' and body['params']!={'name':'get_status','arguments':{}}:raise ValueError('Forbidden tool call')
 env=dict(line.split('=',1) for line in (home/'.config/pins/mcp.env').read_text().splitlines() if '=' in line)
 headers={'Authorization':'Bearer '+env['PINS_MCP_TOKEN'].strip().strip('"'),
  'Content-Type':'application/json','Accept':'application/json, text/event-stream','MCP-Protocol-Version':'2025-11-25'}
 if item.get('session'):headers['Mcp-Session-Id']=item['session']
 req=urllib.request.Request('http://127.0.0.1:4782/mcp',data=None if method=='DELETE' else json.dumps(body).encode(),
                           method='DELETE' if method=='DELETE' else 'POST',headers=headers)
 with urllib.request.urlopen(req,timeout=10) as r:
  raw=r.read();result={'body':json.loads(raw) if raw else {},'session':r.headers.get('Mcp-Session-Id'),'httpStatus':r.status}
elif item['action']=='completion':
 status=api('status');config=json.loads((root/'config.json').read_text())
 if status['modelId']!='qwen3-checkpoint700-lora-diagnostic':raise ValueError('Unexpected model profile')
 pid=status['processId']
 if not pid or status['state']!='ready':raise ValueError('Model is not ready')
 parent=int(next(line.split()[1] for line in pathlib.Path('/proc/'+str(pid)+'/status').read_text().splitlines() if line.startswith('PPid:')))
 pins=int(subprocess.check_output(['systemctl','show','pins.service','-p','MainPID','--value'],text=True))
 if parent!=pins:raise ValueError('Model child is not owned by PINS')
 args=pathlib.Path('/proc/'+str(pid)+'/cmdline').read_bytes().decode().split('\0')
 if args[args.index('--lora')+1]!=config['loraPath']:raise ValueError('Unexpected child adapter')
 port=int(args[args.index('--port')+1]);keypath=pathlib.Path(args[args.index('--api-key-file')+1])
 if keypath.parent!=root:raise ValueError('Unexpected private runtime key path')
 before=temp()
 if before>=70:raise ValueError('Pi is too warm for a conservative probe')
 peak=[before];done=threading.Event()
 def monitor():
  while not done.wait(.1):peak[0]=max(peak[0],temp())
 thread=threading.Thread(target=monitor,daemon=True);thread.start()
 try:
  payload={'prompt':item['prompt'],'n_predict':48,'temperature':0,'seed':3407,'cache_prompt':False,'stream':False,'stop':[item['eos']]}
  request=urllib.request.Request('http://127.0.0.1:'+str(port)+'/completion',data=json.dumps(payload).encode(),
   headers={'Authorization':'Bearer '+keypath.read_text().strip(),'Content-Type':'application/json'})
  started=time.monotonic()
  with urllib.request.urlopen(request,timeout=30) as response:result=json.load(response)
  result.update(wallSeconds=time.monotonic()-started,temperatureBeforeC=before,temperatureAfterC=temp(),
   peakTemperatureC=max(peak[0],temp()),modelPid=pid,pinsPid=pins,pluginOwned=True,adapterVerifiedInChildArgs=True)
 finally:done.set();thread.join(timeout=1)
else:raise ValueError('Unknown action')
print(json.dumps(result))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='192.168.178.109')
    parser.add_argument('--user',default='pi')
    parser.add_argument('--known-hosts',type=Path,default=ROOT/'.local/known_hosts')
    parser.add_argument('--render-python',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new report filename')
    if not os.environ.get('PINS_SSH_PASSWORD'):parser.error('Set PINS_SSH_PASSWORD in your local environment')
    import paramiko
    client=paramiko.SSHClient();client.load_host_keys(str(args.known_hosts));client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(args.host,username=args.user,password=os.environ['PINS_SSH_PASSWORD'],timeout=15)
    worker=subprocess.Popen([str(args.render_python.resolve()),'-u','training/render_inference_prompt.py',
       '--checkpoint',str(args.checkpoint.resolve())],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf-8')
    report={'startedAt':datetime.now(timezone.utc).isoformat(),'host':args.host,'checkpoint':str(args.checkpoint),
            'productionChatBinding':False,'allowedExecutedTools':['get_status'],'hardwareCommandsSent':False,'results':[]}
    def persist():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        temporary=args.output.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8');temporary.replace(args.output)
    def remote(item):
        code=REMOTE.replace('item=PAYLOAD','item=json.loads('+repr(json.dumps(item))+')')
        inp,out,err=client.exec_command('python3 -',timeout=40);inp.write(code);inp.channel.shutdown_write()
        raw=out.read().decode();errors=err.read().decode();status=out.channel.recv_exit_status()
        if status:raise RuntimeError(errors)
        return json.loads(raw)
    def local(item):
        worker.stdin.write(json.dumps(item)+'\n');worker.stdin.flush()
        raw=worker.stdout.readline()
        if not raw:raise RuntimeError('Tokenizer worker exited')
        result=json.loads(raw)
        if 'error' in result:raise ValueError(result['error'])
        return result
    session=None;rid=0
    def rpc(method,params=None,notification=False):
        nonlocal rid,session
        body={'jsonrpc':'2.0','method':method}
        if not notification:rid+=1;body['id']=rid
        if params is not None:body['params']=params
        result=remote({'action':'mcp','body':body,'session':session})
        if result['session']:session=result['session']
        if 'error' in result['body']:raise RuntimeError(result['body']['error'])
        return result['body'].get('result')
    try:
        report['serverInfo']=rpc('initialize',{'protocolVersion':'2025-11-25','capabilities':{},
                   'clientInfo':{'name':'pins-checkpoint700-readonly-probe','version':'1'}})['serverInfo']
        rpc('notifications/initialized',notification=True)
        catalog=rpc('tools/list')['tools'];report['liveTools']=catalog
        exposed={t['name']:t for t in catalog}
        probe=exposed['get_status']
        tools=[{'type':'function','function':{'name':probe['name'],'description':probe['description'].split('. ')[0][:180],
                                           'parameters':probe['inputSchema']}}]
        state=rpc('tools/call',approve_readonly({'name':'get_status','arguments':{}},exposed))
        payload=state.get('structuredContent') or json.loads(next(c['text'] for c in state['content'] if c['type']=='text'))
        if payload['sequence']['running']:raise RuntimeError('An imaging sequence is active; no inference probe started')
        report['initialStatus']=state
        report['modelStatus']=remote({'action':'api','path':'load','post':True});persist()
        policy=local({'action':'policy'})['policy']
        for index,(locale,prompt) in enumerate([('en','Check the current PINS status.'),('de','Prüfe den aktuellen PINS-Status.')]):
            if index:time.sleep(30)
            messages=[{'role':'system','content':policy},{'role':'user','content':prompt}]
            rendered=local({'action':'render','messages':messages,'tools':tools,'maxNewTokens':48})
            generated=remote({'action':'completion',**rendered})
            row={'locale':locale,'prompt':prompt,'inputTokens':rendered['inputTokens'],'generation':generated,'toolExecuted':False}
            try:
                calls=local({'action':'parse','text':generated['content'],'tools':tools})['calls']
                if len(calls)!=1:raise ValueError('Expected exactly one get_status proposal')
                proposal=approve_readonly(calls[0],exposed)
                result=rpc('tools/call',proposal)
                if result.get('isError'):raise RuntimeError('Read-only MCP call failed')
                row.update(proposal=proposal,toolExecuted=True,mcpResult=result)
                if locale=='en':
                    payload=result.get('structuredContent') or json.loads(next(c['text'] for c in result['content'] if c['type']=='text'))
                    history=[*messages,{'role':'assistant','content':'','tool_calls':[{'type':'function','function':proposal}]},
                             {'role':'tool','name':'get_status','content':json.dumps(payload,separators=(',',':'))}]
                    feedback=local({'action':'render','messages':history,'tools':tools,'maxNewTokens':48})
                    row['followup']=remote({'action':'completion',**feedback})
            except (ValueError,RuntimeError) as error:row['proposalOrFeedbackError']=str(error)
            report['results'].append(row);persist()
            print(json.dumps({'locale':locale,'toolExecuted':row['toolExecuted'],'raw':generated['content'],
                              'peakTemperatureC':generated['peakTemperatureC']},ensure_ascii=True),flush=True)
        report['completed']=True
    except BaseException as error:
        report.update(completed=False,error=type(error).__name__+': '+str(error));raise
    finally:
        try:report['finalModelStatus']=remote({'action':'api','path':'unload','post':True})
        except Exception as error:report['unloadError']=str(error)
        if session:
            try:report['sessionClosed']=remote({'action':'close-session','session':session})['httpStatus']
            except Exception as error:report['sessionCloseError']=str(error)
        worker.stdin.close()
        try:worker.wait(timeout=10)
        except subprocess.TimeoutExpired:worker.kill();worker.wait()
        client.close();report['finishedAt']=datetime.now(timezone.utc).isoformat();persist()


if __name__=='__main__':main()
