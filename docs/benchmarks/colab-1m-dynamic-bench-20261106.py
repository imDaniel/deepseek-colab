#!/usr/bin/env python3
"""Same local-client workload on either engine; no credentials are written to output."""
import argparse, concurrent.futures, hashlib, http.client, json, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--cache',type=int,required=True);p.add_argument('--ubatch',type=int,required=True);p.add_argument('--dyn',type=int,required=True);p.add_argument('--draft-max',type=int,default=2);p.add_argument('--idx-chunk',type=int,default=0);p.add_argument('--mode',choices=('screen','baseline','long','validation'),default='screen');a=p.parse_args();a.port=18081;a.model='ds41'
key=Path('/content/secrets/key').read_text().strip()
prompts={
'prose':('Explain, in precise technical prose, how speculative decoding can increase language-model serving throughput while preserving the target model\'s output distribution. Discuss acceptance rate, verification cost, batching, and latency.'),
'code':('Write a complete Python implementation of an LRU cache using only the standard library. Include type hints, O(1) get and put methods, capacity validation, and a short executable example. Explain the complexity after the code.')}
def save():
 target=Path(a.out);tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(R,ensure_ascii=False,indent=2)+'\n');tmp.replace(target)
def request(prompt, max_tokens=256, effort='high', temperature=0, messages=None, keep_reasoning=False):
 body=json.dumps({'model':a.model,'messages':messages if messages is not None else [{'role':'user','content':prompt}],'max_tokens':max_tokens,'temperature':temperature,'top_p':1,'seed':42,**({'reasoning_effort':effort} if effort is not None else {}),'stream':True,'stream_options':{'include_usage':True}}).encode();start=time.monotonic();first=visible=last=None;usage={};timings={};done=False;error=None;finish=None;output=[];thoughts=[];digest=hashlib.sha256();status=None
 try:
  c=http.client.HTTPConnection('127.0.0.1',a.port,timeout=1800);c.request('POST','/v1/chat/completions',body,{'Authorization':'Bearer '+key,'Content-Type':'application/json'});r=c.getresponse();status=r.status
  if status!=200:error='HTTP %s: %s'%(status,r.read(500).decode(errors='replace'))
  else:
   while True:
    line=r.readline()
    if not line:break
    if not line.startswith(b'data: '):continue
    value=line[6:].strip()
    if value==b'[DONE]':done=True;break
    try:e=json.loads(value)
    except ValueError:continue
    if e.get('error'):error=str(e['error']);break
    usage=e.get('usage') or usage;timings=e.get('timings') or timings
    for ch in e.get('choices') or []:
     finish=ch.get('finish_reason') or finish;d=ch.get('delta') or {};t=d.get('reasoning_content') or d.get('reasoning') or '';v=d.get('content') or ''
     if t or v:
      now=time.monotonic();first=first or now;last=now
      if v and visible is None:visible=now
      digest.update((t+v).encode());output.append(v)
      if keep_reasoning:thoughts.append(t)
  c.close()
 except Exception as e:error=type(e).__name__+': '+str(e)[:200]
 end=time.monotonic();ct=usage.get('completion_tokens');elapsed=end-start
 return {'ok':status==200 and error is None and first is not None and done,'status':status,'error':error,'done':done,'finish':finish,'elapsed_s':round(elapsed,4),'ttft_s':round(first-start,4) if first else None,'first_visible_s':round(visible-start,4) if visible else None,'completion_tokens':ct,'prompt_tokens':usage.get('prompt_tokens'),'cached_tokens':(usage.get('prompt_tokens_details') or {}).get('cached_tokens'),'e2e_tps':round(ct/elapsed,3) if ct else None,'decode_tps':round((ct-1)/(last-first),3) if ct and ct>1 and first and last and last>first else None,'timings':timings,'output_sha256':digest.hexdigest()[:16],'visible_chars':sum(map(len,output)),'visible_output':''.join(output)[:20000],**({'_reasoning_content':''.join(thoughts)} if keep_reasoning else {})}

import threading,subprocess,base64,struct,zlib,os
phase='init';resource_stop=threading.Event()
resource_path=Path('/content/res/tune1mdynamic-resource.jsonl');resource_path.write_text('')
def resources():
 while not resource_stop.is_set():
  try:
   m=Path('/proc/meminfo').read_text().splitlines();avail=int(next(x for x in m if x.startswith('MemAvailable:')).split()[1]);psi=Path('/proc/pressure/memory').read_text().splitlines();gpu_raw=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,clocks.sm,power.draw,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5).stdout.strip().split(',');gpu=', '.join(x.strip() for x in gpu_raw[:2])
   state={k:(float(x.strip()) if x.strip().replace('.','',1).isdigit() else None) for k,x in zip(('sm_clock_mhz','power_w','temp_c','util_pct'),gpu_raw[2:])}
   v={'at':time.time(),'phase':phase,'available_kib':avail,'pressure':psi,'gpu':gpu,'gpu_state':state}
   with resource_path.open('a') as f:f.write(json.dumps(v)+'\n')
  except Exception:pass
  resource_stop.wait(5)
resource_thread=threading.Thread(target=resources,daemon=True);resource_thread.start()
R={'context_per_slot':1048576,'slots':1,'engine':'deepseek-colab b11963 cached build','model':'ISTA IQ2_XXS + AtomicChat','images':True,'draft':True,'draft_max':a.draft_max,'cache_mib':a.cache,'ubatch':a.ubatch,'dyn':a.dyn,'idx_chunk_mib':a.idx_chunk,'mode':a.mode,'transport':'Colab loopback HTTP','benchmark_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'temperature':0,'top_p':1,'seed':42,'short':[],'prefill':[],'full':[],'long_decode':[],'concurrent':[],'complete':False}
def mark(label):
 global phase
 phase=label;R['phase']=label;save();print('START',label,flush=True)
def record(v,label):
 print('DONE',label,'ok',v['ok'],'tokens',v['completion_tokens'],'input',v['prompt_tokens'],'cache',v['cached_tokens'],'finish',v['finish'],'visible',v['visible_chars'],'decode',v['timings'].get('predicted_per_second'),'ttft',v['ttft_s'],'hash',v['output_sha256'],flush=True)
 if not v['ok']:raise RuntimeError(label+' failed')
try:
 line=next(x for x in Path('/content/res/timeline.txt').read_text().splitlines() if 'boot start ' in x)
 assert all(x in line.split() for x in ('ctx=1048576','slots=1','ubatch='+str(a.ubatch),'cache='+str(a.cache),'dyn='+str(a.dyn),'images=1','draft=1','uncensored=1')),'boot mismatch'
 import shlex
 argv=shlex.split(Path('/content/res/serve_cmd.txt').read_text());assert argv[argv.index('-np')+1]=='1' and argv[argv.index('-c')+1]=='1048576'
 assert argv[argv.index('--spec-draft-n-max')+1]=='2'
 pid=Path('/content/res/server.pid').read_text().strip();env=dict(x.split(b'=',1) for x in (Path('/proc')/pid/'environ').read_bytes().split(b'\0') if b'=' in x)
 assert int(env.get(b'LLAMA_DSV41_IDX_CHUNK_MIB',b'0'))==a.idx_chunk
 assert env.get(b'LLAMA_MOE_DEVCACHE_DMA')==b'0'
 R['boot_verified']=line;R['reasoning_effort']='max';R['short_budget']=512
 R['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip();print('BOOT_VERIFIED',line,'idx',a.idx_chunk,'DMA0','draft2',flush=True)
 def image(size):
  label='image-'+str(size);mark(label)
  def ch(k,d):return struct.pack('!I',len(d))+k+d+struct.pack('!I',zlib.crc32(k+d))
  png=b'\x89PNG\r\n\x1a\n'+ch(b'IHDR',struct.pack('!IIBBBBB',size,size,8,2,0,0,0))+ch(b'IDAT',zlib.compress((b'\x00'+b'\xff\x00\x00'*size)*size))+ch(b'IEND',b'')
  m=[{'role':'user','content':[{'type':'text','text':'What is the dominant color? Answer in one word.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}}]}]
  v=request('',512,'low',messages=m);R[label]=v;save();record(v,label);assert 'red' in v['visible_output'].lower()
 image(1024)
 if a.mode in ('baseline','validation'):image(4096)
 for repeat in range(3):
  for kind in ('code','prose'):
   label='short-'+kind+'-'+str(repeat+1);mark(label);v=request(prompts[kind],512,'max');v.update(kind=kind,repeat=repeat+1);R['short'].append(v);save();record(v,label)
 for target in (32768,131072):
  label='prefill-'+str(target);mark(label);v=request('Benchmark 1M dynamic screen-'+str(target)+'. Read the repeated filler: '+(' a'*target)+'\nEnd of filler. What is 9 times 7? Reply with only the number.',128,'low');v['target']=target;R['prefill'].append(v);save();record(v,label);assert v['visible_output'].strip()=='63'
 for repeat in range(3 if a.mode in ('baseline','validation') else 1):
  label='full-short-code-temperature0.6-'+str(repeat+1);mark(label);v=request(prompts['code'],32768,'max',temperature=0.6);v.update(kind='code',scope='short',temperature=0.6,repeat=repeat+1,max_output_tokens=32768);R['full'].append(v);save();record(v,label)
  if v['finish']!='stop' or not v['visible_chars']:break
 mark('full-short-code-temperature0');v=request(prompts['code'],32768,'max');v.update(kind='code',scope='short',temperature=0,max_output_tokens=32768);R['full'].append(v);save();record(v,'full-short-code-temperature0')
 if a.mode in ('baseline','validation','long'):
  initial='Benchmark fixed-million-records-v2. Read the repeated filler and records: '+(' a'*100000)+'\nRecord kiwi code: 482917.\n'+(' a'*400000)+'\nRecord cobalt code: 691438.\n'+(' a'*400000)+'\nRecord maple code: 275604.\n'+(' a'*100000)+'\nEnd of filler. '+prompts['code']
  for repeat in range(2):
   label='filledmillion-complete-code-temperature0.6-'+str(repeat+1);mark(label);v=request(initial,32768,'max',temperature=0.6);v.update(kind='code',scope='filledmillion',temperature=0.6,repeat=repeat+1,max_output_tokens=32768);R['full'].append(v);save();record(v,label)
  if a.mode in ('baseline','validation'):
   # Keep the same effort prefix; use the model tokenizer to reserve a 4K retrieval budget.
   # This checks near-full capacity and ordinary Chat history without another redundant cold prefill.
   mark('near-full-appended-1m-retrieval')
   previous=next(x for x in reversed(R['full']) if x.get('scope')=='filledmillion')
   assert previous['finish']=='stop' and previous['visible_chars'],'long code must finish before append test'
   c=http.client.HTTPConnection('127.0.0.1',a.port,timeout=20);c.request('POST','/tokenize',json.dumps({'content':previous['visible_output'],'add_special':False}).encode(),{'Authorization':'Bearer '+key,'Content-Type':'application/json'});tr=c.getresponse();token_data=json.loads(tr.read());c.close();assert tr.status==200 and isinstance(token_data.get('tokens'),list)
   padding=1048576-4096-previous['prompt_tokens']-len(token_data['tokens'])-256;assert padding>0;R['append_padding_tokens']=padding
   messages=[{'role':'user','content':initial},{'role':'assistant','content':previous['visible_output']},{'role':'user','content':(' a'*padding)+'\nEnd of appended filler. Return only a JSON object with the earlier kiwi, cobalt, maple codes and math equal to 9 times 7.'}]
   v=request('',4096,'max',messages=messages);v['scope']='near-full-appended';v['max_output_tokens']=4096
   import re
   try:answer=json.loads(re.search(r'\{.*?\}',v['visible_output'],re.S).group(0));R['near_full_retrieval_pass']=all(str(answer.get(k))==str(value) for k,value in {'kiwi':482917,'cobalt':691438,'maple':275604,'math':63}.items())
   except (ValueError,AttributeError):R['near_full_retrieval_pass']=False
   R['near_full']=[v];save();record(v,'near-full-appended');assert v['prompt_tokens']>=1043000 and v['prompt_tokens']+4096<=1048576;assert v['cached_tokens']>=999000,'million-prefix reuse failed';print('NEAR_FULL_CORRECT',R['near_full_retrieval_pass'],flush=True);assert R['near_full_retrieval_pass']
  for repeat in range(3):
   label='filled-code-max512-'+str(repeat+1);mark(label);v=request(initial,512,'max');v.update(kind='code',repeat=repeat+1);R['long_decode'].append(v);save();record(v,label)
 if a.mode in ('baseline','validation'):
  for repeat in range(2):
   for kind in ('code','prose'):
    label='c4-'+kind+'-'+str(repeat+1);mark(label);start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lambda _:request(prompts[kind],1024,'max'),range(4)))
    elapsed=time.monotonic()-start;v={'kind':kind,'repeat':repeat+1,'requests':rows,'elapsed_s':elapsed,'aggregate_tps':sum(x['completion_tokens'] or 0 for x in rows)/elapsed};R['concurrent'].append(v);save();print('DONE',label,'aggregate',v['aggregate_tps'],flush=True);assert all(x['ok'] for x in rows)
  image(1024)
 resource_stop.set();resource_thread.join(6);rows=[json.loads(x) for x in resource_path.read_text().splitlines()]
 R['resource_summary']={'samples':len(rows),'min_available_kib':min(x['available_kib'] for x in rows),'min_gpu_free_mib':min(int(x['gpu'].split(',')[1]) for x in rows),'max_full_psi_avg60':max(float(next(v for v in x['pressure'] if v.startswith('full ')).split('avg60=')[1].split()[0]) for x in rows)}
 R['complete']=True;R['phase']='complete';save();print('BENCH_COMPLETE',flush=True)
except BaseException as e:
 import re
 R['error']=type(e).__name__+': '+str(e)
 try:R['server_error_lines']=[x[:500] for x in Path('/content/res/server.log').read_text(errors='replace').splitlines() if re.search(r'out of memory|CUDA error|failed|error:',x,re.I)][-16:]
 except OSError:pass
 save();raise
finally:resource_stop.set()
