#!/usr/bin/env python3
"""Reject obvious secrets and host/runtime files in staged or tracked content."""
import argparse,re,subprocess,sys
from pathlib import PurePosixPath
p=argparse.ArgumentParser();p.add_argument('--staged',action='store_true');p.add_argument('--tracked',action='store_true');a=p.parse_args()
mode=['diff','--cached','--name-only','--diff-filter=ACMR','-z'] if a.staged else ['ls-files','-z']
files=subprocess.check_output(['git',*mode]).decode().split('\0');bad=[]
patterns=[rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'AKIA[0-9A-Z]{16}',rb'AIza[0-9A-Za-z_-]{30,}',rb'gh[pousr]_[A-Za-z0-9]{25,}']
for name in filter(None,files):
 path=PurePosixPath(name)
 if set(path.parts)&{'.private','.local','.venv','.runtime','__pycache__','sync-outbox'} or (path.parts[0] in ('handoff','imports')) or 'data' in path.parts or 'evidence' in path.parts:
  bad.append((name,'host/runtime path'));continue
 if path.name.startswith('.env') and path.name!='.env.example':bad.append((name,'environment file'));continue
 raw=subprocess.check_output(['git','show',':'+name]) if a.staged else open(name,'rb').read()
 if any(re.search(pattern,raw) for pattern in patterns):bad.append((name,'credential pattern'))
 # Reject literal non-placeholder credentials in JSON; source variable names are fine.
 if path.suffix=='.json':
  import json
  try: value=json.loads(raw)
  except (ValueError,UnicodeDecodeError):continue
  def scan(v):
   if isinstance(v,dict):
    for key,item in v.items():
     if key.lower() in ('api_key','api_key_id','secret','password','access_token','refresh_token') and isinstance(item,str) and item and not item.startswith(('REPLACE','PROVISION','YOUR_','<')):
      bad.append((name,'non-placeholder credential field'))
     scan(item)
   elif isinstance(v,list):
    for item in v:scan(item)
  scan(value)
if bad:
 for name,reason in bad:print(f'BLOCKED {name}: {reason}',file=sys.stderr)
 raise SystemExit(1)
print(f'File/credential checks passed ({sum(bool(x) for x in files)} files). Review still required.')
