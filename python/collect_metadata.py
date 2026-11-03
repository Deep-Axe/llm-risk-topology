"""Checkpointed metadata crawl. New: --limit 50000 --output PATH; resume: --resume --output PATH."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse
import requests
from huggingface_hub import get_token

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = 'https://huggingface.co/api/models'

def now():
    return datetime.now(timezone.utc).isoformat()

def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def atomic_json(path, value):
    temporary = path.with_name(path.name+'.partial')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)

@contextmanager
def collection_lock(output):
    # OS lock releases on process death; leftover filenames do not block resume.
    with (output/'.collector.lock').open('a+b') as stream:
        if stream.tell() == 0: stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally:
            stream.seek(0)
            if os.name == 'nt': msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(stream, fcntl.LOCK_UN)

def valid_endpoint(url):
    p = urlparse(url)
    return p.scheme == 'https' and p.hostname == 'huggingface.co' and p.path == '/api/models'

def request_page(url, params):
    if not valid_endpoint(url): raise ValueError('Unexpected pagination endpoint')
    headers = {'User-Agent': 'llm-risk-topology-checkpointed-crawl/1'}
    if get_token(): headers['Authorization'] = 'Bearer '+get_token()
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=(10,45))
            if r.status_code in [429,500,502,503,504] and attempt < 4:
                time.sleep(min(15,2**attempt)); continue
            r.raise_for_status()
            rows, next_url = r.json(), r.links.get('next',{}).get('url')
            if next_url and not valid_endpoint(next_url): raise ValueError('Unexpected pagination endpoint')
            if not isinstance(rows,list) or any(not isinstance(row,dict) or not isinstance(row.get('id'),str) for row in rows):
                raise ValueError('Invalid API rows')
            return rows,next_url
        except requests.RequestException:
            if attempt == 4: raise RuntimeError('Request failed after retries; resume the checkpoint') from None
            time.sleep(min(15,2**attempt))

def initialize(output, limit):
    query = {'limit': min(100,limit),'filter':'text-generation','sort':'downloads','direction':-1,'full':'true','cardData':'true'}
    state = {'checkpoint_version':1,'status':'collecting','started_at':now(),'requested_records':limit,
        'query':query,'next_url':ENDPOINT,'next_params':query,'pages':[],'unique_records':0,'resume_count':0}
    (output/'pages').mkdir()
    atomic_json(output/'collection_checkpoint.json',state)
    return state

def load_checkpoint(output, limit=None):
    if (output/'metadata_snapshot.json').exists(): raise ValueError('Already frozen; choose a new output directory')
    state = json.loads((output/'collection_checkpoint.json').read_text(encoding='utf-8'))
    if state['checkpoint_version'] != 1 or state['status'] not in ['collecting','finalizing']: raise ValueError('Invalid checkpoint status')
    if limit is not None and limit != state['requested_records']: raise ValueError('Cannot change resumed target')
    seen,url,params = set(),ENDPOINT,state['query']
    for index,entry in enumerate(state['pages']):
        if entry['path'] != f'pages/{index:06d}.json' or file_hash(output/entry['path']) != entry['sha256']: raise ValueError('Page checksum/order mismatch')
        page = json.loads((output/entry['path']).read_text(encoding='utf-8'))
        if page['request_url'] != url or page['request_params'] != params: raise ValueError('Broken pagination chain')
        for row in page['rows']:
            seen.add(row['id'])
            if len(seen) >= state['requested_records']: break
        url,params = page['next_url'],None
    if state['unique_records'] != len(seen) or state['next_url'] != url or state['next_params'] != params: raise ValueError('Checkpoint/page mismatch')
    return state,seen

def append_page(output,state,seen,fetch=request_page):
    index = len(state['pages']); path = output/'pages'/f'{index:06d}.json'
    if path.exists():
        # Adopt a fully written page orphaned by a crash before checkpoint commit.
        page = json.loads(path.read_text(encoding='utf-8'))
        if page['request_url'] != state['next_url'] or page['request_params'] != state['next_params']: raise ValueError('Orphan cursor mismatch')
    else:
        rows,next_url = fetch(state['next_url'],state['next_params'])
        page = {'request_url':state['next_url'],'request_params':state['next_params'],'received_at':now(),'next_url':next_url,'rows':rows}
        atomic_json(path,page)
    if not isinstance(page['rows'],list) or any(not isinstance(row,dict) or not isinstance(row.get('id'),str) for row in page['rows']): raise ValueError('Invalid page rows')
    for row in page['rows']:
        seen.add(row['id'])
        if len(seen) >= state['requested_records']: break
    state['pages'].append({'path':f'pages/{index:06d}.json','sha256':file_hash(path),'response_records':len(page['rows'])})
    state.update(unique_records=len(seen),next_url=page['next_url'],next_params=None,updated_at=now())
    atomic_json(output/'collection_checkpoint.json',state)

def finalize(output,state):
    if (output/'metadata_snapshot.json').exists(): raise ValueError('Refusing to overwrite frozen snapshot')
    if state['unique_records'] != state['requested_records']: raise ValueError('Incomplete collection cannot freeze')
    state.update(status='finalizing'); atomic_json(output/'collection_checkpoint.json',state)
    seen = set()
    raw,tsv_path = output/'models.jsonl.gz.partial',output/'models.tsv.partial'
    with raw.open('wb') as binary, gzip.GzipFile(filename='models.jsonl',fileobj=binary,mode='wb',mtime=0) as zipped, tsv_path.open('w',encoding='utf-8',newline='\n') as tsv:
        for entry in state['pages']:
            if file_hash(output/entry['path']) != entry['sha256']: raise ValueError('Page changed before freezing')
            page = json.loads((output/entry['path']).read_text(encoding='utf-8'))
            for row in page['rows']:
                if row['id'] in seen or len(seen) >= state['requested_records']: continue
                seen.add(row['id'])
                zipped.write((json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n').encode('utf-8'))
                tsv.write('\t'.join([row['id'],row.get('sha') or '-',json.dumps(row.get('cardData') or {},ensure_ascii=False,separators=(',',':'))])+'\n')
    if len(seen) != state['requested_records']: raise ValueError('Final count mismatch')
    os.replace(raw,output/'models.jsonl.gz'); os.replace(tsv_path,output/'models.tsv')
    manifest = {'schema_version':'1.0.0','status':'frozen','started_at':state['started_at'],'finished_at':now(),'endpoint':ENDPOINT,
        'query':state['query'],'requested_records':state['requested_records'],'record_count':len(seen),
        'collection':{'checkpoint_version':1,'committed_pages':len(state['pages']),'resume_count':state['resume_count'],
                      'journal_sha256':hashlib.sha256(json.dumps(state['pages'],sort_keys=True).encode()).hexdigest()},
        'selection_note':'Top-download text-generation development slice, not representative. Pagination, especially after resume, is not atomic. Saved page responses are authoritative observations.',
        'artifacts':[{'path':name,'sha256':file_hash(output/name),'bytes':(output/name).stat().st_size} for name in ['models.jsonl.gz','models.tsv']]}
    # The snapshot manifest is published LAST: incomplete folders are not pipeline inputs.
    atomic_json(output/'metadata_snapshot.json',manifest)
    state.update(status='frozen'); atomic_json(output/'collection_checkpoint.json',state)
    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-pages',type=int,help='Stop safely at a checkpoint without publishing a snapshot')
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 50000: parser.error('limit must be 1..50000')
    if args.stop_after_pages is not None and args.stop_after_pages < 1: parser.error('stop-after-pages must be positive')
    if args.resume and args.output is None: parser.error('resume requires output')
    output = args.output or ROOT/'data/raw/metadata'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if not args.resume: output.mkdir(parents=True,exist_ok=False)
    with collection_lock(output):
        if args.resume:
            state,seen = load_checkpoint(output,args.limit)
            state['resume_count'] += 1; atomic_json(output/'collection_checkpoint.json',state)
            print(f'Resuming {len(seen)} committed records in {len(state["pages"])} pages',flush=True)
        else: state,seen = initialize(output,args.limit or 1000),set()
        count = 0
        while len(seen) < state['requested_records']:
            if not state['next_url']: raise ValueError('API exhausted; incomplete checkpoint retained')
            append_page(output,state,seen); count += 1
            if len(state['pages']) % 10 == 0 or count == 1: print(f'Committed {len(seen)}/{state["requested_records"]} unique records ({len(state["pages"])} pages)',flush=True)
            if args.stop_after_pages and count >= args.stop_after_pages:
                print('Stopped safely; no frozen snapshot published.',flush=True); return
        finalize(output,state)
        print(f'Snapshot: {output}\nSnapshot ID: {file_hash(output/"metadata_snapshot.json")}',flush=True)

if __name__ == '__main__': main()
