"""Sample Linux memory and Java RSS; observed peaks include existing stack services."""
import argparse
import json
from pathlib import Path
import signal
import time

stop = False
def finish(*_):
    global stop
    stop = True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM,finish)
    signal.signal(signal.SIGINT,finish)
    samples = []
    while not stop:
        memory = {line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        java_rss = 0
        for process in Path('/proc').glob('[0-9]*'):
            try:
                if (process/'comm').read_text().strip() == 'java':
                    status = (process/'status').read_text().splitlines()
                    java_rss += next(int(s.split()[1]) for s in status if s.startswith('VmRSS:'))
            except (OSError,StopIteration): pass
        samples.append({'unix_time':time.time(),'available_kib':memory['MemAvailable'],
            'used_kib':memory['MemTotal']-memory['MemAvailable'],'swap_used_kib':memory['SwapTotal']-memory['SwapFree'], 'all_java_rss_kib':java_rss})
        args.output.write_text(json.dumps({'sampling_seconds':2,'scope':'WSL system and all Java processes, including pre-existing services; not isolated job memory',
            'samples':len(samples),'baseline':samples[0], 'minimum_available_kib':min(s['available_kib'] for s in samples),
            'maximum_used_kib':max(s['used_kib'] for s in samples),'maximum_java_rss_kib':max(s['all_java_rss_kib'] for s in samples),
            'maximum_swap_used_kib':max(s['swap_used_kib'] for s in samples)},indent=2)+'\n')
        time.sleep(2)

if __name__ == '__main__': main()
