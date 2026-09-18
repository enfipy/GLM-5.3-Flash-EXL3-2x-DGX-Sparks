#!/usr/bin/env python3
import pathlib,time,subprocess
root=pathlib.Path(__file__).resolve().parent
deadline=time.monotonic()+1800
while time.monotonic()<deadline:
    values={line.split(':')[0]:int(line.split()[1]) for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:', 'SwapFree:'))}
    available=values['MemAvailable']/1048576
    print(time.strftime('%FT%T'),f'head_available_gib={available:.2f}',flush=True)
    if available<6:
        print('LOW MEMORY: stopping glm53',flush=True)
        subprocess.run([str(root/'model.sh'),'stop'],timeout=90)
        raise SystemExit(2)
    time.sleep(1)
print('Monitoring interval complete',flush=True)
