import json, subprocess, sys, time, urllib.request
URL = "http://100.76.243.97:8888"
def metrics():
    t = urllib.request.urlopen(URL + "/metrics", timeout=5).read().decode()
    out = {}
    for line in t.splitlines():
        for k in ("spec_decode_num_drafts_total", "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total", "generation_tokens_total"):
            if line.startswith("vllm:" + k + "{"):
                out[k] = float(line.rsplit(" ", 1)[1])
    return out
for r in range(1, int(sys.argv[1]) + 1):
    b = metrics(); t0 = time.strftime("%H:%M:%S", time.gmtime())
    res = subprocess.run([sys.executable, "decode_bench.py", "--url", URL, "--concurrency", "1", "--repeats", "1"], capture_output=True, text=True).stdout
    tps = [l for l in res.splitlines() if l.startswith("x1")]
    t1 = time.strftime("%H:%M:%S", time.gmtime()); a = metrics()
    d = {k: a[k] - b[k] for k in b}
    steps = d["spec_decode_num_drafts_total"]; acc = d["spec_decode_num_accepted_tokens_total"] / steps if steps else 0
    tok = d["generation_tokens_total"]
    print(f"run{r} {t0}-{t1} {tps[0][4:60] if tps else res[-80:]} | steps={steps:.0f} accepted/step={acc:.2f} gen_tok={tok:.0f} tok/step={(tok/steps if steps else 0):.2f}", flush=True)
