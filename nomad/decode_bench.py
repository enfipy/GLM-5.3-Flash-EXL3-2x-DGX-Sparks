#!/usr/bin/env python3
"""sparkDash-equivalent decode benchmark (server/collectors/DecodeBench.js protocol).

Per stream: chat completion, stream=true with usage, temperature 0, top_p 1,
thinking off, max_tokens=min_tokens=N, ignore_eos. Warmup 32 tokens first.
Stream tok/s = (completion_tokens - 1) / (last_token_time - first_token_time).
Aggregate tok/s = total decode tokens / (max(last) - min(first)) over the wave.
"""
import argparse, json, statistics, sys, threading, time, urllib.request

PROSE = ("Write a detailed step-by-step explanation of how a hash map works, "
         "including collision handling, resizing, and time complexity. Be thorough.")
TEXT_PROMPTS = [
  "Write a clear essay explaining unified memory on NVIDIA GB10 Sparks for a technical but non-specialist reader. Keep expanding with examples and analogies.",
  "Write a vivid sci-fi scene set in a liquid-cooled server room at 3 a.m. Keep expanding the scene with sensory detail and dialogue.",
  "Write a pirate-captain monologue explaining KV-cache pressure and prefill vs decode to the crew. Keep expanding with more shanties and metaphors.",
  "Write a nursery-rhyme style poem about thermal throttling and power caps. Add many stanzas and keep going.",
  "Write naturalistic dialogue between two ops engineers debugging a stuck vLLM queue. Continue for many turns without wrapping up.",
  "Write a courtroom cross-examination where the witness is a tokenizer. Keep adding Q&A exchanges.",
  "Write a travel-brochure parody for visiting a liquid-cooled GPU rack. Flowery marketing tone; keep expanding sections.",
  "Write a radio weather report for a GPU cluster: temperature fronts across racks, token-storm warnings. Keep broadcasting.",
]

def pick(concurrency, mode):
    # sparkDash pickDecodeBenchPrompts: same base prompt per stream with a "(stream i/n)" suffix.
    if concurrency <= 1 or mode == "sparkdash":
        return [PROSE] if concurrency <= 1 else [f"{PROSE} (stream {i+1}/{concurrency})" for i in range(concurrency)]
    return [PROSE] + TEXT_PROMPTS[: concurrency - 1]

def one_stream(url, model, prompt, max_tokens, out, idx, fill=True):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "top_p": 1, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
    if fill:
        body.update({"min_tokens": max_tokens, "ignore_eos": True, "stop": []})
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = last = None; usage = None; n = 0; text = []
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": break
                ev = json.loads(data)
                if ev.get("usage"): usage = ev["usage"]
                ch = ev.get("choices") or []
                if ch and (ch[0].get("delta") or {}).get("content"):
                    now = time.perf_counter()
                    if first is None: first = now
                    last = now; n += 1; text.append(ch[0]["delta"]["content"])
    except Exception as e:
        out[idx] = {"error": str(e)}; return
    comp = (usage or {}).get("completion_tokens") or n
    out[idx] = {"ttft_ms": (first - t0) * 1000 if first else None, "completion_tokens": comp, "chunks": n,
                "first": first, "last": last, "tps": (comp - 1) / (last - first) if first and last and last > first else 0.0,
                "prompt_tokens": (usage or {}).get("prompt_tokens"), "sample": "".join(text)[:80]}

def wave(url, model, concurrency, max_tokens, mode):
    prompts = pick(concurrency, mode); out = [None] * concurrency
    th = [threading.Thread(target=one_stream, args=(url, model, prompts[i], max_tokens, out, i)) for i in range(concurrency)]
    for t in th: t.start()
    for t in th: t.join()
    ok = [r for r in out if r and "tps" in r]
    firsts = [r["first"] for r in ok]; lasts = [r["last"] for r in ok]
    window = (max(lasts) - min(firsts)) if ok else 0
    total = sum(r["completion_tokens"] for r in ok)
    return {"concurrency": concurrency, "streams_ok": len(ok), "mean_tps": round(statistics.mean(r["tps"] for r in ok), 2) if ok else 0,
            "median_tps": round(statistics.median(r["tps"] for r in ok), 2) if ok else 0,
            "min_tps": round(min(r["tps"] for r in ok), 2) if ok else 0, "max_tps": round(max(r["tps"] for r in ok), 2) if ok else 0,
            "mean_ttft_ms": round(statistics.mean(r["ttft_ms"] for r in ok), 1) if ok else 0,
            "aggregate_tps": round(total / window, 2) if window > 0 else 0, "total_decode_tokens": total,
            "errors": [r.get("error") for r in out if r and "error" in r], "streams": out}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8888"); ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--concurrency", default="1,2,3,4"); ap.add_argument("--max-tokens", type=int, default=400); ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--mode", default="sparkdash", choices=["sparkdash", "varied"]); ap.add_argument("--out"); a = ap.parse_args()
    url = a.url.rstrip("/") + "/v1/chat/completions"
    w = [None]; one_stream(url, a.model, "Count from 1 to 40. Output only the numbers.", 32, w, 0, fill=False)
    print("warmup:", {k: w[0].get(k) for k in ("tps", "ttft_ms", "completion_tokens", "error")}, flush=True)
    results = []
    for c in [int(x) for x in a.concurrency.split(",")]:
        for r in range(a.repeats):
            res = wave(url, a.model, c, a.max_tokens, a.mode); results.append(res)
            print(f"x{c} run{r+1}: stream mean {res['mean_tps']} median {res['median_tps']} (min {res['min_tps']} max {res['max_tps']}) "
                  f"aggregate {res['aggregate_tps']} tok/s, ttft {res['mean_ttft_ms']} ms, ok {res['streams_ok']}/{c} errors={res['errors']}", flush=True)
    if a.out: json.dump(results, open(a.out, "w"), indent=1)

if __name__ == "__main__": main()
