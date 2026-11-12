# Worked write-ups for the drill (blameless; read yours against these after you have written them)

## Incident 1: first tokens five times slower after a deploy

**Impact.** Kodekart's agent steps: median time to first token 252 ms to 1,241 ms, p95 482 ms to 2,303 ms, over the 2.0 s promise (4 of 32 requests, burn rate 2.5). Recorded run, not production.

**Timeline.** A gateway change shipped. Within the first minute the reused-token count per request fell from about 2,900 to 0 and tokens computed per request rose from 284 to 3,200.

**Cause.** The change sent `cache_prompt: false`, so the server recomputed the whole 3,200-token prompt on every step. Prefill grew 5.8 times (119.6 to 694.8 ms); queue and decode times grew as shadows of that, because the four slots share one GPU.

**What caught it, and what would have caught it sooner.** The trace split (12.2) and the `/metrics` counters did, afterwards. Sooner: an alert on the prefix-cache hit rate (reused / (computed + reused) was 83.6% before and 0% after) and a pre-rollout replay of a Kodekart trace that fails if the hit rate drops.

**Fix and follow-ups.** Revert the flag. Add the hit-rate alert (page if below half its 24-hour baseline for 10 minutes). Add the replay to the rollout gate.

## Incident 2: the engine crashed at 02:00 during the nightly batch (simulated)

**Impact.** All 39 in-flight requests were dropped; the supervisor restarted the engine.

**Cause.** The engine's memory was sized at 90% of the L4's 24 GB, leaving about 0.2 GB free. A second process, the embeddings model Kabir had placed on the same GPU to save money, held 1.7 GB and grew to 1.88 GB when the batch arrived. When the sampler asked for a 24 MB buffer, 20 MB was free.

**What it was not.** The KV cache was at 79%, not full, and PyTorch's own memory snapshot would have shown nothing wrong: it sees only memory PyTorch allocated, not the neighbour's.

**What would have caught it sooner.** Free GPU memory outside the pool, per process, from the driver; an alert below 1 GB. A restart-count alert.

**Fix and follow-ups.** Move the embeddings model to its own GPU, or lower the engine's memory share to leave headroom, and cap the number of concurrent sequences. Never share a GPU with a process that has its own allocator.

## Incident 3: answers got worse and nothing alarmed

**Impact.** Canary score 25/30 to 18/30; 8 of 30 replies contained Arabic or Chinese script in an English conversation. Latency and error rates were unchanged (the smaller file is, if anything, faster).

**Cause.** A smaller quantized file (Q2_K) replaced Q8_0 to save memory. Nobody evaluated it.

**What would have caught it.** The canary gate (12.5) run on the new file before it took traffic: it fails at a drop of more than 3. After release: the stray-script rate and the canary score on the live service.

**Fix and follow-ups.** Restore the Q8_0 file. Make the deploy compare the model file's hash with the approved one and run the canary before any traffic moves. Alert on canary score below 22/30 and on any rise in stray-script replies.
