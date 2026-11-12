# Deploy and tear-down commands, written from each provider's docs (read 2026-11-12). NOT RUN: no GPU cloud account
# was used to write this lesson. Check every flag against the page named beside it before you run it.

# Google Cloud Run with an L4 (docs.cloud.google.com/run/docs/configuring/services/gpu).
# L4 needs at least 4 CPU and 16 GiB; the page recommends 8 and 32. GPU services must stay under your regional quota.
gcloud run deploy monsoon-engine \
    --image IMAGE_URL \
    --gpu 1 --gpu-type nvidia-l4 --cpu 8 --memory 32Gi \
    --no-cpu-throttling --max-instances 1
# tear down:
gcloud run services delete monsoon-engine

# Modal (modal.com/docs/examples/vllm_snapshot): a deployed app lives until you stop it.
modal deploy ministral3_inference.py
modal app stop monsoon-engine   # APP_IDENTIFIER is the app's name or id (`modal app list`); or stop it on modal.com/apps
