# KServe quickstart, standard mode (kserve.github.io/website/docs/getting-started/quickstart-guide, v0.21,
# Kubernetes 1.32 or higher). Needs Docker, kind and kubectl; none are installed on the author's Mac, so not run here.
kind create cluster
curl -sL "https://github.com/kserve/kserve/releases/download/v0.21.0/kserve-standard-mode-full-install-with-manifests.sh" | bash
kubectl create namespace kserve-test
kubectl apply -f inferenceservice.yaml
kubectl get inferenceservices qwen-llm -n kserve-test
# when finished:
kubectl delete inferenceservice qwen-llm -n kserve-test
kind delete cluster
