# Four calls against the gateway on :8091 (first upstream is a dead port, so every success is a fallback).
U=http://127.0.0.1:8091/v1/chat/completions
B='{"model":"m","max_tokens":200,"messages":[{"role":"user","content":"Say hi."}],"chat_template_kwargs":{"enable_thinking":false}}'
echo "no key:    $(curl -s -o /dev/null -w '%{http_code}' -X POST $U -d "$B" -H 'content-type: application/json')"
echo "bad key:   $(curl -s -o /dev/null -w '%{http_code}' -X POST $U -d "$B" -H 'content-type: application/json' -H 'Authorization: Bearer nope')"
for i in 1 2 3 4; do
  echo "good key $i: $(curl -s -o /dev/null -w '%{http_code} retry-after=%header{retry-after}' -X POST $U -d "$B" -H 'content-type: application/json' -H 'Authorization: Bearer sk-tidewater')"
done
