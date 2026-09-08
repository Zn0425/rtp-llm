# Backend input embedding smoke

`//rtp_llm/test/smoke:input_embedding_rpc_test` starts a Qwen2.5-0.5B backend
with frontend and PromptGenerator processes disabled. It sends requests through
`ModelRpcClient` and checks hidden states and logits for equivalent embedding
rows, independently perturbed spans, multiple return sequences, and invalid
embedding widths.

Supply a local pretrained Qwen2.5-0.5B-Instruct checkpoint when running outside
the CI model mount:

```bash
bazelisk test //rtp_llm/test/smoke:input_embedding_rpc_test \
  --config=cuda12_9 --config=sm9x \
  --test_env=INPUT_EMBEDDING_SMOKE_MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct
```

The checkpoint must contain the model configuration and safetensors weights.
The test reads the required embedding rows without modifying the checkpoint.

The embedding requests additionally exercise the production visitor and
`HostService` domain routing, without starting a frontend process. A reachable
gRPC test Master rejects metadata-only requests like a BATCH deployment; a
control request verifies this rejection, then embedding requests must bypass
that endpoint and reach the real backend. This checks the client bypass, not
Java dispatcher behavior.

Custom embeddings cannot be inlined into FlexLB's scheduling payload. Single
embedding requests therefore use configured backend domains or explicit role
addresses. Master-only deployments must supply these backend addresses. Local
clients with an explicit backend endpoint continue to work without discovery.
Domain routing does not provide FlexLB queue ordering, priority scheduling, or
its global concurrency limits; configure request concurrency at the caller or
service boundary. The cached Master queue rejection remains an advisory check.
This test does not add remote domain routing support to the batch client API.
