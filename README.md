Reap — Pruning & Router Fine-tuning for MoE Models

**How to use**:
- **Entry point**: The main entry for pruning runs is the shell wrapper `prune.sh` at the repository root. Run this script to start the pruning workflow.
- **Pruning parameters**: Configure pruning-related options in `experiments/pruning-cli.sh` (this file contains the pruning flags and configuration used by `prune.sh`).
- **Router fine-tuning parameters**: Configure router fine-tuning options in `experiments/finetuning-cli.sh` (this controls parameters used by `src/reap/finetune.py`).

**What we changed / added**:
- **`src/reap/finetune.py`**: Added a router fine-tuning module and CLI entry to rehabilitate router weights after physical pruning (small-batch training with gradient accumulation, freezing non-router params).
- **`src/reap/gpt_experts.py`**: Added GptOss-specific MoE utilities: activation extraction for batched-expert MLPs and a physical pruning routine that slices batched expert tensors safely (`.index_select(...).clone().contiguous()`), plus router row-slicing.
- **`src/reap/model_util.py`**: Updated `MODEL_ATTRS` and helpers to include `GptOssForCausalLM` (attribute names and a marker for batched experts) so pruning and inspection logic can handle GptOss layouts.
- **`src/reap/prune.py`**: Integrated the GptOss pruning path and optionally calls the router fine-tune routine after pruning. The pruning flow now detects batched expert implementations and uses `prune_gptoss_experts` when appropriate.
- **Other files touched**: minor dataset/IO helpers and glue code (for example `src/reap/data.py`) were reviewed to ensure tokenization and dataset access remain compatible with the new fine-tuning flow.

Brief principle behind the changes:
- Many GptOss-style models store experts as batched tensors rather than separate ModuleList entries. To support them we added code to:
	- Compute activations and router selections for the batched-expert MLPs so observers can collect per-expert statistics.
	- Physically remove experts by slicing the batched expert tensors and router rows, and ensuring tensors are `contiguous()` to avoid safetensors/stride issues on save.
	- Update model attributes and config entries so downstream tooling (saving, eval, and finetuning) sees the reduced expert counts.
	- Provide a small router fine-tune routine to re-calibrate the router after pruning, since row-slicing changes the softmax normalization over experts.

**Experiments & example**:
- We used the Hugging Face model `unsloth/gpt-oss-20b-BF16` as an example and ran the pruning + router rehabilitation workflow.
- The dataset used for specialization was `m-a-p/CodeFeedback-Filtered-Instruction` (code-focused instruction dataset). Using this dataset we converted the GPT-Oss-20B instance into a code-specialist variant and ran evaluation on WildBench — the workflow completed successfully.

If you want me to produce a short reproduction script, example command lines, or commit these changes, tell me which you prefer next.

