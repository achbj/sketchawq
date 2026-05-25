# Comparison Methods

Use GPTQ as the first paper-eval baseline because it is common, weight-only, and already supported by this project.

Useful next methods:

1. AWQ / official llm-awq
   - Closest conceptual baseline to SketchAWQ because SketchAWQ extends AWQ-style scaling.
   - Good for answering: does the Hessian sketch signal improve over standard AWQ?
   - Use the external `/Users/achbj/Work/quantization` `mit-han-lab/llm-awq` harness as the paper-style AWQ reference for OPT-2.7B.
   - Current OPT-2.7B anchors: W4-g128 fake-quant WikiText2 PPL `12.7043`; W3-g128 fake-quant WikiText2 PPL `13.5499`.
   - Treat GPTQModel-AWQ as an implementation/backend check only until it reproduces the official `llm-awq` numbers. The `awq_torch` run with PPL `9837.5970` is rejected.

2. HQQ
   - Strong no-calibration or light-calibration quantization baseline.
   - Good for showing whether SketchAWQ's calibration cost buys quality.

3. LLMCompressor AWQ
   - Practical ecosystem baseline used by deployment pipelines.
   - Good for "can users reproduce this with existing compression tooling?" comparisons.

4. bitsandbytes NF4
   - Useful as a broad 4-bit reference, but not a perfect apples-to-apples comparison because it is not the same weight-only W3/W4 setup.
   - Keep it separate from the main paper-eval board unless the caption clearly states the difference.

5. Dense FP16/BF16
   - Always include as the quality anchor through `original_perplexity`.
   - Do not treat it as a compression baseline; use it as the reference for PPL ratio and degradation.

For the main table, prefer this order: FP16 reference, GPTQ, AWQ, HQQ, LLMCompressor-AWQ, SketchAWQ.
