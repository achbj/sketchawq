# Comparison Methods

Use GPTQ as the first paper-eval baseline because it is common, weight-only, and already supported by this project.

Useful next methods:

1. AWQ / GPTQModel-AWQ
   - Closest conceptual baseline to SketchAWQ because SketchAWQ extends AWQ-style scaling.
   - Good for answering: does the Hessian sketch signal improve over standard AWQ?

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
