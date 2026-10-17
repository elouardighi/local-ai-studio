"""GGUF architectures supported by the installed llama.cpp build.

Generated from llama.cpp source `src/llama-arch.cpp` at tag b11712
(0.7.0-dev, commit 10b8b7553) - the exact engine build shipped in
`engine/llama.cpp`. Regenerate with `scripts/fetch-archs.ps1` when
upgrading the engine.

An architecture absent from this set is *unverified*, not necessarily
unsupported: the engine itself is the final authority.
"""

ENGINE_TAG = "b11712"
ENGINE_VERSION = "0.7.0-dev"

SUPPORTED_ARCHS = frozenset({
'clip', 'llama', 'llama4', 'deci', 'falcon', 'grok', 'gpt2', 'gptj', 'gptneox', 'mpt', 'baichuan', 'starcoder', 'refact', 'bert', 'modern-bert', 'nomic-bert', 'nomic-bert-moe', 'neo-bert', 'jina-bert-v2', 'jina-bert-v3', 'eurobert', 'bloom', 'stablelm', 'qwen', 'qwen2', 'qwen2moe', 'qwen2vl', 'qwen3', 'qwen3moe', 'qwen3next', 'qwen3vl', 'qwen3vlmoe', 'qwen35', 'qwen35moe', 'clef', 'vllm-decision2', 'qwen4exp', 'phi2', 'phi3', 'phimoe', 'plamo', 'plamo2', 'plamo3', 'codeshell', 'orion', 'internlm2', 'minicpm', 'minicpm3', 'gemma', 'gemma2', 'gemma3', 'gemma3n', 'gemma4', 'gemma4-assistant', 'gemma-embedding', 'gemma-embedding2', 'starcoder2', 'mamba', 'mamba2', 'maple', 'jamba', 'falcon-h1', 'xverse', 'command-r', 'cohere2', 'cohere2moe', 'dbrx', 'olmo', 'olmo2', 'olmoe', 'muse-glimmer', 'muse-spark', 'openelm', 'arctic', 'deepseek', 'deepseek2', 'deepseek2-ocr', 'deepseek32', 'deepseek4', 'chatglm', 'glm4', 'glm4moe', 'glm-dsa', 'bitnet', 't5', 't5encoder', 'jais', 'jais2', 'nemotron', 'nemotron_h', 'nemotron_h_moe', 'exaone', 'exaone4', 'exaone-moe', 'rwkv6', 'rwkv6qwen2', 'rwkv7', 'arwkv7', 'granite', 'granitemoe', 'granitehybrid', 'graniteswitch', 'granite_swa', 'chameleon', 'wavtokenizer-dec', 'plm', 'bailingmoe', 'bailingmoe2', 'bailingmoe3', 'dots1', 'dots3note', 'arcee', 'afmoe', 'laguna', 'ernie4_5', 'ernie4_5-moe', 'hunyuan-moe', 'hunyuan-dense', 'hunyuan_vl', 'hy_v3', 'hy_v4', 'smollm3', 'gpt-oss', 'lfm2', 'lfm2moe', 'dream', 'smallthinker', 'llada', 'llada-moe', 'seed_oss', 'grovemoe', 'apertus', 'minimax-01', 'hrm_text', 'minimax-m2', 'minimax-m3', 'cogvlm', 'rnd1', 'pangu-embedded', 'mistral3', 'eagle3', 'dflash', 'mistral4', 'paddleocr', 'mimo2', 'step35', 'spark2_5', 'llama-embed', 'maincoder', 'maion-coder', 'kimi-linear', 'kimi-k3', 'glm5-next', 'talkie', 'mellum', 'nanbeige', 'qwen3tts', 'pockettts', 'k2-horizon', 'beam', 'kolibri1'
})

# Families that commonly ship a multimodal projector (mmproj) file.
VISION_ARCH_HINTS = ("vl", "clip", "minicpmv", "mllama", "pixtral")

# Substrings indicating an embedding-only architecture.
EMBEDDING_ARCH_HINTS = ("embedding", "bert", "jina", "gte")


def classify_arch(arch: str) -> dict:
    """Return compatibility facts for a GGUF architecture string."""
    arch = (arch or "").lower()
    return {
        "arch": arch,
        "supported_by_engine": arch in SUPPORTED_ARCHS,
        "is_vision": any(v in arch for v in VISION_ARCH_HINTS),
        "is_moe": "moe" in arch,
        "is_embedding": any(h in arch for h in EMBEDDING_ARCH_HINTS),
    }
