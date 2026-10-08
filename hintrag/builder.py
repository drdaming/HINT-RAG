import torch

from .concepts import CHEXPERT_CONCEPTS
from .data import build_prompt_ids, load_annotation, normalize_report
from .models.hintrag import HINTRAG
from .models.split_llm import SplitLLM
from .models.visual_encoder import VisualEncoder


def concepts_from_config(cfg):
    concepts = cfg.model.get("concepts")
    return list(concepts) if concepts else list(CHEXPERT_CONCEPTS)


def _corpus(cfg):
    annotation = load_annotation(cfg.data.ann_path)
    texts = [normalize_report(item["report"]) for split in annotation.values() for item in split]
    texts += [cfg.data.prompt_prefix, cfg.data.prompt_suffix] + concepts_from_config(cfg)
    return texts


def build_word_tokenizer(corpus):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "<unk>", "<s>", "</s>"]
    splitter = pre_tokenizers.Whitespace()
    words = sorted({w for text in corpus for w, _ in splitter.pre_tokenize_str(text.lower())} - set(specials))
    vocab = {token: i for i, token in enumerate(specials + words)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.normalizer = normalizers.Lowercase()
    backend.pre_tokenizer = splitter
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )


def build_tokenizer(cfg):
    if cfg.model.llm == "tiny":
        return build_word_tokenizer(_corpus(cfg))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.get("tokenizer") or cfg.model.llm, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    return tokenizer


def _dtype_argument():
    import transformers
    from packaging import version

    return "dtype" if version.parse(transformers.__version__) >= version.parse("4.56.0") else "torch_dtype"


def build_llm(cfg, tokenizer):
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    model_cfg = cfg.model
    dtype = getattr(torch, model_cfg.torch_dtype)
    if model_cfg.llm == "tiny":
        tiny = model_cfg.tiny_llm
        config = LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=tiny.hidden_size,
            intermediate_size=tiny.intermediate_size,
            num_hidden_layers=tiny.num_layers,
            num_attention_heads=tiny.num_heads,
            num_key_value_heads=tiny.num_heads,
            max_position_embeddings=2048,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            attn_implementation=model_cfg.attn_implementation,
        )
        return LlamaForCausalLM(config).to(dtype)
    kwargs = {_dtype_argument(): dtype, "attn_implementation": model_cfg.attn_implementation, "low_cpu_mem_usage": True}
    return AutoModelForCausalLM.from_pretrained(model_cfg.llm, **kwargs)


def build_text_encoder(cfg, tokenizer):
    from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel

    name = cfg.model.text_encoder
    if name == "tiny":
        tiny = cfg.model.tiny_text
        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=tiny.hidden_size,
            num_hidden_layers=tiny.num_layers,
            num_attention_heads=tiny.num_heads,
            intermediate_size=2 * tiny.hidden_size,
            max_position_embeddings=512,
            pad_token_id=tokenizer.pad_token_id,
        )
        return tokenizer, BertModel(config).eval()
    return AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()


def text_hidden_size(cfg):
    if cfg.model.text_encoder == "tiny":
        return int(cfg.model.tiny_text.hidden_size)
    from transformers import AutoConfig

    return int(AutoConfig.from_pretrained(cfg.model.text_encoder).hidden_size)


@torch.no_grad()
def encode_reports(texts, tokenizer, model, device, batch_size=64, max_length=256):
    model = model.to(device).eval()
    outputs = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        outputs.append(((hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)).cpu())
    return torch.cat(outputs) if outputs else torch.zeros(0, model.config.hidden_size)


def load_projector(model, path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    selected = {k.split("mm_projector.", 1)[1]: v for k, v in state.items() if "mm_projector." in k}
    model.projector.load_state_dict(selected or state, strict=True)


def build_model(cfg, tokenizer, value_dim):
    model_cfg = cfg.model
    visual = VisualEncoder(model_cfg.visual_encoder, model_cfg.image_size)
    llm = build_llm(cfg, tokenizer)
    split = SplitLLM(
        llm,
        model_cfg.split_layer,
        lora_rank=model_cfg.lora_rank,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
    )
    prefix_ids, suffix_ids = build_prompt_ids(tokenizer, cfg.data.prompt_prefix, cfg.data.prompt_suffix)
    model = HINTRAG(cfg, visual, split, tokenizer, value_dim, prefix_ids, suffix_ids, concepts_from_config(cfg))
    if model_cfg.get("projector_ckpt"):
        load_projector(model, model_cfg.projector_ckpt)
    return model
