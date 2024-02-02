
import sys
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer

import re

indic_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-dist-200M" #"ai4bharat/indictrans2-indic-indic-1B"  
indic_en_ckpt_dir = "ai4bharat/indictrans2-indic-en-dist-200M" # ai4bharat/indictrans2-indic-en-dist-200M
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if len(sys.argv) > 1:
    quantization = sys.argv[1]
else:
    quantization = ""


def initialize_model_and_tokenizer(ckpt_dir, direction, quantization):
    if quantization == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None

    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        quantization_config=qconfig,
    )

    if qconfig == None:
        model = model.to(DEVICE)
        # model.half()

    model.eval()

    return tokenizer, model


def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]

        # Preprocess the batch and extract entity mappings
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        # Tokenize the batch and generate input encodings
        inputs = tokenizer(
            batch,
            src=True,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(DEVICE)

        # Generate translations using the model
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        # Decode the generated tokens into text
        generated_tokens = tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        torch.cuda.empty_cache()

    return translations

ip = IndicProcessor(inference=True)

src_lang, tgt_lang = "eng_Latn", "tam_Taml"

indic_indic_tokenizer, indic_indic_model = initialize_model_and_tokenizer(
    indic_indic_ckpt_dir, "en-indic", quantization
)

indic_en_tokenizer, indic_en_model = initialize_model_and_tokenizer(
    indic_en_ckpt_dir, "indic-en", quantization
)


def translate(item: str, reverse=False):
    if (reverse):
        translated = batch_translate(
            re.split(r'\.|\n', item), tgt_lang, src_lang, indic_en_model, indic_en_tokenizer, ip
        )
        return "\n".join(translated)
    translated = batch_translate(
        re.split(r'\.|\n', item), src_lang, tgt_lang, indic_indic_model, indic_indic_tokenizer, ip
    )
    return "\n".join(translated)

