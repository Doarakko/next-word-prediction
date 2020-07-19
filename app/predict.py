import logging
import math
import string

import fasttext
import MeCab
import jieba
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertJapaneseTokenizer

logging.basicConfig(level=logging.INFO)

# ref: https://huggingface.co/transformers/pretrained_models.html
BERT_CHINESE = 'bert-base-chinese'
BERT_DUTCH = 'wietsedv/bert-base-dutch-cased'
BERT_ENGLISH = 'bert-base-uncased'
BERT_FINNISH = 'TurkuNLP/bert-base-finnish-uncased-v1'
BERT_GERMAN = 'bert-base-german-cased'
BERT_JAPANESE = 'cl-tohoku/bert-base-japanese-whole-word-masking'

english_tokenizer = BertTokenizer.from_pretrained(BERT_ENGLISH)
english_model = BertForMaskedLM.from_pretrained(BERT_ENGLISH).eval()

japanese_tokenizer = BertJapaneseTokenizer.from_pretrained(BERT_JAPANESE)
japanese_model = BertForMaskedLM.from_pretrained(BERT_JAPANESE).eval()

chinese_tokenizer = BertTokenizer.from_pretrained(BERT_CHINESE)
chinese_model = BertForMaskedLM.from_pretrained(BERT_CHINESE).eval()

# german_tokenizer = BertTokenizer.from_pretrained(BERT_GERMAN)
# german_model = BertForMaskedLM.from_pretrained(BERT_GERMAN).eval()

# dutch_tokenizer = BertTokenizer.from_pretrained(BERT_DUTCH)
# dutch_model = BertForMaskedLM.from_pretrained(BERT_DUTCH).eval()

# finnish_tokenizer = BertTokenizer.from_pretrained(BERT_FINNISH)
# finnish_model = BertForMaskedLM.from_pretrained(BERT_FINNISH).eval()

language_model = fasttext.load_model('lid.176.bin')

wakati = MeCab.Tagger("-Owakati")


def predict_language(s: str) -> str:
    return language_model.predict(s, k=1)[0][0].replace('__label__', '')


def _get_tokenizer_and_model(language: str):
    if language == 'ch':
        return chinese_tokenizer, chinese_model
    elif language == 'de':
        return german_tokenizer, german_model
    elif language == 'en':
        return english_tokenizer, english_model
    elif language == 'fi':
        return finnish_tokenizer, finnish_model
    elif language == 'ja':
        return japanese_tokenizer, japanese_model
    elif language == 'nl':
        return dutch_tokenizer, dutch_model

    raise Exception


def _split_sentence(s: str, language: str):
    if language == 'ch':
        return _split_chinese_sentence(s)
    elif language == 'ja':
        return _split_japanese_sentence(s)

    return s.split(' ')


def _split_chinese_sentence(s: str):
    return jieba.cut(s)


def _split_japanese_sentence(s: str):
    return wakati.parse(s).split()


def _decode(tokenizer, predictions, mask_idx):
    _, predicted_indexes = torch.topk(predictions[0, mask_idx], k=5)

    predicted_tokens = tokenizer.convert_ids_to_tokens(
        predicted_indexes.tolist())

    return predicted_tokens


def _encode(tokenizer, body, language):
    if body.count('[mask]') > 1:
        raise Exception

    if language in ['ja', 'ch']:
        mask_idx = body.find('[mask]')
        logging.info(mask_idx)

        body = body.replace('[mask]', ' ')
        tokenized_text = _split_sentence(body, language)

        if mask_idx == 0:
            tokenized_text.insert(0, '[MASK]')
        else:
            sum = 0
            for i, word in enumerate(tokenized_text):
                sum += len(word)

                if sum == mask_idx:
                    tokenized_text.insert(i+1, '[MASK]')
    else:
        body = body.replace('[mask]', tokenizer.mask_token)
        tokenized_text = _split_sentence(body, language)

    tokenized_text.insert(0, '[CLS]')
    tokenized_text.append('[SEP]')

    logging.info(tokenized_text)

    input_ids = torch.tensor(
        [tokenizer.encode(tokenized_text)])

    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]

    logging.info(input_ids)
    logging.info(mask_idx)

    return input_ids, mask_idx


def predict_next_words(body, language):
    tokenizer, model = _get_tokenizer_and_model(language=language)
    input_ids, mask_idx = _encode(tokenizer, body, language)

    with torch.no_grad():
        pred = model(input_ids)[0]

    return _decode(tokenizer, pred, mask_idx)
