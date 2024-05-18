import argparse
import os
import time
from gpt_subtitle_translator.constants import TOKENS_PER_CHUNK, DEFAULT_MODEL, MAX_RETRIES, DEFAULT_TEMPERATURE
from gpt_subtitle_translator.models.claude import Claude
from gpt_subtitle_translator.models.gemeni import Gemeni
from gpt_subtitle_translator.models.gpt import GPT
from gpt_subtitle_translator.subtitle_translator import SubtitleTranslator, TranslationError
from gpt_subtitle_translator.logger import logger
import chardet

def get_output_filename(input_filename):
    token = int(time.time())
    directory, filename = os.path.split(input_filename)
    return os.path.join(directory, f"{filename.split('.')[0]}_{token}_translated.srt")

def main():
    parser = argparse.ArgumentParser(description='Translate a transcript file.')
    parser.add_argument('file', help='The transcript file to translate.', nargs='?', default='')
    parser.add_argument('-l', '--language', type=str, default="English", help='Language to translate to.')
    parser.add_argument('-t', '--threads', type=int, default=1, help='Number of threads to use.')
    parser.add_argument('-temp', '--temperature', type=float, default=DEFAULT_TEMPERATURE, help='Temperature for generation.')
    parser.add_argument('-s', '--chunk_size', type=int, default=TOKENS_PER_CHUNK, help='Number of tokens per chunk.')
    parser.add_argument('-m', '--model', type=str, default=DEFAULT_MODEL, help='Model to use.')
    parser.add_argument('-r', '--retries', type=int, default=MAX_RETRIES, help='Number of retries.')

    args = parser.parse_args()

    with open(args.file, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']

    with open(args.file, 'r', encoding=encoding) as f:
        srt_data = f.read()

    filename = get_output_filename(args.file)
    # model = Gemeni()
    model = GPT(args.model) if args.model.startswith("gpt") else Claude(args.model)
    translator = SubtitleTranslator(
        model=model,
        lang=args.language,
        num_threads=args.threads,
        tokens_per_chunk=args.chunk_size,
        max_retries=args.retries,
        temperature=args.temperature
    )

    try:
        result = translator.translate_subtitles(srt_data)
    except TranslationError as e:
        return logger.error(e)
    finally:
        logger.info(f"Total API cost: ${model.get_total_cost():.5f}")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(result)

    logger.info(f"Translated subtitles with {args.model}, file written to {filename}")

if __name__ == '__main__':
    main()
