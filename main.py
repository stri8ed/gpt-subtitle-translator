import argparse
import os
import time
from src.constants import TOKENS_PER_CHUNK, DEFAULT_MODEL
from src.models.claude import Claude
from src.models.gpt import GPT
from src.subtitle_translator import SubtitleTranslator
from src.logger import logger
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
    parser.add_argument('-s', '--chunk_size', type=int, default=TOKENS_PER_CHUNK, help='Number of tokens per chunk.')
    parser.add_argument('-m', '--model', type=str, default=DEFAULT_MODEL, help='Model to use.')

    args = parser.parse_args()

    with open(args.file, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']

    with open(args.file, 'r', encoding=encoding) as f:
        srt_data = f.read()

    filename = get_output_filename(args.file)
    model = GPT(args.model) if args.model.startswith("gpt") else Claude(args.model)
    translator = SubtitleTranslator(
        model=model,
        lang=args.language,
        num_threads=args.threads,
        tokens_per_chunk=args.chunk_size
    )
    result = translator.translate_subtitles(srt_data)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(result)

    logger.info(f"Translated subtitles with {args.model}, file written to {filename}")

if __name__ == '__main__':
    main()
