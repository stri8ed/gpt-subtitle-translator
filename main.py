import argparse
import os
import time

from constants import TOKENS_PER_CHUNK
from translate import translate_subtitles
from logger import logger
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

    args = parser.parse_args()

    with open(args.file, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']

    with open(args.file, 'r', encoding=encoding) as f:
        srt_data = f.read()

    filename = get_output_filename(args.file)
    result = translate_subtitles(
        lang=args.language,
        srt_data = srt_data,
        num_threads = args.threads,
        tokens_per_chunk=args.chunk_size
    )

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(result)

    logger.info(f"Translated subtitles written to {filename}")

if __name__ == '__main__':
    main()
