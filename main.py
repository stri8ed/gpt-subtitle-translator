import argparse
import time
from translate import translate_subtitles

def get_output_filename(input_filename):
    token = int(time.time())
    return f"{input_filename.split('.')[0]}_{token}_translated.srt"

def main():
    parser = argparse.ArgumentParser(description='Translate a transcript file.')
    parser.add_argument('file', help='The transcript file to translate.', nargs='?', default='')
    parser.add_argument('-t', type=int, default=1, help='Number of threads to use.')

    args = parser.parse_args()

    with open(args.file, 'r', encoding='utf-8') as f:
        transcript = f.read()

    filename = get_output_filename(args.file)
    result = translate_subtitles(transcript, args.t)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(result)

    print(f"Translated transcript written to {filename}")

if __name__ == '__main__':
    main()