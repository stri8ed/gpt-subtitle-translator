# GPT Subtitle Translator

This tool is designed to translate subtitles with GPT model(s). It's optimized for use with the "gpt-4-1106-preview" model, and using this model is highly recommended for best results.
### Features
- Strips the timestamps from the subtitles before translating, to reduce input tokens.
- Runs multiple chunks in parallel to speed up the translation process.
- Tries to ensure model does not skip or merge subtitles while translating.

## Installation

Install the dependencies:

```
pip install -r requirements.txt
```

Create an environment file:
1. Rename the `.env.example` at the root of your project to `.env`
2. In the new `.env` file, replace `YOUR_OPENAI_API_KEY`, and `TARGET_LANGUAGE` with the correct values.

## Usage

```
python main.py path/to/subtitles.srt -t 2
```

## Options

```
-t  Number of threads to use (default: 1)                      
```
