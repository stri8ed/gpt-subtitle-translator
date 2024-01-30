MODEL = "gpt-4-turbo-preview"
TOKENS_PER_CHUNK = 2_500  # Safe value, might be able to increase depending on the language
MAX_OUTPUT_TOKENS = 4_095  # Fixed by OpenAI
MAX_RETRIES = 2  # Despite instructions, model sometimes skips/merges subtitles. Retrying helps.
TEMPERATURE = 0.4