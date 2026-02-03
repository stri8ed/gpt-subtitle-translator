DEFAULT_MODEL = "gemini-3-flash-preview"
TOKENS_PER_CHUNK = 4000  # Safe value, might be able to increase depending on the language and content
MAX_RETRIES = 3  # Despite instructions, model sometimes skips/merges subtitles. Retrying helps.
DEFAULT_TEMPERATURE = 0.1
COMPRESSION_RATIO_THRESHOLD = 5.0