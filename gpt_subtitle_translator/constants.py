DEFAULT_MODEL = "claude-3-haiku-20240307"
TOKENS_PER_CHUNK = 2500  # Safe value, might be able to increase depending on the language and content
MAX_RETRIES = 3  # Despite instructions, model sometimes skips/merges subtitles. Retrying helps.
DEFAULT_TEMPERATURE = 0.3
COMPRESSION_RATIO_THRESHOLD = 2.5