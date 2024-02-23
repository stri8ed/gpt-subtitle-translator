MODEL = "gpt-3.5-turbo-0125"
TOKENS_PER_CHUNK = 2_500  # Safe value, might be able to increase depending on the language and content
MAX_RETRIES = 2  # Despite instructions, model sometimes skips/merges subtitles. Retrying helps.
TEMPERATURE = 0.4
DELIMITER = "||"