def build_json_schema(target_language: str = None):
    language_def = {"type": "string", "enum": [target_language]} if target_language else {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "language": language_def,
            "subtitles": {
                "type": "array",
                "items": {
                    "type": "array",
                    "prefixItems": [
                        {"type": "integer"},   # id
                        {"type": "string"},    # thoughts
                        {"type": "string"}     # translation
                    ],
                    "minItems": 3,
                    "maxItems": 3
                }
            }
        },
        "required": ["language", "subtitles"]
    }
