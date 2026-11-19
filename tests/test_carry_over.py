import json
import unittest

from gpt_subtitle_translator.subtitle_processor import SubtitleProcessor
from gpt_subtitle_translator.subtitle_translator import SubtitleTranslator, TranslationError


class TokenizerStubModel:
    def __init__(self, tokens_per_cue, failing_text=None):
        self.tokens_per_cue = tokens_per_cue
        self.failing_text = failing_text
        self.prompts = []

    def init_vocab(self, text):
        pass

    def num_tokens_from_string(self, text):
        return self.tokens_per_cue

    def max_output_tokens(self):
        return 100_000

    def generate_completion(self, prompt, temperature, target_language=None):
        self.prompts.append(prompt)
        if self.failing_text and self.failing_text in prompt:
            raise RuntimeError("503 UNAVAILABLE")
        subtitles = [[int(id_), "", text.replace("cue", "translated")]
                     for id_, text in SubtitleProcessor.TAG_PATTERN.findall(prompt)]
        return json.dumps({"language": "English", "subtitles": subtitles}), 100


def make_srt(count):
    return "\n\n".join(
        f"{i}\n00:00:{i:02d},000 --> 00:00:{i:02d},500\ncue {i}" for i in range(1, count + 1)
    )


def translator_for(model):
    translator = SubtitleTranslator.__new__(SubtitleTranslator)
    translator.model = model
    translator.processor = SubtitleProcessor(model)
    translator.lang = "English"
    translator.temperature = 0.1
    translator.num_threads = 1
    translator.tokens_per_chunk = 18
    translator.max_retries = 0
    translator.retry_on_refusal = False
    translator.check_untranslated = True
    translator.prompt_template = "{subtitles}"
    return translator


class CarryOverTest(unittest.TestCase):
    def test_carry_over_between_models_with_different_chunk_boundaries(self):
        """Seen on job 3408cf62: gemini-3-flash-preview put cues 1-165 in chunk 1 and gemini-4-hydrogen put
        1-169 in its chunk 1, so carrying chunk 1 over by index lost 166-169 and translated others twice."""
        srt = make_srt(10)

        three_per_chunk = TokenizerStubModel(tokens_per_cue=5, failing_text="cue 4<")
        with self.assertRaises(TranslationError) as raised:
            translator_for(three_per_chunk).translate_subtitles(srt)
        self.assertEqual(len(raised.exception.completed_chunks), 1)

        four_per_chunk = TokenizerStubModel(tokens_per_cue=3)
        result = translator_for(four_per_chunk).translate_subtitles(
            srt, completed_chunks=raised.exception.completed_chunks
        )

        texts = [block.split("\n")[2] for block in result.strip().split("\n\n")]
        self.assertEqual(texts, [f"translated {i}" for i in range(1, 11)])
        retranslated = {int(text.removeprefix("cue ")) for prompt in four_per_chunk.prompts
                        for _, text in SubtitleProcessor.TAG_PATTERN.findall(prompt)}
        self.assertEqual(retranslated, set(range(4, 11)))

    def test_chunk_never_spans_translated_cues(self):
        completed = [
            "\n".join(f"<{i}>translated {i}</{i}>" for i in ids) for ids in (range(1, 4), range(7, 10))
        ]
        model = TokenizerStubModel(tokens_per_cue=3)
        result = translator_for(model).translate_subtitles(make_srt(10), completed_chunks=completed)

        texts = [block.split("\n")[2] for block in result.strip().split("\n\n")]
        self.assertEqual(texts, [f"translated {i}" for i in range(1, 11)])
        prompted = [[int(text.removeprefix("cue ")) for _, text in SubtitleProcessor.TAG_PATTERN.findall(prompt)]
                    for prompt in model.prompts]
        self.assertEqual(sorted(prompted), [[4, 5, 6], [10]])


if __name__ == "__main__":
    unittest.main()
