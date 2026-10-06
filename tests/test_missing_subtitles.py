import json
import threading
import unittest

from gpt_subtitle_translator.subtitle_processor import SubtitleProcessor, Chunk
from gpt_subtitle_translator.subtitle_translator import SubtitleTranslator, MissingSubtitlesError, EmptySubtitlesError


class StubModel:
    def __init__(self, translations=None):
        self.translations = translations or {}
        self.calls = 0

    def max_output_tokens(self):
        return 100_000

    def generate_completion(self, prompt, temperature, target_language=None):
        """Answers with self.translations (keyed by source text), so it works whatever ids/order the prompt uses."""
        self.calls += 1
        subtitles = [[int(id_), "", self.translations[text]]
                     for id_, text in SubtitleProcessor.TAG_PATTERN.findall(prompt) if text in self.translations]
        return json.dumps({"language": "Spanish", "subtitles": subtitles}), 100


def make_translator(model=None, max_retries=0):
    translator = SubtitleTranslator.__new__(SubtitleTranslator)
    translator.model = model or StubModel()
    translator.processor = SubtitleProcessor(translator.model)
    translator.check_untranslated = True
    translator.max_retries = max_retries
    translator.retry_on_refusal = False
    translator.temperature = 0.1
    translator.lang = "Spanish"
    translator.prompt_template = "{subtitles}"
    return translator


def tag(rows):
    return "\n".join(f"<{i}>{text}</{i}>" for i, text in rows)


class EmptySubtitlesTest(unittest.TestCase):
    def setUp(self):
        self.translator = make_translator()
        self.processor = self.translator.processor

    def test_cue_merged_into_neighbour_and_left_empty_is_detected(self):
        """Seen with Muse Spark: cue 66's text merged into 65, leaving 66 empty."""
        source = tag([(65, "여기가 내가 사는 청담동 나 따라"), (66, "나온 거야?"), (67, "응.")])
        response = tag([(65, "This is Cheongdam-dong where I live, did you follow\nme here?"), (66, ""), (67, "Yeah.")])

        self.assertEqual(self.processor.get_empty_subtitles(response, source), {"66": "나온 거야?"})
        self.assertEqual(self.processor.get_missing_subtitles(response, source), {})
        with self.assertRaisesRegex(EmptySubtitlesError, "1 empty subtitles"):
            self.translator.validate_response(response, source, 1, response, 100)

    def test_markup_only_translation_counts_as_empty(self):
        source = tag([(1, "<i>Hello there</i>"), (2, "Bye")])
        response = tag([(1, "<i></i>"), (2, "Adiós")])

        self.assertEqual(set(self.processor.get_empty_subtitles(response, source)), {"1"})

    def test_junk_source_may_be_dropped(self):
        source = tag([(1, "////"), (2, ".88"), (3, "Hello")])
        response = tag([(1, ""), (2, ""), (3, "Hola")])

        self.assertEqual(self.processor.get_empty_subtitles(response, source), {})

    def test_numeric_translation_is_not_empty(self):
        source = tag([(1, "Fourteen."), (2, "Okay")])
        response = tag([(1, "14."), (2, "Vale")])

        self.assertEqual(self.processor.get_empty_subtitles(response, source), {})

    def test_all_cues_empty_fails_as_missing_all(self):
        source = tag([(1, "One"), (2, "Two")])
        response = tag([(1, ""), (2, "")])

        with self.assertRaises(MissingSubtitlesError) as ctx:
            self.translator.validate_response(response, source, 1, response, 100)
        self.assertNotIsInstance(ctx.exception, EmptySubtitlesError)


class EmptySubtitlesRetryTest(unittest.TestCase):
    def translate(self, translations, max_retries):
        model = StubModel(translations)
        translator = make_translator(model, max_retries=max_retries)
        chunk = Chunk(text=tag([(1, "Hello"), (2, "dh, bwy, shdkhw"), (3, "Bye")]), num_tokens=10, idx=0)
        return model, translator.translate_chunk(chunk, threading.Event(), 0)

    def test_empty_cue_is_retried_then_accepted(self):
        """OCR garbage the model keeps dropping must not fail the file."""
        model, (_, response, _) = self.translate({"Hello": "Hola", "dh, bwy, shdkhw": "", "Bye": "Adiós"}, max_retries=2)

        self.assertEqual(model.calls, 3)
        self.assertEqual(response, tag([(1, "Hola"), (2, ""), (3, "Adiós")]))

    def test_absent_cue_still_fails_after_retries(self):
        model = StubModel({"Hello": "Hola", "Bye": "Adiós"})
        translator = make_translator(model, max_retries=1)
        chunk = Chunk(text=tag([(1, "Hello"), (2, "Middle"), (3, "Bye")]), num_tokens=10, idx=0)

        with self.assertRaises(MissingSubtitlesError) as ctx:
            translator.translate_chunk(chunk, threading.Event(), 0)
        self.assertNotIsInstance(ctx.exception, EmptySubtitlesError)
        self.assertEqual(model.calls, 2)


if __name__ == "__main__":
    unittest.main()
