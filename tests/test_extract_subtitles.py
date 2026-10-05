import json
import unittest

from gpt_subtitle_translator.subtitle_processor import SubtitleProcessor


class ExtractSubtitlesTest(unittest.TestCase):
    def setUp(self):
        self.processor = SubtitleProcessor(None)

    def test_well_formed_tuples_and_objects(self):
        response = json.dumps({"subtitles": [[1, "", "Hola"], {"id": 2, "translation": "Adiós"}]})
        self.assertEqual(self.processor.extract_subtitles(response, {}), "<1>Hola</1>\n<2>Adiós</2>")

    def test_malformed_items_return_empty_instead_of_raising(self):
        """Models without a strictly enforced schema (e.g. GPT-6 Luna) can return malformed items."""
        for items in ([[1, "", "Hola"], "2: Adiós"], [["one", "", "Hola"]], [[1, "", None]]):
            with self.subTest(items=items):
                self.assertEqual(self.processor.extract_subtitles(json.dumps({"subtitles": items}), {}), "")


if __name__ == "__main__":
    unittest.main()
