import unittest
import re
from unittest.mock import MagicMock

from gpt_subtitle_translator.subtitle_processor import SubtitleProcessor


class TestSubtitleProcessor(unittest.TestCase):
    def setUp(self):
        mock_model = MagicMock()
        mock_model.num_tokens_from_string.return_value = 50
        mock_model.max_output_tokens.return_value = 1000
        self.processor = SubtitleProcessor(mock_model)

    def test_parse_srt(self):
        srt_content = "1\n00:00:01,000 --> 00:00:04,000\nHello World\n\n2\n00:00:05,000 --> 00:00:08,000\nGoodbye\n"
        expected = {
            1: {"timestamp": "00:00:01,000 --> 00:00:04,000", "text": "Hello World"},
            2: {"timestamp": "00:00:05,000 --> 00:00:08,000", "text": "Goodbye"}
        }
        self.assertEqual(self.processor.parse_srt(srt_content), expected)

    def test_preprocess(self):
        parsed_data = {1: {"timestamp": "00:00:01,000 --> 00:00:04,000", "text": "Hello World"}}
        expected = "<1>Hello World</1>"
        self.assertEqual(self.processor.preprocess(parsed_data), expected)

    def test_split_on_tags(self):
        tagged_text = "<1>Hello World</1>\n<2>Goodbye</2>"
        expected = ["<1>Hello World</1>", "<2>Goodbye</2>"]
        self.assertEqual(self.processor.split_on_tags(tagged_text), expected)

    def test_randomize_ids(self):
        tagged_text = "<1>Hello World</1>\n<2>Goodbye</2>"
        randomized_text, mapping = self.processor.randomize_ids(tagged_text)

        original_ids = re.findall(r"<(\d+)>", tagged_text)
        randomized_ids = re.findall(r"<(\d+)>", randomized_text)

        self.assertNotEqual(set(original_ids), set(randomized_ids), "IDs were not randomized")
        self.assertEqual(len(set(randomized_ids)), len(original_ids), "Not all IDs are unique after randomization")
        self.assertEqual(len(original_ids), len(randomized_ids), "Number of tags changed after randomization")

    def test_revert_id_randomization(self):
        original_text = "<1>Hello World</1>\n<2>Goodbye</2>"
        randomized_text, mapping = self.processor.randomize_ids(original_text)
        reverted_text = self.processor.revert_id_randomization(randomized_text, mapping)
        self.assertEqual(reverted_text, original_text)

    def test_post_process_text(self):
        original_subs = {1: {"timestamp": "00:00:01,000 --> 00:00:04,000", "text": "Hello World"}}
        content = "<1>Hello World</1>"
        expected_output = "1\n00:00:01,000 --> 00:00:04,000\nHello World"
        self.assertEqual(self.processor.post_process_text(content, original_subs), expected_output)

    def test_get_missing_subtitles(self):
        original_text = "<1>Hello</1>\n<2>World</2>"
        translated_text = "<1>Hello</1>"
        missing = self.processor.get_missing_subtitles(translated_text, original_text)
        self.assertIn("2", missing)

if __name__ == '__main__':
    unittest.main()
