import unittest

import srt

from gpt_subtitle_translator.subtitle_processor import SubtitleProcessor


def cue(index, text):
    return f"{index}\n00:00:0{index},000 --> 00:00:0{index},900\n{text}"


class CleanTextTest(unittest.TestCase):
    def test_closing_italics_at_cue_end_keeps_tag_and_cue_separation(self):
        text = cue(1, "<i>Hola</i>") + "\n\n" + cue(2, "Adiós")

        cleaned = SubtitleProcessor.clean_text(text)

        self.assertEqual(cleaned, text)
        self.assertEqual([s.content for s in srt.parse(cleaned)], ["<i>Hola</i>", "Adiós"])

    def test_multiline_italics_and_font_tags_are_kept(self):
        text = (cue(1, "<i>A minister comes\nand you're partying?</i>") + "\n\n"
                + cue(2, '<font color="#ffff00">Thank you.</font>') + "\n\n"
                + cue(3, "Last"))

        self.assertEqual(SubtitleProcessor.clean_text(text), text)

    def test_stray_angle_brackets_at_line_end_are_removed(self):
        text = cue(1, "Where are you going >\nCome back <") + "\n\n" + cue(2, "Fine>  ")

        cleaned = SubtitleProcessor.clean_text(text)

        self.assertEqual(cleaned, cue(1, "Where are you going \nCome back ") + "\n\n" + cue(2, "Fine"))
        self.assertEqual(len(list(srt.parse(cleaned))), 2)

    def test_stray_bracket_after_text_with_inline_tag_is_removed(self):
        self.assertEqual(SubtitleProcessor.clean_text("so <i>yaar</i> lame>"), "so <i>yaar</i> lame")


if __name__ == "__main__":
    unittest.main()
