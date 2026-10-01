import unittest
from pathlib import Path

from gpt_subtitle_translator.subtitle_processor import SubtitleProcessor
from gpt_subtitle_translator.subtitle_translator import SubtitleTranslator, UntranslatedResponseError

FIXTURES = Path(__file__).parent / "fixtures"


class StubModel:
    def max_output_tokens(self):
        return 100_000


def make_translator():
    translator = SubtitleTranslator.__new__(SubtitleTranslator)
    translator.model = StubModel()
    translator.processor = SubtitleProcessor(translator.model)
    return translator


def load(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def tag(rows):
    return "\n".join(f"<{i}>{text}</{i}>" for i, text in rows)


class UntranslatedDetectionTest(unittest.TestCase):
    def setUp(self):
        self.translator = make_translator()

    def test_near_verbatim_chunk_is_rejected(self):
        """Subtitle 459230: chunk copied back in Turkish with one dropped letter ("düşündük" -> "düşünük")."""
        source = load("untranslated_chunk_source.txt")
        response = load("untranslated_chunk_response.txt")

        self.assertNotEqual(self.translator.normalize_text(source), self.translator.normalize_text(response))
        with self.assertRaisesRegex(UntranslatedResponseError, "verbatim without translating"):
            self.translator.validate_response(response, source, 7, response, 1000)

    def test_real_translation_passes(self):
        source = load("translated_chunk_source.txt")
        response = load("translated_chunk_response.txt")

        eligible, untranslated = self.translator.find_untranslated_subtitles(response, source)
        self.assertGreater(eligible, 50)
        self.assertEqual(untranslated, [])
        self.translator.validate_response(response, source, 8, response, 1000)

    def test_names_and_short_lines_are_ignored(self):
        source = tag([(1, "Kenan!"), (2, "Havva."), (3, "Maşallah."), (4, "Kenan! Kenan!"), (5, "Ceyhun."),
                      (6, "Leyla."), (7, "Medet!")])

        eligible, untranslated = self.translator.find_untranslated_subtitles(source, source)
        self.assertEqual(eligible, 0)
        self.translator.check_untranslated_subtitles(source, source, 1)

    def test_partial_copy_below_threshold_passes(self):
        source_rows, response_rows = self.partial_copy(8)
        self.translator.check_untranslated_subtitles(tag(response_rows), tag(source_rows), 1)

    def test_partial_copy_above_threshold_is_rejected(self):
        source_rows, response_rows = self.partial_copy(9)
        with self.assertRaises(UntranslatedResponseError):
            self.translator.check_untranslated_subtitles(tag(response_rows), tag(source_rows), 1)

    def test_close_but_different_translation_passes(self):
        source_rows = [
            (1, "No sé qué hacer ahora."), (2, "Vamos a la casa de mi madre."), (3, "¿Dónde está el coche?"),
            (4, "Tengo que hablar con él."), (5, "No es tan fácil como parece."), (6, "Ella no quiere venir."),
            (7, "Mañana vamos a la playa."), (8, "Es un buen momento para hablar."), (9, "No tengo tiempo para eso."),
            (10, "Mi hermano vive en Lisboa."),
        ]
        response_rows = [
            (1, "Não sei o que fazer agora."), (2, "Vamos à casa da minha mãe."), (3, "Onde está o carro?"),
            (4, "Tenho que falar com ele."), (5, "Não é tão fácil como parece."), (6, "Ela não quer vir."),
            (7, "Amanhã vamos à praia."), (8, "É um bom momento para falar."), (9, "Não tenho tempo para isso."),
            (10, "O meu irmão vive em Lisboa."),
        ]

        eligible, untranslated = self.translator.find_untranslated_subtitles(tag(response_rows), tag(source_rows))
        self.assertEqual(eligible, 10)
        self.assertEqual(untranslated, [])

    @staticmethod
    def partial_copy(copied):
        source_rows = [(i, f"Bu cümle numara {i} için yazıldı.") for i in range(1, 11)]
        response_rows = [(i, f"Ez a {i}. számú mondat.") for i in range(1, 11)]
        response_rows[:copied] = source_rows[:copied]
        return source_rows, response_rows


if __name__ == "__main__":
    unittest.main()
