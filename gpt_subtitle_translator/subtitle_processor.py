import random
import re
from typing import NamedTuple

import srt

from gpt_subtitle_translator.models.base_model import BaseModel

class Chunk(NamedTuple):
    text: str
    num_tokens: int
    idx: int

class InvalidSRTFile(Exception):
    pass

class SubtitleProcessor:
    TAG_PATTERN = re.compile(r"^<(\d+)>(.*?)</\1>$", re.DOTALL | re.MULTILINE)
    OUT_TAG_PATTERN = re.compile(r"^<?(\d+)>?\n.+?\n<>(.*?)</>$", re.DOTALL | re.MULTILINE)

    def __init__(self, model: BaseModel):
        self.model = model

    @staticmethod
    def parse_srt(srt_content: str) -> dict:
        subtitles = {}
        try:
            for sub in srt.parse(srt_content):
                ts = f"{srt.timedelta_to_srt_timestamp(sub.start)} --> {srt.timedelta_to_srt_timestamp(sub.end)}"
                subtitles[sub.index] = {"timestamp": ts, "text": sub.content}
        except Exception as e:
            raise InvalidSRTFile("Invalid SRT file. Please check your input. " + str(e)[:1000])
        return subtitles

    def preprocess(self, srt_data):
        return "\n".join(f"<{key}>{value['text']}</{key}>" for key, value in srt_data.items())

    def make_chunks(self, text: str, max_tokens_per_chunk: int) -> list[Chunk]:
        items = self.split_on_tags(text)
        chunks = []
        current_piece = ""
        current_token_count = 0

        self.model.init_vocab(text)

        for text in items:
            text_token_count = self.model.num_tokens_from_string(text)
            candidate_length = current_token_count + text_token_count + 1
            if candidate_length <= max_tokens_per_chunk:
                current_piece += ("\n" + text)
                current_token_count = candidate_length
            else:
                chunks.append(
                    Chunk(text=current_piece, num_tokens=current_token_count, idx=len(chunks))
                )
                current_piece = text
                current_token_count = text_token_count

        chunks.append(Chunk(text=current_piece, num_tokens=current_token_count, idx=len(chunks)))
        return chunks

    def split_on_tags(self, text):
        return [match.group(0) for match in self.TAG_PATTERN.finditer(text)]

    def randomize_ids(self, subtitles):
        """
        Randomize subtitle IDs to avoid skipping/merging subtitles.

        Valid subtitles have consecutive numeric IDs, which seems to make GPT more likely to skip/merge neighboring subtitles.
        By assigning new IDs randomly, while preserving the order, we can help GPT avoid this behavior.
        """
        items = self.TAG_PATTERN.findall(subtitles)
        tag_count = len(items)
        new_ids = random.sample(range(1, tag_count * 10), tag_count)
        mapping = {}
        updated = []
        for index, (original_id, text) in enumerate(items):
            new_id = new_ids[index]
            mapping[original_id] = new_id
            updated.append(f"<{new_id}>{text}</{new_id}>")

        return "\n".join(updated), {v: k for k, v in mapping.items()}

    def reindex_ids(self, subtitles):
        """
        Re-index subtitle IDs sequentially
        """
        items = self.TAG_PATTERN.findall(subtitles)
        mapping = {}
        updated = []

        for index, (original_id, text) in enumerate(items):
            new_id = index + 1  # Start from 1
            mapping[original_id] = new_id
            updated.append(f"<{new_id}>{text}</{new_id}>")

        return "\n".join(updated), {v: k for k, v in mapping.items()}

    def shuffle_order(self, subtitles):
        items = self.TAG_PATTERN.findall(subtitles)
        random.shuffle(items)
        return "\n".join([f"<{id_}>{text}</{id_}>" for id_, text in items])

    def extract_subtitles(self, response, id_mapping):
        def replace_with_original(match):
            sub_id = int(match.group(1))
            original_id = id_mapping.get(sub_id, sub_id)
            return f"<{original_id}>{match.group(2)}</{original_id}>" if original_id else None

        tagged_texts = re.finditer(self.OUT_TAG_PATTERN, response)
        reverted = [replace_with_original(match) for match in tagged_texts]
        reverted = [x for x in reverted if x is not None]
        reverted.sort(key=lambda x: int(self.TAG_PATTERN.match(x).group(1)))
        return "\n".join(reverted)

    def post_process_text(self, text, original_subtitles):
        text = self.insert_timestamps(original_subtitles, text)
        text = re.sub(r'\n{3,}', "\n\n", text)
        text = text.strip()
        return self.clean_text(text)

    @staticmethod
    def clean_text(text: str) -> str:
        output_string = re.sub(r'[<>]\s*$', "", text, flags=re.MULTILINE)  # breaks subtitle parsing on Plex
        return output_string

    @staticmethod
    def insert_timestamps(parsed_subtitles, content):
        def replacement(match):
            subtitle_id = int(match.group(1))
            return f'\n{subtitle_id}\n{parsed_subtitles[subtitle_id]["timestamp"]}\n{match.group(2)}'
        return re.sub(r'^<(\d+)>(.*?)</\1>$', replacement, content, flags=re.DOTALL | re.MULTILINE)

    def get_missing_subtitles(self, translated, original_text):
        translated_ids = set(re.findall(r'^<(\d+)>', translated.strip(), flags=re.MULTILINE))
        original_entries = {id_: text for id_, text in self.TAG_PATTERN.findall(original_text.strip())}
        return {key: value for key, value in original_entries.items() if key not in translated_ids}