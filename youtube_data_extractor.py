from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)
import re
from model_configration import Config
from langchain_groq import ChatGroq
from typing import List
from langchain_core.documents import Document


class YouTubeExtractor:
    """
    Extracts transcript from a YouTube video.
    Translates to English using LLaMA 70b if needed.
    Returns the full transcript as one LangChain Document.
    """

    def _get_video_id(self, url: str) -> str:

        # List of URL patterns
        patterns = [
            r'(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]{11})',

            r'(?:youtu\.be\/)([a-zA-Z0-9_-]{11})',

            r'(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',

            r'(?:youtube\.com\/v\/)([a-zA-Z0-9_-]{11})',

            r'^([a-zA-Z0-9_-]{11})$',

        ]

        # ── Try each pattern until one works ──────
        for pattern in patterns:
            match = re.search(pattern, url)

            # re.search looks for the pattern

            if match:
                video_id = match.group(1)

                # group(1) gets what was inside the () in our pattern

                print(f"   Video ID : {video_id}")
                return video_id

        # ── No pattern matched → bad URL ──────────
        raise ValueError(
            f"Could not extract video ID from: {url}\n"
            f"Make sure it's a valid YouTube URL"
        )

    def _get_transcript(self, video_id: str) -> str:

        print(f"   Fetching transcript...")
        api = YouTubeTranscriptApi()

        try:
            # Attempt 1: english captions
            fetched = api.fetch(video_id, languages=["en"])
            language = "en"
            print(f"   Language : English (manual)")

        except NoTranscriptFound:

            try:
                # Attempt 2: auto-generated
                  # ← create object first
                transcript_list = api.list(video_id)
                fetched = transcript_list.find_generated_transcript(["en"]).fetch()
                language = "en"
                print(f"   Language : English (auto-generated)")

            except NoTranscriptFound:
                # no English at all so we grab whatever language exists
                try:
                    # Attempt 3: Any language
                    transcript_list = api.list(video_id)
                    first = next(iter(transcript_list))
                    fetched = first.fetch()
                    language = first.language_code
                    print(f"   Language : {language} (non-english)")

                except Exception as e:
                    raise ValueError(f"No transcript available: {e}")

        # Stitch chunks into one text
        text = " ".join(snippet.text for snippet in fetched)


        return text, language
        # transcript_list looks like:
        #   {"text": "hello",   "start": 0.0, "duration": 1.5},
        #   so we merge the text chunks together




    def _translate_to_english(self, text: str) -> str:
        """
        Translate non-English text to English

        """


        print(f"\n   Translating to English ...")

        llm = Config.get_qa_llm()


        # Split into chunks
        words= text.split()
        chunk_size = 1000
        # 1000 words per chunk
        # well within the model token limit
        # leaves room for the prompt itself

        chunks = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

        print(f"   Chunks    : {len(chunks)} × ~{chunk_size} words")

        # ── Translate each chunk ──────────────────
        translated_chunks = []

        for i, chunk in enumerate(chunks):
            print(f"   Translating chunk {i+1}/{len(chunks)}...", end="\r")

            prompt = f"""Translate the following text to English.
                       Only return the translated text, nothing else.
                       Do not add any explanation or comments.

                       Text to translate:
                       {chunk}

                       English translation:"""

            response = llm.invoke(prompt)
            translated_chunks.append(response.content)

        #Join translated chunks
        full_translation = " ".join(translated_chunks)

        print(f"\n    Translation complete!")
        print(f"   Characters : {len(full_translation)}")

        return full_translation


    def _clean_text(self, text: str) -> str:
        text = re.sub(r'$$.*?$$', '', text)  # remove [Music] tags
        text = re.sub(r'\s+', ' ', text)      # remove extra spaces
        return text.strip()


    def extract(self, url: str) -> List[Document]:

        print(f"\n YouTube Extractor")
        print(f"   URL : {url}")

        try:
            #  1: Get video id
            video_id = self._get_video_id(url)

            #  2: Get transcript + language
            raw_text, language = self._get_transcript(video_id)


            #  3: Translate if not English
            if language != "en":
                print(f"     Non-English transcript detected: {language}")
                raw_text = self._translate_to_english(raw_text)
                # raw_text is now replaced with the English version
            else:
                print(f"    Already English — no translation needed")

            #  Clean the text
            clean_text = self._clean_text(raw_text)

        except TranscriptsDisabled:
            raise ValueError(f" Transcripts disabled for: {url}")
        except VideoUnavailable:
            raise ValueError(f" Video unavailable: {url}")

        print(f"   Characters : {len(clean_text)}")
        print(f"   Words      : {len(clean_text.split())}")

        # ── Step 5: Create Document ────────────────
        document = Document(
            page_content = clean_text,
            metadata     = {
                "source_type"      : "youtube",
                "video_id"         : video_id,
                "url"              : url,
                "original_language": language,
                "translated"       : language != "en",
                # translated True :was translated to English
                # translated False :was already English
            }
        )

        print(f" Done — transcript ready")
        return [document]


