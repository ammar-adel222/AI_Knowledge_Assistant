# youtube_summarizer.py

from langchain_core.documents import Document
from langchain_core.documents import Document
from model_configration import Config
from typing import List


class YouTubeSummarizer:
    """
    Takes a YouTube transcript Document
    Summarizes it using LLaMA 8b
    Returns a clean summarized Document ready for RAG
    """

    def __init__(self):
        self.llm = Config.get_summarizer_llm()


        self.chunk_size = 2000


    # ──────────────────────────────────────────────
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split long transcript into chunks of ~2000 words
        so each chunk fits within the model token limit
        """

        words  = text.split()

        #  split text into individual words


        chunks = [
            " ".join(words[i : i + self.chunk_size])
            for i in range(0, len(words), self.chunk_size)
        ]
        #   words[0:2000]    first chunk
        #   words[2000:4000]  second chunk


        return chunks

    # ──────────────────────────────────────────────
    def _summarize_chunk(self, chunk: str, chunk_num: int, total: int) -> str:
        """
        Send one chunk to LLaMA 8b and get a summary back
        """

        print(f"   Summarizing chunk {chunk_num}/{total}...")

        prompt = f"""You are summarizing a section of a YouTube video transcript.
Write a clear and concise summary of the key points.
Focus on the main ideas, concepts, and important details.
Remove any filler words, repetition, or off-topic content.
Write in clear paragraphs.

Transcript section:
{chunk}

Summary:"""

        response = self.llm.invoke(prompt)


        return response.content




    def _combine_summaries(self, summaries: List[str]) -> str:
        """
        If we had multiple chunks → multiple summaries
        Combine them into ONE final coherent summary
        """


        if len(summaries) == 1:
            return summaries[0]

        print(f"   Combining {len(summaries)} summaries into one...")

        # join all summaries into one text
        combined = "\n\n".join(summaries)


        # send to LLaMA 8b one more time to merge them
        prompt = f"""You have multiple summaries from different sections of the same YouTube video.
Combine them into ONE coherent and well structured summary.
Keep all the key points and important details.
Remove any repetition between sections.
Write in clear paragraphs.

Section summaries:
{combined}

Final combined summary:"""

        response = self.llm.invoke(prompt)
        return response.content


    # ──────────────────────────────────────────────
    def summarize(self, document: Document) -> Document:
        """
        Main method:
        Takes a transcript Document
        Returns a summarized Document
        """

        print(f"\n YouTube Summarizer")


        text = document.page_content


        original_word_count = len(text.split())
        print(f"   Input     : {original_word_count} words")

        # Step 1: Split into chunks
        chunks = self._split_into_chunks(text)
        print(f"   Chunks    : {len(chunks)} x ~{self.chunk_size} words")

        # Step 2: Summarize each chunk
        summaries = []

        for i, chunk in enumerate(chunks):
            summary = self._summarize_chunk(
                chunk    = chunk,
                chunk_num= i + 1,
                total    = len(chunks)
            )
            summaries.append(summary)

        #  Step 3: Combine all summaries
        final_summary = self._combine_summaries(summaries)

        final_word_count = len(final_summary.split())
        compression      = round((1 - final_word_count / original_word_count) * 100)


        print(f"   Output    : {final_word_count} words")
        print(f"   Compression: {compression}% smaller")
        print(f"   Summarization complete!")

        # Step 4: Return new Document
        summarized_doc = Document(
            page_content = final_summary,
            metadata     = {
                **document.metadata,

                #  keep ALL original metadata
                #  (source_type, video_id, url, language, translated)
                #  ** spreads the dict into key=value pairs

                "summarized"          : True,
                "original_word_count" : original_word_count,
                "summary_word_count"  : final_word_count,
                "compression_percent" : compression,
                "summarizer_model"    : "llama3-8b-8192",
            }
        )

        return summarized_doc

