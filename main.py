# main.py

from data_reader              import data_extractor
from youtube_data_extractor   import YouTubeExtractor
from youtube_summarizer       import YouTubeSummarizer
from Rag__pipline             import Chunker, Embedder, VectorStore
from chat                     import ChatEngine, format_answer
from langchain_core.documents import Document
from typing                   import List
from model_configration import Config


class DataLoader:
    """
    Interactively asks the user for data sources.
    Collects all documents into one list.
    """

    def __init__(self):
        self.extractor  = data_extractor()

        self.youtube_extractor = YouTubeExtractor()
        self.youtube_summarizer= YouTubeSummarizer()
        self.all_documents     : List[Document] = []

    # ──────────────────────────────────────────────
    def _load_pdf(self) -> None:
        """Ask for PDF path and load it"""

        path = input("\n   📄 Enter PDF file path: ").strip()

        try:
            docs = self.extractor.extract_pdf(path)
            self.all_documents.extend(docs)
            #                   ↑
            #    extend adds ALL docs to the list
            #    (not as a nested list)
            #
            #    append → [[doc1, doc2], [doc3]]  ❌
            #    extend → [doc1, doc2, doc3]       ✅

            print(f"   ✅ Added {len(docs)} pages from PDF")

        except FileNotFoundError:
            print(f"   ❌ File not found: {path}")
        except ValueError as e:
            print(f"   ❌ Error: {e}")


    # ──────────────────────────────────────────────
    def _load_sheet(self) -> None:
        """Ask for Excel/CSV path and load it"""

        path = input("\n   📊 Enter Excel/CSV file path: ").strip()

        try:
            docs = self.extractor.extract_spreadsheet(path)
            self.all_documents.extend(docs)
            print(f"   ✅ Added {len(docs)} rows from spreadsheet")

        except FileNotFoundError:
            print(f"   ❌ File not found: {path}")
        except ValueError as e:
            print(f"   ❌ Error: {e}")


    # ──────────────────────────────────────────────
    def _load_youtube(self) -> None:
        """Ask for YouTube URL, extract + summarize"""

        url = input("\n   🎥 Enter YouTube URL: ").strip()

        try:

            print(f"\n   Extracting transcript...")
            docs = self.youtube_extractor.extract(url)

            print(f"\n   Summarizing transcript...")
            summarized = self.youtube_summarizer.summarize(docs[0])

            self.all_documents.append(summarized)

            print(f"   ✅ Added YouTube video summary")
            print(f"      Original : {summarized.metadata['original_word_count']} words")
            print(f"      Summary  : {summarized.metadata['summary_word_count']} words")
            print(f"      Reduced  : {summarized.metadata['compression_percent']}%")

        except ValueError as e:
            print(f"   ❌ Error: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")


    # ──────────────────────────────────────────────
    def _show_loaded(self) -> None:
        """Show what has been loaded so far"""

        if not self.all_documents:
            print("\n   No sources loaded yet")
            return

        # count by source type
        counts = {}
        for doc in self.all_documents:
            source_type = doc.metadata.get("source_type", "unknown")
            counts[source_type] = counts.get(source_type, 0) + 1
            #                              ↑
            #                   .get(key, default)
            #                   if key not in dict → return 0
            #                   then add 1

        print(f"\n   📚 Currently loaded:")
        print(f"   {'─'*30}")
        for source_type, count in counts.items():
            icon = {"pdf": "📄", "spreadsheet": "📊", "youtube": "🎥"}.get(source_type, "📁")
            print(f"   {icon} {source_type:15} : {count} documents")
        print(f"   {'─'*30}")
        print(f"   Total : {len(self.all_documents)} documents")


    # ──────────────────────────────────────────────
    def run(self) -> List[Document]:
        """
        Main interactive loop.
        Keeps asking user to add sources
        until they type 'done'
        Returns all collected documents
        """

        print(f"\n{'─'*50}")
        print("  📥 Data Source Loader")
        print(f"{'─'*50}")
        print("  Add as many sources as you want.")
        print("  Type 'done' when finished.\n")

        while True:

            print(f"\n  What would you like to add?")
            print(f"  [1] 📄 PDF file")
            print(f"  [2] 📊 Excel / CSV file")
            print(f"  [3] 🎥 YouTube video")
            print(f"  [4] 📚 Show loaded sources")
            print(f"  [5] ✅ Done — build knowledge base")

            choice = input("\n  Your choice (1/2/3/4/5): ").strip()

            if choice == "1":
                self._load_pdf()

            elif choice == "2":
                self._load_sheet()

            elif choice == "3":
                self._load_youtube()

            elif choice == "4":
                self._show_loaded()

            elif choice == "5":
                # user is done adding sources
                if not self.all_documents:
                    print("\n  ⚠️  No sources loaded!")
                    print("  Please add at least one source first.")
                    continue
                    # go back to the menu
                    # don't let them proceed with nothing

                print(f"\n  ✅ Done loading!")
                self._show_loaded()
                return self.all_documents
                # return all collected documents
                # to be chunked + embedded

            else:
                print("  ⚠️  Invalid choice — type 1, 2, 3, 4 or 5")


# ══════════════════════════════════════════════════════════
#   MAIN — ties everything together
# ══════════════════════════════════════════════════════════

def main():

    print("=" * 50)
    print("  🤖 AI Knowledge Assistant")
    print("=" * 50)

    # ── Ask user: load new or use existing ────────
    embedder = Embedder()
    store    = VectorStore()

    if store.is_saved():
        print(f"\n📂 Found existing knowledge base!")
        print(f"   [1] Load existing knowledge base")
        print(f"   [2] Build a new one (replaces existing)")
        choice = input("\n   Your choice (1/2): ").strip()

        if choice == "1":
            # just load what we saved before
            store.load(embedder)
            print("✅ Knowledge base loaded!")

        elif choice == "2":
            # build fresh
            documents = DataLoader().run()
            _build_store(documents, embedder, store)

        else:
            print("⚠️  Invalid choice — loading existing")
            store.load(embedder)

    else:
        # no saved store → must build
        print("\n📭 No existing knowledge base found.")
        print("   Let's build one!\n")
        documents = DataLoader().run()
        _build_store(documents, embedder, store)

    # ── Launch chat ────────────────────────────────
    engine = ChatEngine(store)

    print(f"\n{'─'*50}")
    print("  Commands during chat:")
    print("  'exit'    → quit")
    print("  'clear'   → clear history")
    print("  'history' → show history")
    print(f"{'─'*50}\n")

    # Q&A loop
    while True:

        question = input("❓ You: ").strip()

        if not question:
            print("   Please type a question")
            continue

        elif question.lower() == "exit":
            print("\n👋 Goodbye!")
            break

        elif question.lower() == "clear":
            engine.cache.clear()

        elif question.lower() == "history":
            engine.cache.show()

        else:
            answer = engine.chat(question)
            format_answer(answer)


# ──────────────────────────────────────────────────────────
def _build_store(
    documents: List[Document],
    embedder : Embedder,
    store    : VectorStore
) -> None:
    """Chunk + embed + save the vector store"""

    # chunk all documents
    chunker = Chunker(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
    chunks  = chunker.split(documents)

    # embed + build FAISS index
    store.build(chunks, embedder)

    # save to disk
    store.save()

    print(f"\n✅ Knowledge base built and saved!")
    print(f"   Total chunks : {len(chunks)}")


# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()