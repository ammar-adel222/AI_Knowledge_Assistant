# chat.py

from Rag__pipline import Embedder, VectorStore
from model_configration import Config
from langchain_core.documents import Document
from typing import List

class QAChain:
    """
    Searches the vector store
    Builds a prompt with retrieved chunks
    Gets answer from LLaMA 70b
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm          = Config.get_qa_llm()
        self.k            = 4


    def _build_prompt(self, question: str, chunks: List[Document]) -> str:

        context_parts = []

        for i, chunk in enumerate(chunks):
            source_type = chunk.metadata.get("source_type", "unknown")

            if source_type == "pdf":
                source_info = (f"PDF: {chunk.metadata.get('file_name','?')} "
                               f"page {chunk.metadata.get('page','?')}")
            elif source_type == "spreadsheet":
                source_info = (f"Spreadsheet: {chunk.metadata.get('file_name','?')} "
                               f"row {chunk.metadata.get('row_number','?')}")
            elif source_type == "youtube":
                source_info = f"YouTube: {chunk.metadata.get('video_id','?')}"
            else:
                source_info = source_type

            context_parts.append(
                f"[Source {i+1} — {source_info}]\n{chunk.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)



    def search(self, question: str) -> List[Document]:
        """Search vector store — returns relevant chunks"""
        return self.vector_store.search(question, k=self.k)



class ConversationCache:
    """
    Stores the conversation history.
    Simple list of question/answer pairs.
    Passed to LLaMA 70b so it remembers context.
    """

    def __init__(self, max_history: int = 5):

        self.history = []

        self.max_history = max_history

        #    5 exchanges = 10 messages (5 user + 5 assistant)




    # ──────────────────────────────────────────────
    def add(self, question: str, answer: str) -> None:

        self.history.append({
            "role"   : "user",
            "content": question
        })
        self.history.append({
            "role"   : "assistant",
            "content": answer
        })

        max_messages = self.max_history * 2
        #              ↑
        #   each exchange = 2 messages (user + assistant)
        #   5 exchanges   = 10 messages

        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

            #  oldest ones get dropped
            #  [-10:] = last 10 messages


    def get_history_text(self) -> str:
        """
        Format history as readable text
        to inject into the prompt
        """

        if not self.history:
            return ""


        lines = []

        for message in self.history:
            if message["role"] == "user":
                lines.append(f"User      : {message['content']}")
            else:
                lines.append(f"Assistant : {message['content']}")

        return "\n".join(lines)



    def clear(self) -> None:
        """Clear all history — fresh start"""
        self.history = []



    def show(self) -> None:
        """Print the full conversation history"""

        if not self.history:
            print("   No conversation history yet")
            return

        print(f"\n{'─'*50}")
        print(f"📜 Conversation History ({len(self.history)//2} exchanges):")
        print(f"{'─'*50}")

        for i, message in enumerate(self.history):
            role = "❓ You" if message["role"] == "user" else "🤖 AI "
            print(f"{role} : {message['content'][:100]}...")
            #                                    ↑
            #                    only show first 100 chars
            #                    keeps it readable

        print(f"{'─'*50}")


# ══════════════════════════════════════════════════════════
#   CHAT ENGINE — QA + MEMORY COMBINED
# ══════════════════════════════════════════════════════════
class ChatEngine:

    def __init__(self, vector_store: VectorStore):
        self.qa_chain = QAChain(vector_store)
        #    ↑
        #    QAChain is NOW defined in this same file
        #    no cross-file dependency ✅

        self.cache    = ConversationCache(max_history=5)
        self.llm      = Config.get_qa_llm()
        print(f"\n🚀 Chat Engine ready!")


    def _build_full_prompt(
        self,
        question : str,
        chunks   : List[Document]
    ) -> str:

        context      = self.qa_chain._build_prompt(question, chunks)
        history_text = self.cache.get_history_text()

        if history_text:
            return f"""You are a helpful AI assistant with access to a knowledge base.
Answer using ONLY the context provided.
If the answer is not in the context say "I don't have enough information."
Always mention which source your answer comes from.

KNOWLEDGE BASE CONTEXT:
{context}

CONVERSATION HISTORY:
{history_text}

CURRENT QUESTION:
{question}

ANSWER:"""

        else:
            return f"""You are a helpful AI assistant with access to a knowledge base.
Answer using ONLY the context provided.
If the answer is not in the context say "I don't have enough information."
Always mention which source your answer comes from.

KNOWLEDGE BASE CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


    def chat(self, question: str) -> str:

        # search knowledge base
        chunks = self.qa_chain.search(question)

        if not chunks:
            answer = "I don't have enough information to answer this."
            self.cache.add(question, answer)
            return answer

        # build prompt with memory + context
        prompt = self._build_full_prompt(question, chunks)

        # ask LLaMA 70b
        response = self.llm.invoke(prompt)
        answer   = response.content

        # save to cache
        self.cache.add(question, answer)

        return answer


