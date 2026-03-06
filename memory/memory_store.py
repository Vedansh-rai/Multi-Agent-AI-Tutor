"""
Memory Store for Multimodal Math Mentor.
Provides persistent storage (SQLite) and vector-based similarity search
for past interactions, enabling self-learning from corrections.
"""

import os
import json
import sqlite3
from datetime import datetime
from utils.config import SQLITE_DB_PATH, CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME
from utils.logger import get_logger

logger = get_logger("memory.store")


class MemoryStore:
    """
    Dual-layer memory system:
      1. SQLite — structured storage of all interactions.
      2. ChromaDB — vector embeddings for similarity search.
    """

    def __init__(self):
        os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
        self.conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self._memory_collection = None

    def _init_db(self):
        """Create the interactions table if it doesn't exist."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_type TEXT NOT NULL,
                raw_input TEXT NOT NULL,
                parsed_problem TEXT,
                topic TEXT,
                retrieved_context TEXT,
                solution_steps TEXT,
                final_answer TEXT,
                explanation TEXT,
                confidence REAL,
                user_feedback TEXT,
                feedback_comment TEXT,
                corrected_answer TEXT
            )
            """
        )
        self.conn.commit()
        logger.info("Memory database initialized at %s", SQLITE_DB_PATH)

    def _get_memory_collection(self):
        """Lazy-init a ChromaDB collection for memory embeddings."""
        if self._memory_collection is None:
            try:
                import chromadb
                from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

                abs_persist = os.path.abspath(CHROMA_PERSIST_DIR)
                client = chromadb.PersistentClient(path=abs_persist)
                embedding_fn = DefaultEmbeddingFunction()
                self._memory_collection = client.get_or_create_collection(
                    name="memory_interactions",
                    embedding_function=embedding_fn,
                )
            except Exception as e:
                logger.error("Failed to init memory vector collection: %s", e)
        return self._memory_collection

    def save_interaction(
        self,
        input_type: str,
        raw_input: str,
        parsed_problem: str = "",
        topic: str = "",
        retrieved_context: str = "",
        solution_steps: str = "",
        final_answer: str = "",
        explanation: str = "",
        confidence: float = 0.0,
        user_feedback: str = "",
        feedback_comment: str = "",
        corrected_answer: str = "",
    ) -> int:
        """
        Save an interaction to both SQLite and vector memory.

        Returns:
            The row ID of the saved interaction.
        """
        timestamp = datetime.now().isoformat()

        cursor = self.conn.execute(
            """
            INSERT INTO interactions
            (timestamp, input_type, raw_input, parsed_problem, topic,
             retrieved_context, solution_steps, final_answer, explanation,
             confidence, user_feedback, feedback_comment, corrected_answer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                input_type,
                raw_input,
                parsed_problem,
                topic,
                retrieved_context,
                solution_steps,
                final_answer,
                explanation,
                confidence,
                user_feedback,
                feedback_comment,
                corrected_answer,
            ),
        )
        self.conn.commit()
        row_id = cursor.lastrowid

        # Also store in vector DB for similarity search
        collection = self._get_memory_collection()
        if collection is not None:
            doc_text = f"{parsed_problem} | {final_answer}"
            try:
                collection.upsert(
                    ids=[f"interaction_{row_id}"],
                    documents=[doc_text],
                    metadatas=[
                        {
                            "row_id": row_id,
                            "topic": topic,
                            "confidence": confidence,
                            "feedback": user_feedback,
                            "timestamp": timestamp,
                        }
                    ],
                )
            except Exception as e:
                logger.warning("Failed to store in vector memory: %s", e)

        logger.info("Saved interaction #%d (type=%s, topic=%s)", row_id, input_type, topic)
        return row_id

    def get_similar_problems(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Find similar past problems using vector similarity.

        Args:
            query: The current problem text.
            top_k: Number of similar problems to retrieve.

        Returns:
            List of dicts with past interaction details.
        """
        collection = self._get_memory_collection()
        if collection is None:
            return []

        try:
            count = collection.count()
            if count == 0:
                return []

            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, count),
            )
        except Exception as e:
            logger.error("Similar problem search failed: %s", e)
            return []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        similar = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            row_id = meta.get("row_id")
            # Fetch full details from SQLite
            row = self.conn.execute(
                "SELECT * FROM interactions WHERE id = ?", (row_id,)
            ).fetchone()
            if row:
                similar.append(
                    {
                        "problem": row["parsed_problem"],
                        "answer": row["final_answer"],
                        "explanation": row["explanation"],
                        "confidence": row["confidence"],
                        "feedback": row["user_feedback"],
                        "corrected_answer": row["corrected_answer"],
                        "similarity_score": round(1.0 / (1.0 + dist), 3),
                    }
                )

        logger.info("Found %d similar problems for query: %s", len(similar), query[:60])
        return similar

    def apply_correction(
        self, interaction_id: int, feedback: str, comment: str, corrected_answer: str
    ) -> bool:
        """
        Store user correction for self-learning.

        Args:
            interaction_id: ID of the interaction to correct.
            feedback: 'correct' or 'incorrect'.
            comment: User's comment about the error.
            corrected_answer: The correct answer provided by user.

        Returns:
            True if update succeeded.
        """
        try:
            self.conn.execute(
                """
                UPDATE interactions
                SET user_feedback = ?, feedback_comment = ?, corrected_answer = ?
                WHERE id = ?
                """,
                (feedback, comment, corrected_answer, interaction_id),
            )
            self.conn.commit()

            # Update vector metadata
            collection = self._get_memory_collection()
            if collection:
                try:
                    row = self.conn.execute(
                        "SELECT parsed_problem FROM interactions WHERE id = ?",
                        (interaction_id,),
                    ).fetchone()
                    if row:
                        doc_text = f"{row['parsed_problem']} | {corrected_answer or ''}"
                        collection.upsert(
                            ids=[f"interaction_{interaction_id}"],
                            documents=[doc_text],
                            metadatas=[
                                {
                                    "row_id": interaction_id,
                                    "feedback": feedback,
                                    "corrected": True,
                                }
                            ],
                        )
                except Exception as e:
                    logger.warning("Failed to update vector memory: %s", e)

            logger.info("Applied correction to interaction #%d", interaction_id)
            return True
        except Exception as e:
            logger.error("Failed to apply correction: %s", e)
            return False

    def get_history(self, limit: int = 20) -> list[dict]:
        """Get recent interaction history."""
        rows = self.conn.execute(
            "SELECT * FROM interactions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self):
        """Close the database connection."""
        self.conn.close()
