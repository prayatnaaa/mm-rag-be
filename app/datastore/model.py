import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import errorcode
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel


load_dotenv()

DB_CONFIG = {
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'host': os.getenv('MYSQL_HOST'),
    'database': 'chat_application_db'
}

class ChatCreateRequest(BaseModel):
    """Request model for creating a new chat."""
    topic: str

class ChatCreateResponse(BaseModel):
    """Response model after successfully creating a chat."""
    chat_id: int
    topic: str

class Message(BaseModel):
    """Response model for a single message."""
    content: str
    sender: str
    created_at: datetime

class Source(BaseModel):
    """Response model for a single source."""
    source_name: str
    source_url: Optional[str] = None
    retrieved_at: datetime

class ChatHistoryResponse(BaseModel):
    """The full response model for a chat's history."""
    id: int
    topic: str
    created_at: datetime
    messages: List[Message]
    sources: List[Source]

def get_db_connection():
    """Establishes and returns a database connection."""
    try:
        cnx = mysql.connector.connect(**DB_CONFIG)
        return cnx
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("DB Error: Access denied. Check your username or password.")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print(f"DB Error: Database '{DB_CONFIG['database']}' does not exist.")
        else:
            print(f"DB Error: {err}")
        return None

def create_new_chat(topic: str) -> Optional[int]:
    """
    Creates a new chat record in the database.

    Args:
        topic (str): The initial topic for the chat, usually the first query.

    Returns:
        Optional[int]: The ID of the newly created chat, or None if creation fails.
    """
    cnx = get_db_connection()
    if not cnx:
        return None

    cursor = cnx.cursor()
    chat_id = None
    try:
        add_chat_query = "INSERT INTO chat (topic) VALUES (%s)"
        cursor.execute(add_chat_query, (topic[:255],))
        chat_id = cursor.lastrowid
        cnx.commit()
        print(f"Created new chat with ID: {chat_id} and Topic: '{topic[:50]}...'")
    except mysql.connector.Error as err:
        print(f"Database Error: Failed to create new chat. {err}")
        cnx.rollback()
    finally:
        cursor.close()
        cnx.close()
        
    return chat_id

def get_chat_history(chat_id: int) -> Optional[Dict]:
    """
    Retrieves the full history for a given chat_id, including messages and sources.

    Args:
        chat_id (int): The ID of the chat to retrieve.

    Returns:
        Optional[Dict]: A dictionary containing chat details, messages, and sources,
                        or None if the chat is not found.
    """
    cnx = get_db_connection()
    if not cnx:
        return None

    history = None
    # Use a dictionary cursor to get column names in the result
    cursor = cnx.cursor(dictionary=True)
    try:
        # 1. Get the main chat info
        cursor.execute("SELECT id, topic, created_at FROM chat WHERE id = %s", (chat_id,))
        chat_info = cursor.fetchone()

        if not chat_info:
            return None # Chat not found

        # 2. Get all messages for the chat, ordered by creation time
        cursor.execute(
            "SELECT content, sender, created_at FROM message WHERE chat_id = %s ORDER BY created_at ASC",
            (chat_id,)
        )
        messages = cursor.fetchall()

        # 3. Get all sources for the chat
        cursor.execute(
            "SELECT source_name, source_url, retrieved_at FROM source WHERE chat_id = %s",
            (chat_id,)
        )
        sources = cursor.fetchall()

        # 4. Assemble the final history object
        history = {
            "id": chat_info["id"],
            "topic": chat_info["topic"],
            "created_at": chat_info["created_at"],
            "messages": messages,
            "sources": sources
        }

    except mysql.connector.Error as err:
        print(f"Database Error: Failed to get chat history for ID {chat_id}. {err}")
    finally:
        cursor.close()
        cnx.close()
        
    return history

def save_chat_contents(chat_id: int, query: str, answer: str, used_contexts: List[Dict]):
    """
    Saves the messages (user query, assistant answer) and sources for a given chat_id.

    Args:
        chat_id (int): The ID of the chat to save contents to.
        query (str): The user's query.
        answer (str): The assistant's answer.
        used_contexts (List[Dict]): The list of sources used for the answer.
    """
    cnx = get_db_connection()
    if not cnx:
        print("Could not save contents: No database connection.")
        return

    cursor = cnx.cursor()
    try:
        messages_to_add = [
            (chat_id, query, 'user'),
            (chat_id, answer, 'assistant')
        ]
        add_message_query = "INSERT INTO message (chat_id, content, sender) VALUES (%s, %s, %s)"
        cursor.executemany(add_message_query, messages_to_add)
        print(f"Saved {len(messages_to_add)} messages to chat {chat_id}.")

        if used_contexts:
            sources_to_add = []
            add_source_query = "INSERT INTO source (chat_id, source_name, source_url) VALUES (%s, %s, %s)"
            
            for context in used_contexts:
                meta = context.get('metadata', {})
                source_name = meta.get('source_id', meta.get('title', 'unknown_source'))
                source_url = meta.get('image_url', meta.get('youtube_url'))
                sources_to_add.append((chat_id, source_name, source_url))

            if sources_to_add:
                cursor.executemany(add_source_query, sources_to_add)
                print(f"Saved {len(sources_to_add)} context sources to chat {chat_id}.")

        cnx.commit()
        print(f"Successfully saved contents for chat session {chat_id}.")

    except mysql.connector.Error as err:
        print(f"Database transaction failed for chat {chat_id}: {err}")
        cnx.rollback()
    finally:
        cursor.close()
        cnx.close()