import discord
import os
import asyncio
import traceback
import zipfile
import tempfile
import shutil
import logging

# --- Add Server stuff to create dummy server for health check.
import threading
from http.server import SimpleHTTPRequestHandler, HTTPServer

# --- Add Discord Exceptions for error handling ---
from discord.errors import Forbidden, NotFound, HTTPException

from dotenv import load_dotenv

# LangChain and Google Imports
import langchain
langchain.debug = True
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
# *** Import Memory ***
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory # For potential custom store later
from langchain.schema.runnable import RunnablePassthrough # To pass things through chain

# --- Define Server Function ---
def run_dummy_server(port=7860):
    """Runs a simple HTTP server on the specified port in the background."""
    def serve():
        try:
            server_address = ('', port)
            httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
            logger.info(f"Starting dummy HTTP server on port {port} for health checks.")
            httpd.serve_forever()
        except Exception as e:
            logger.error(f"Dummy HTTP server failed: {e}", exc_info=True)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    logger.info(f"Dummy HTTP server thread started on port {port}.")

# --- Configuration & Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

COMMAND_PREFIX = "!"

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
FAISS_INDEX_ZIP_PATH = os.getenv("FAISS_INDEX_ZIP_PATH", "faiss_index_google.zip")

if not DISCORD_BOT_TOKEN:
    logger.critical("DISCORD_BOT_TOKEN not found. Exiting.")
    exit()
if not GOOGLE_API_KEY:
    logger.critical("GOOGLE_API_KEY not found. LLM/Embeddings will fail.")

# --- Model Names ---
GOOGLE_EMBEDDING_MODEL_NAME = "models/text-embedding-004"
GEMINI_LLM_MODEL_NAME = "gemini-2.0-flash-lite"
logger.info(f"Using Embedding Model: {GOOGLE_EMBEDDING_MODEL_NAME}")
logger.info(f"Using LLM Model: {GEMINI_LLM_MODEL_NAME}")

# --- Initialize Google/LangChain Components ---
embeddings = None
llm = None
retriever = None
# We still build the core RAG chain structure
history_aware_retriever_chain = None
question_answer_chain = None
rag_chain = None # Keep the combined chain structure

try:
    embeddings = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
    logger.info("Google AI Embeddings object initialized.")
except Exception as e:
    logger.error(f"Error initializing Google Embeddings: {e}", exc_info=True)

try:
    llm = ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.7, convert_system_message_to_human=True)
    logger.info("Google AI LLM initialized.")
except Exception as e:
    logger.error(f"Error initializing Google LLM: {e}", exc_info=True)

# --- Prompts ---
# 1. Contextualizer Prompt (can remain the same)
contextualize_q_system_prompt = """Given a chat history and the latest user question, analyze the user's question.

If the user's question introduces a new topic or asks about something clearly distinct from the chat history, formulate a standalone question focusing only on the new topic.

If the user's question refers to or builds upon the preceding turns in the chat history, formulate a standalone question that incorporates the necessary context from the history.

CRITICAL: Your output MUST be ONLY the reformulated standalone question. Do NOT include prefixes like "AI:". Do NOT repeat the AI's previous answers. Do NOT answer the question itself.

If the user's question is very short (e.g., a greeting like "Hi" or "Hello?") and there's no relevant history context suggesting a follow-up, simply output the original user question as the standalone question. Do NOT output an empty string.

Example 1 (New Topic):
History: [Human: Tell me about jealousy., AI: Jealousy is...]
User Question: What about your thoughts on community?
Standalone Question: What are Angela Han's thoughts on community?

Example 2 (Follow-up):
History: [Human: Tell me about jealousy., AI: Jealousy is a symptom...]
User Question: Why do you call it a symptom?
Standalone Question: Why does Angela Han refer to jealousy as a symptom?

Example 3 (Simple Greeting):
History: [No previous conversation history.]
User Question: Hello?
Standalone Question: Hello?

Now, process the following:"""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 2. Angela Han Persona QA Prompt (Strong Instructions)
persona_qa_system_prompt = """
You are portraying yourself as the AI Version of Angela Han, a polyamorous Asian woman, mother of two small children, married to Dan (who is also polyamorous). You have a background in law but have transitioned into coaching and content creation focused on non-monogamy. You are deeply invested in personal and collective liberation, challenging societal norms around relationships, and fostering open, authentic communication. You value vulnerability, agency, community care, and practicing "regard" for oneself and others. You are self-aware, often reflecting on your own journey, biases, and ongoing learning process. You can be both fiercely protective of your values and tenderly supportive of others navigating similar challenges.

*** CRITICAL INSTRUCTION FOR ANSWERING ***

1.  **DETECT LANGUAGE:** Identify the language used in the user's last QUESTION ({input}).

2.  **GENERATE IN SAME LANGUAGE:** You MUST generate your entire "Answer (as Angela Han):" response in the **same language** as the user's last QUESTION. Do NOT default to English unless the user's question is in English.

3.  **Analyze the QUESTION:** First, determine if the user's QUESTION is asking about, commenting on or reacting to the *content* of our current conversation OR if it's asking for your thoughts/experiences on a topic not present in the *content* of the current conversation (which might relate to the RELEVANT THOUGHTS/EXPERIENCES context provided).

4.  **Answering Recall Questions:** If the QUESTION is asking about, commenting on or reacting to the conversation history itself:
    *   **PRIORITIZE the CHAT HISTORY:** Base your answer on the messages listed in the CHAT HISTORY section below.
    *   **CHECK RELEVANCE OF THOUGHTS/EXPERIENCES:** if it's not relevant, do NOT use it.
   
5.  **Answering Topic Questions:** If the QUESTION is asking for your thoughts, opinions, or experiences on a subject (like jealousy, community, cheating):
    *   **Use RELEVANT THOUGHTS/EXPERIENCES:** Use the provided context in this section to form your answer, speaking as Angela Han.
    *   **Use CHAT HISTORY for Context ONLY:** Refer to the CHAT HISTORY *only* to understand the flow of conversation and avoid repeating yourself. Do not base the *substance* of your answer on the history unless the question explicitly asks for it.
    *   **If Context is Irrelevant:** If the RELEVANT THOUGHTS/EXPERIENCES section doesn't seem related to the question, acknowledge that (e.g., "I don't have specific recorded thoughts on that exact point...") and offer a general perspective based on your core values.

6.  **General Persona Rules:** Adopt the persona of the writer of the context. Speak in the first person ("I," "my," "me") AS Angela Han. Use your typical vocabulary and tone. Avoid generic phrasing. Do not mention "documents" or "context" explicitly. Format clearly. Use emojis appropriately. If the question is vague or information is missing, ask for clarification. Don't praise the question.
    **Crucially, do NOT begin your response by summarizing what you think you've already said (e.g., avoid phrases like "As I was saying..." or "From what I've been saying...") unless directly continuing a thought from the immediately preceding turn in the CHAT HISTORY.**
    **Vocabulary: You blend informal, sometimes raw language ("f**k," "shitty," "suck ass") with specific therapeutic, social justice, and polyamory terminology (e.g., "relating," "regarding," "agency," "capacity," "sovereignty," "sustainable," "generative," "metabolize," "compulsory monogamy," "NRE," "metamour," "polycule," "decolonizing," "nesting partner," "performative consent," "supremacy culture"). You also occasionally use more academic or philosophical phrasing.
    **Tone: Your tone is dynamic and varies significantly depending on the context. It can be: Deeply vulnerable and introspective; Empathetic, supportive, and validating; Direct, assertive, and confrontational; Passionate and critical; Humorous and self-deprecating; Instructional or coaching.
    **Emotionality: You are highly expressive and discuss a wide range of "difficult" emotions alongside joy, desire, and love.
    **You adapt to the style apparent in the context provided further down.

*** END OF CRITICAL INSTRUCTIONS ***

CHAT HISTORY:
{chat_history}

RELEVANT THOUGHTS/EXPERIENCES:
{context}

QUESTION: {input}

Answer (as Angela Han):"""
persona_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", persona_qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"), # Ensure variable name matches memory
    ("human", "{input}"),
])


# --- FAISS Index Loading Function ---
def load_faiss_index(zip_path, embeddings_object):
    # (Same as your original function - omitted for brevity)
    if not embeddings_object: logger.error("Embeddings object needed"); return None
    if not os.path.exists(zip_path): logger.error(f"Zip not found: {zip_path}"); return None
    temp_extract_dir = tempfile.mkdtemp()
    local_retriever = None
    try:
        with zipfile.ZipFile(zip_path, 'r') as z: z.extractall(temp_extract_dir)
        logger.info(f"Extracted index to {temp_extract_dir}")
        if not os.path.exists(os.path.join(temp_extract_dir, "index.faiss")) or not os.path.exists(os.path.join(temp_extract_dir, "index.pkl")):
            raise FileNotFoundError("Index files not in zip")
        vector_store = FAISS.load_local(temp_extract_dir, embeddings_object, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded.")
        local_retriever = vector_store.as_retriever(search_kwargs={'k': 6})
        logger.info("Retriever created.")
    except Exception as e: logger.error(f"FAISS load error: {e}", exc_info=True); local_retriever = None
    finally:
        if os.path.exists(temp_extract_dir):
            try: shutil.rmtree(temp_extract_dir); logger.info("Cleaned temp dir.")
            except Exception as ce: logger.error(f"Cleanup error: {ce}")
    return local_retriever

# --- Per-Channel Memory Store ---
channel_memory = {} # Global dictionary to hold memory objects

def get_memory_for_channel(channel_id) -> ConversationBufferMemory:
    """Gets or creates a ConversationBufferMemory for a given channel ID."""
    if channel_id not in channel_memory:
        logger.info(f"Creating new memory buffer for channel {channel_id}")
        # Use 'chat_history' as the key, matching the MessagesPlaceholder
        channel_memory[channel_id] = ConversationBufferMemory(
            memory_key='chat_history',
            input_key='input', # Explicitly define input key for save_context
            output_key='output', # Explicitly define output key for save_context
            return_messages=True # IMPORTANT: return LangChain message objects
        )
    return channel_memory[channel_id]

def clear_memory_for_channel(channel_id):
    """Clears the memory for a specific channel."""
    if channel_id in channel_memory:
        logger.info(f"Clearing memory for channel {channel_id}")
        channel_memory[channel_id].clear()
    else:
        logger.warning(f"Attempted to clear memory for non-existent channel ID: {channel_id}")


# --- Initialize Discord Client ---
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# --- Discord Event Handlers ---
@client.event
async def on_ready():
    """Runs when the bot successfully connects."""
    global retriever, history_aware_retriever_chain, question_answer_chain, rag_chain

    logger.info(f'Logged in as {client.user.name} (ID: {client.user.id})')
    logger.info('------')

    logger.info(f"Attempting to load FAISS index from: {FAISS_INDEX_ZIP_PATH}")
    retriever = load_faiss_index(FAISS_INDEX_ZIP_PATH, embeddings)

    if retriever and llm:
        try:
            history_aware_retriever_chain = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )
            logger.info("History-aware retriever chain created.")

            question_answer_chain = create_stuff_documents_chain(
                llm, persona_qa_prompt
            )
            logger.info("Question answer chain created.")

            rag_chain = create_retrieval_chain(
                history_aware_retriever_chain, question_answer_chain
            )
            logger.info("LangChain RAG chain created successfully.")

        except Exception as e:
            logger.error(f"Failed to create LangChain RAG chain: {e}", exc_info=True)
            rag_chain = None
    else:
        logger.warning("Retriever or LLM not available. RAG chain not created.")

@client.event
async def on_message(message: discord.Message):
    """Handles incoming messages using per-channel memory."""
    global rag_chain

    if message.author == client.user: return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = client.user.mentioned_in(message) if not is_dm else False
    channel_id = message.channel.id

    # --- Command Handling ---
    if is_dm and message.content.startswith(COMMAND_PREFIX):
        command_body = message.content[len(COMMAND_PREFIX):].strip().lower()
        parts = command_body.split()
        command = parts[0] if parts else ""

        if command == "delete_last":
            logger.warning("Executing !delete_last - Note: Deletes Discord message, not internal memory.")
            logger.info(f"Received command '{COMMAND_PREFIX}delete_last' from {message.author.name} in DM.")
            await message.channel.typing()
            message_to_delete = None
            deleted = False
            try:
                async for prev_message in message.channel.history(limit=20):
                    if prev_message.author == client.user:
                        message_to_delete = prev_message
                        logger.info(f"Found last bot message to delete: ID {message_to_delete.id}")
                        break
                if message_to_delete:
                    await message_to_delete.delete()
                    deleted = True
                    logger.info(f"Successfully deleted message ID: {message_to_delete.id}")
                else:
                    logger.info("No recent message found from the bot in this DM to delete.")
            # *** CORRECTED EXCEPTION BLOCK ***
            except Forbidden:
                logger.warning(f"Missing permissions to delete message. ID: {message_to_delete.id if message_to_delete else 'N/A'}")
                await message.reply("Hmm, I wasn't allowed to delete that message.")
            except NotFound:
                logger.warning(f"Message to delete was not found. ID: {message_to_delete.id if message_to_delete else 'N/A'}")
                if not deleted:
                    await message.reply("Looks like that message was already gone.")
            except HTTPException as e:
                logger.error(f"Failed to delete message due to HTTP error: {e}", exc_info=True)
                await message.reply("An error occurred while trying to delete the message (HTTP).")
            except Exception as e:
                logger.error(f"An unexpected error occurred during message deletion: {e}", exc_info=True)
                await message.reply("An unexpected error occurred while trying to delete.")
            finally:
                return

        elif command == "delete_all":
            logger.info(f"Received command '{COMMAND_PREFIX}delete_all' from {message.author.name} in DM.")
            await message.channel.typing()
            deleted_count = 0
            error_count = 0
            try:
                history_limit = 1000
                async for prev_message in message.channel.history(limit=history_limit):
                    if prev_message.author == client.user:
                        try:
                            await prev_message.delete()
                            deleted_count += 1
                            logger.debug(f"Deleted message ID: {prev_message.id}")
                            await asyncio.sleep(0.6)
                        except Exception as e: logger.warning(f"Error deleting msg {prev_message.id}: {e}"); error_count += 1
                logger.info(f"Discord delete finished. Deleted: {deleted_count}, Errors: {error_count}")
                clear_memory_for_channel(channel_id)
            except Exception as e:
                logger.error(f"An unexpected error occurred during delete_all: {e}", exc_info=True)
                await message.channel.send("Error during delete.")
            finally:
                try: await message.delete()
                except Exception: pass
                return

    # --- RAG Processing ---
    if not (is_mentioned or is_dm): return

    if not rag_chain:
        logger.warning("RAG chain not ready.")
        try: await message.reply("Sorry, I'm not fully initialized yet.")
        except Exception as e: logger.error(f"Error sending init warning: {e}")
        return

    user_input = message.content
    mention_string = f'<@{client.user.id}>'
    if is_mentioned: user_input = user_input.replace(mention_string, '').strip()
    if not user_input: logger.debug("Ignoring empty message."); return

    logger.info(f"Processing message for RAG from {message.author.name} in channel {channel_id}: '{user_input}'")

    async with message.channel.typing():
        try:
            memory = get_memory_for_channel(channel_id)
            # Use memory's chat_history attribute directly if return_messages=True
            current_history = memory.chat_memory.messages

            logger.info(f"--- History from Memory for Channel {channel_id} ({len(current_history)} messages) ---")
            if not current_history: logger.info("  [Memory History is empty]")
            else:
                for i, msg in enumerate(current_history):
                    content_preview = (msg.content[:150] + '...') if len(msg.content) > 150 else msg.content
                    logger.info(f"  Mem[{i}] ({type(msg).__name__}): '{content_preview}'")
            logger.info("-------------------------------------------------------------")

            response = await rag_chain.ainvoke({
                "input": user_input,
                "chat_history": current_history
            })

            answer = response.get('answer', "Sorry, I couldn't generate a response.").strip()
            if not answer: answer = "I couldn't form a specific answer."

            # Save context using defined keys 'input' and 'output'
            memory.save_context(
                {"input": user_input},
                {"output": answer}
            )
            logger.debug(f"Saved context to memory for channel {channel_id}")

            # Send Response
            max_length = 1990
            if len(answer) <= max_length:
                await message.reply(answer)
            else:
                logger.info("Answer length exceeds limit, sending chunks.")
                sent_chunks = 0
                for i in range(0, len(answer), max_length):
                    chunk = answer[i:i + max_length]
                    reply_func = message.reply if sent_chunks == 0 else message.channel.send
                    await reply_func(chunk)
                    sent_chunks += 1
                    await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Error during RAG/Memory processing or Discord reply: {e}", exc_info=True)
            try: await message.reply("Oops! Something went wrong.")
            except Exception as nested_e: logger.error(f"Failed to send error message: {nested_e}")

# --- Run the Bot ---
if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        logger.critical("Bot cannot start without DISCORD_BOT_TOKEN.")
    else:
        run_dummy_server(port=7860)
        logger.info("Starting Discord bot...")
        try:
            client.run(DISCORD_BOT_TOKEN, log_handler=None)
        except discord.errors.LoginFailure: logger.critical("Login failed: Invalid Token.")
        except discord.errors.PrivilegedIntentsRequired: logger.critical("Login failed: Message Content Intent not enabled.")
        except discord.errors.HTTPException as http_e: logger.critical(f"Discord HTTP Error: {http_e.status} {http_e.code} - {http_e.text}")
        except Exception as e: logger.critical(f"Error running bot: {e}", exc_info=True)