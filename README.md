# Angela Han AI - Discord RAG Bot

This Discord bot leverages Retrieval Augmented Generation (RAG) to answer questions and engage in conversations, embodying the persona of Angela Han. It uses Google's Gemini LLM, Google's text embeddings, and a FAISS vector store for knowledge retrieval. The bot maintains conversational memory per channel and can respond in the language of the user's last question.

## ✨ Features

*   **Conversational AI:** Engages in natural-sounding conversations powered by Google's Gemini LLM.
*   **Angela Han Persona:** Responds as an AI version of Angela Han, drawing on her perspectives, experiences, and communication style (as defined in the system prompt).
*   **Retrieval Augmented Generation (RAG):** Uses a FAISS vector store to retrieve relevant information from a knowledge base, providing contextually rich answers.
*   **Contextual Memory:** Remembers the conversation history on a per-channel basis using LangChain's `ConversationBufferMemory`.
*   **Multi-Language Response:** Detects the language of the user's question and aims to respond in the same language.
*   **DM Commands:**
    *   `!delete_last`: Deletes the bot's last message in the DM.
    *   `!delete_all`: Deletes all bot messages in the DM and clears the associated conversation memory for that DM channel.
*   **Interaction:** Responds to direct messages (DMs) or when @mentioned in server channels.
*   **Health Check:** Includes a simple HTTP server for health checks (e.g., for deployment platforms).
*   **Configurable:** Uses a `.env` file for API keys and paths.

## 🛠️ Technologies Used

*   **Python 3.8+**
*   **Discord.py:** For Discord bot integration.
*   **LangChain:** Framework for building LLM applications.
    *   `langchain_google_genai`: For Google Gemini LLM and Embeddings.
    *   `langchain_community.vectorstores.FAISS`: For FAISS vector store.
    *   `ConversationBufferMemory`: For chat history.
*   **Google Generative AI:**
    *   LLM Model: `gemini-2.0-flash-lite` (or as configured)
    *   Embedding Model: `models/text-embedding-004` (or as configured)
*   **FAISS:** For efficient similarity search in the vector store.
*   **python-dotenv:** For managing environment variables.
*   **Standard Python Libraries:** `os`, `asyncio`, `traceback`, `zipfile`, `tempfile`, `shutil`, `logging`, `threading`, `http.server`.

## ⚙️ Prerequisites

1.  **Python 3.8 or higher.**
2.  **Discord Bot Token:**
    *   Go to the [Discord Developer Portal](https://discord.com/developers/applications).
    *   Create a new application.
    *   Go to the "Bot" tab, add a bot.
    *   **Enable "Message Content Intent"** under Privileged Gateway Intents.
    *   Copy the bot token.
3.  **Google API Key:**
    *   Go to [Google AI Studio](https://aistudio.google.com/app/apikey) or Google Cloud Console.
    *   Create an API key.
    *   **Enable the "Generative Language API"** (also known as Gemini API) for your project.
4.  **FAISS Index ZIP File:**
    *   You need a `faiss_index_google.zip` file (or whatever you name it in `.env`).
    *   This ZIP file must contain `index.faiss` and `index.pkl` files.
    *   These files should be generated from your knowledge base documents using `GoogleGenerativeAIEmbeddings` (specifically with the model `models/text-embedding-004` or the one configured in the bot).
    *   *The bot does not include code to create this index; you must prepare it separately.*

## 🚀 Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rurounigit/discord_bot.git
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    discord.py
    python-dotenv
    langchain
    langchain-google-genai
    langchain-community
    faiss-cpu
    google-generativeai
    # Add any other specific versions if needed
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you have a CUDA-enabled GPU and want to use `faiss-gpu`, install it instead of `faiss-cpu`.*

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory of the project and add your credentials and paths:
    ```env
    DISCORD_BOT_TOKEN="YOUR_DISCORD_BOT_TOKEN_HERE"
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    FAISS_INDEX_ZIP_PATH="faiss_index_google.zip" # Or the path to your FAISS index ZIP
    ```
    *Make sure your `FAISS_INDEX_ZIP_PATH` points to the actual ZIP file containing your `index.faiss` and `index.pkl`.*

5.  **Add the Bot to Your Discord Server:**
    *   In the Discord Developer Portal, go to your application -> "OAuth2" -> "URL Generator".
    *   Select the `bot` scope.
    *   Under "Bot Permissions", select:
        *   `Read Messages/View Channels`
        *   `Send Messages`
        *   `Send Messages in Threads`
        *   `Read Message History`
        *   `Mention Everyone` (optional, if you want it to be able to @everyone - usually not needed for its core function)
        *   `Manage Messages` (if you want `!delete_last` and `!delete_all` to work on its own messages in channels, though currently these commands are DM-only where it implicitly has permission for its own messages).
    *   Copy the generated URL and open it in your browser to invite the bot to your server.

## ▶️ Running the Bot

Execute the Python script:
```bash
python bot.py
```
You should see log messages indicating the bot has connected to Discord, loaded the FAISS index, and initialized the LangChain components. The dummy HTTP server for health checks will also start on port `7860`.

## 💬 Usage

*   **In DMs:** Simply send a message to the bot.
*   **In Server Channels:** Mention the bot (e.g., `@AngelaHanAI What are your thoughts on community?`).
*   **DM Commands:**
    *   `!delete_last`: In a DM with the bot, type this to make the bot delete its most recent message in that DM.
    *   `!delete_all`: In a DM with the bot, type this to make the bot delete all of its previous messages in that DM and clear its memory for your DM channel. Your command message will also be deleted.

The bot will use its RAG pipeline and persona to respond to your queries.

## 🧠 FAISS Index (Knowledge Base)

The bot relies on a pre-built FAISS index for its RAG capabilities.

*   **Contents:** The `FAISS_INDEX_ZIP_PATH` (e.g., `faiss_index_google.zip`) must contain:
    *   `index.faiss`: The FAISS index itself.
    *   `index.pkl`: The LangChain FAISS docstore and index_to_docstore_id mapping.
*   **Embeddings:** This index **must** be created using `langchain_google_genai.GoogleGenerativeAIEmbeddings` with the model specified by `GOOGLE_EMBEDDING_MODEL_NAME` (default: `models/text-embedding-004`). If your index was created with different embeddings, the bot will likely not function correctly.
*   **Creation:** The script `bot.py` *does not* include functionality to create this index. You will need a separate script to:
    1.  Load your source documents (text files, PDFs, etc.).
    2.  Split them into manageable chunks.
    3.  Generate embeddings for these chunks using `GoogleGenerativeAIEmbeddings`.
    4.  Create a FAISS vector store from these embeddings and documents.
    5.  Save the FAISS index locally (`vector_store.save_local("faiss_index_directory")`).
    6.  Zip the contents of `faiss_index_directory` (which will be `index.faiss` and `index.pkl`) into the file specified by `FAISS_INDEX_ZIP_PATH`.

Example (conceptual) for creating the index:
```python
# --- This is NOT part of bot.py, but a separate script you'd run once ---
# from langchain_community.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# import os
# from dotenv import load_dotenv

# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

# # 1. Load documents (replace with your actual document loading)
# # loader = DirectoryLoader('path/to/your/documents', glob="**/*.txt")
# # documents = loader.load()

# # For example purposes, let's use dummy documents
# from langchain_core.documents import Document
# documents = [
#     Document(page_content="Angela Han believes in open communication."),
#     Document(page_content="Polyamory requires a lot of self-reflection according to Angela."),
#     Document(page_content="Community care is a central theme for Angela Han.")
# ]


# # 2. Split documents
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents(documents)

# # 3. Create FAISS vector store
# vector_store = FAISS.from_documents(texts, embeddings)

# # 4. Save FAISS index
# vector_store.save_local("faiss_index_google_temp") # This creates a folder with index.faiss and index.pkl

# # 5. Zip it (manual step or use zipfile library)
# # import zipfile
# # with zipfile.ZipFile("faiss_index_google.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
# #     zf.write("faiss_index_google_temp/index.faiss", "index.faiss")
# #     zf.write("faiss_index_google_temp/index.pkl", "index.pkl")
# # print("FAISS index created and zipped as faiss_index_google.zip")
# # Remember to clean up faiss_index_google_temp folder
```

## 🎨 Persona Customization

The bot's persona is primarily defined in the `persona_qa_system_prompt` variable within `bot.py`. You can modify this prompt to change the bot's personality, background, and response style.

## ⚠️ Known Issues / Limitations

*   **Memory is In-Memory:** Conversation history is stored in RAM and will be lost if the bot restarts. For persistent memory, a database-backed memory solution would be needed.
*   **FAISS Index Creation:** The bot does not create the FAISS index. This must be done manually.
*   **Rate Limits:** Be mindful of API rate limits for both Discord and Google Generative AI.
*   **Error Handling:** While basic error handling is in place, more robust handling could be added for specific API errors or edge cases.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find a bug, please:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature` or `bugfix/YourBugfix`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details (if you add one). If no LICENSE file is present, assume the code is proprietary unless otherwise stated by the author.
```

**Next Steps for You:**

1.  **Replace Placeholders:**
    *   Change `https://github.com/your-username/angela-han-ai-bot.git` to your actual repository URL.
    *   Consider adding a screenshot or GIF where `![Bot Demo Placeholder]` is.
2.  **Create `requirements.txt`:** As mentioned in the "Setup & Installation" section, create this file.
3.  **(Optional) Add a `LICENSE` file:** If you want to make it open source, choose a license (MIT is common and permissive) and add a `LICENSE` file to your repository.
4.  **FAISS Index:** This is the most critical external dependency. Ensure you have a clear process for users (or yourself) to create this index. The example in the README is conceptual and needs to be adapted to your actual document sources.
5.  **Test Thoroughly:** Before sharing widely, test all features, especially the commands and interactions in DMs and channels.

## ✨ Features

*   **Conversational AI:** Engages in natural-sounding conversations powered by Google's Gemini LLM.
*   **Angela Han Persona:** Responds as an AI version of Angela Han, drawing on her perspectives, experiences, and communication style (as defined in the system prompt).
*   **Retrieval Augmented Generation (RAG):** Uses a FAISS vector store to retrieve relevant information from a knowledge base, providing contextually rich answers.
*   **Contextual Memory:** Remembers the conversation history on a per-channel basis using LangChain's `ConversationBufferMemory`.
*   **Multi-Language Response:** Detects the language of the user's question and aims to respond in the same language.
*   **DM Commands:**
    *   `!delete_last`: Deletes the bot's last message in the DM.
    *   `!delete_all`: Deletes all bot messages in the DM and clears the associated conversation memory for that DM channel.
*   **Interaction:** Responds to direct messages (DMs) or when @mentioned in server channels.
*   **Health Check:** Includes a simple HTTP server for health checks (e.g., for deployment platforms).
*   **Configurable:** Uses a `.env` file for API keys and paths.

## 🛠️ Technologies Used

*   **Python 3.8+**
*   **Discord.py:** For Discord bot integration.
*   **LangChain:** Framework for building LLM applications.
    *   `langchain_google_genai`: For Google Gemini LLM and Embeddings.
    *   `langchain_community.vectorstores.FAISS`: For FAISS vector store.
    *   `ConversationBufferMemory`: For chat history.
*   **Google Generative AI:**
    *   LLM Model: `gemini-2.0-flash-lite` (or as configured)
    *   Embedding Model: `models/text-embedding-004` (or as configured)
*   **FAISS:** For efficient similarity search in the vector store.
*   **python-dotenv:** For managing environment variables.
*   **Standard Python Libraries:** `os`, `asyncio`, `traceback`, `zipfile`, `tempfile`, `shutil`, `logging`, `threading`, `http.server`.

## ⚙️ Prerequisites

1.  **Python 3.8 or higher.**
2.  **Discord Bot Token:**
    *   Go to the [Discord Developer Portal](https://discord.com/developers/applications).
    *   Create a new application.
    *   Go to the "Bot" tab, add a bot.
    *   **Enable "Message Content Intent"** under Privileged Gateway Intents.
    *   Copy the bot token.
3.  **Google API Key:**
    *   Go to [Google AI Studio](https://aistudio.google.com/app/apikey) or Google Cloud Console.
    *   Create an API key.
    *   **Enable the "Generative Language API"** (also known as Gemini API) for your project.
4.  **FAISS Index ZIP File:**
    *   You need a `faiss_index_google.zip` file (or whatever you name it in `.env`).
    *   This ZIP file must contain `index.faiss` and `index.pkl` files.
    *   These files should be generated from your knowledge base documents using `GoogleGenerativeAIEmbeddings` (specifically with the model `models/text-embedding-004` or the one configured in the bot).
    *   *The bot does not include code to create this index; you must prepare it separately.*

## 🚀 Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/angela-han-ai-bot.git # Replace with your repo URL
    cd angela-han-ai-bot
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    discord.py
    python-dotenv
    langchain
    langchain-google-genai
    langchain-community
    faiss-cpu
    google-generativeai
    # Add any other specific versions if needed
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you have a CUDA-enabled GPU and want to use `faiss-gpu`, install it instead of `faiss-cpu`.*

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory of the project and add your credentials and paths:
    ```env
    DISCORD_BOT_TOKEN="YOUR_DISCORD_BOT_TOKEN_HERE"
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
    FAISS_INDEX_ZIP_PATH="faiss_index_google.zip" # Or the path to your FAISS index ZIP
    ```
    *Make sure your `FAISS_INDEX_ZIP_PATH` points to the actual ZIP file containing your `index.faiss` and `index.pkl`.*

5.  **Add the Bot to Your Discord Server:**
    *   In the Discord Developer Portal, go to your application -> "OAuth2" -> "URL Generator".
    *   Select the `bot` scope.
    *   Under "Bot Permissions", select:
        *   `Read Messages/View Channels`
        *   `Send Messages`
        *   `Send Messages in Threads`
        *   `Read Message History`
        *   `Mention Everyone` (optional, if you want it to be able to @everyone - usually not needed for its core function)
        *   `Manage Messages` (if you want `!delete_last` and `!delete_all` to work on its own messages in channels, though currently these commands are DM-only where it implicitly has permission for its own messages).
    *   Copy the generated URL and open it in your browser to invite the bot to your server.

## ▶️ Running the Bot

Execute the Python script:
```bash
python bot.py
```
You should see log messages indicating the bot has connected to Discord, loaded the FAISS index, and initialized the LangChain components. The dummy HTTP server for health checks will also start on port `7860`.

## 💬 Usage

*   **In DMs:** Simply send a message to the bot.
*   **In Server Channels:** Mention the bot (e.g., `@AngelaHanAI What are your thoughts on community?`).
*   **DM Commands:**
    *   `!delete_last`: In a DM with the bot, type this to make the bot delete its most recent message in that DM.
    *   `!delete_all`: In a DM with the bot, type this to make the bot delete all of its previous messages in that DM and clear its memory for your DM channel. Your command message will also be deleted.

The bot will use its RAG pipeline and persona to respond to your queries.

## 🧠 FAISS Index (Knowledge Base)

The bot relies on a pre-built FAISS index for its RAG capabilities.

*   **Contents:** The `FAISS_INDEX_ZIP_PATH` (e.g., `faiss_index_google.zip`) must contain:
    *   `index.faiss`: The FAISS index itself.
    *   `index.pkl`: The LangChain FAISS docstore and index_to_docstore_id mapping.
*   **Embeddings:** This index **must** be created using `langchain_google_genai.GoogleGenerativeAIEmbeddings` with the model specified by `GOOGLE_EMBEDDING_MODEL_NAME` (default: `models/text-embedding-004`). If your index was created with different embeddings, the bot will likely not function correctly.
*   **Creation:** The script `bot.py` *does not* include functionality to create this index. You will need a separate script to:
    1.  Load your source documents (text files, PDFs, etc.).
    2.  Split them into manageable chunks.
    3.  Generate embeddings for these chunks using `GoogleGenerativeAIEmbeddings`.
    4.  Create a FAISS vector store from these embeddings and documents.
    5.  Save the FAISS index locally (`vector_store.save_local("faiss_index_directory")`).
    6.  Zip the contents of `faiss_index_directory` (which will be `index.faiss` and `index.pkl`) into the file specified by `FAISS_INDEX_ZIP_PATH`.

Example (conceptual) for creating the index:
```python
# --- This is NOT part of bot.py, but a separate script you'd run once ---
# from langchain_community.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# import os
# from dotenv import load_dotenv

# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

# # 1. Load documents (replace with your actual document loading)
# # loader = DirectoryLoader('path/to/your/documents', glob="**/*.txt")
# # documents = loader.load()

# # For example purposes, let's use dummy documents
# from langchain_core.documents import Document
# documents = [
#     Document(page_content="Angela Han believes in open communication."),
#     Document(page_content="Polyamory requires a lot of self-reflection according to Angela."),
#     Document(page_content="Community care is a central theme for Angela Han.")
# ]


# # 2. Split documents
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents(documents)

# # 3. Create FAISS vector store
# vector_store = FAISS.from_documents(texts, embeddings)

# # 4. Save FAISS index
# vector_store.save_local("faiss_index_google_temp") # This creates a folder with index.faiss and index.pkl

# # 5. Zip it (manual step or use zipfile library)
# # import zipfile
# # with zipfile.ZipFile("faiss_index_google.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
# #     zf.write("faiss_index_google_temp/index.faiss", "index.faiss")
# #     zf.write("faiss_index_google_temp/index.pkl", "index.pkl")
# # print("FAISS index created and zipped as faiss_index_google.zip")
# # Remember to clean up faiss_index_google_temp folder
```

## 🎨 Persona Customization

The bot's persona is primarily defined in the `persona_qa_system_prompt` variable within `bot.py`. You can modify this prompt to change the bot's personality, background, and response style.

## ⚠️ Known Issues / Limitations

*   **Memory is In-Memory:** Conversation history is stored in RAM and will be lost if the bot restarts. For persistent memory, a database-backed memory solution would be needed.
*   **FAISS Index Creation:** The bot does not create the FAISS index. This must be done manually.
*   **Rate Limits:** Be mindful of API rate limits for both Discord and Google Generative AI.
*   **Error Handling:** While basic error handling is in place, more robust handling could be added for specific API errors or edge cases.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find a bug, please:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature` or `bugfix/YourBugfix`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details (if you add one). If no LICENSE file is present, assume the code is proprietary unless otherwise stated by the author.
