import os
import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Загрузчики различных форматов файлов из базы
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


TELEGRAM_BOT_TOKEN = "TG_BOT_TOKEN"  
OPENAI_API_KEY = "SOME_OPENAI_API_KEY"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Путь к существующей БД
# !!!! Надо перенастроить это перед релизом под облачную БД
DOCUMENTS_DATABASE_DIR = "./my_documents_db"  
VECTOR_STORE_DIR = "./chroma_vector_db"

os.makedirs(DOCUMENTS_DATABASE_DIR, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)



# Возьня с LLM моделью

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = None


def scan_and_index_database():

    global vectorstore
    logger.info("Сканирование базы данных документов...")
    
    loaded_documents = []

    md_loader = DirectoryLoader(
        DOCUMENTS_DATABASE_DIR, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
    )
    loaded_documents.extend(md_loader.load())

    docx_loader = DirectoryLoader(
        DOCUMENTS_DATABASE_DIR, glob="**/*.docx", loader_cls=Docx2txtLoader
    )
    loaded_documents.extend(docx_loader.load())

    pdf_loader = DirectoryLoader(
        DOCUMENTS_DATABASE_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader
    )
    loaded_documents.extend(pdf_loader.load())

    if not loaded_documents:
        logger.warning("База данных документов пуста!")
        return False

    # Нарезка документов на смысловые блоки
    # Размер чанков нужно пересмотреть в зависимости от самих файлов
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(loaded_documents)

    # Индексация в векторное хранилище
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    logger.info(f"Успешно проиндексировано {len(loaded_documents)} файлов!")
    return True


# ----------------- ХЕНДЛЕРЫ TELEGRAM -----------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Стартовая команда"""
    await update.message.reply_text(
        "👋 **Бот-ассистент готов к работе!**\n\n"
        "Я подключен к базе данных вашей компании/проекта.\n"
        "Задайте мне любой вопрос, и я найду ответ в имеющихся документах (.md, .docx, .pdf).\n\n"
        "🔄 Для повторного сканирования базы админом используйте команду /reload"
    )


async def reload_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда для ручной переиндексации базы при добавлении новых файлов"""
    msg = await update.message.reply_text("🔄 Сканирую базу данных и обновляю индекс...")
    success = scan_and_index_database()
    if success:
        await msg.edit_text("✅ База данных успешно проиндексирована!")
    else:
        await msg.edit_text("⚠️ В папке базы данных не найдено поддерживаемых документов.")


async def handle_user_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых вопросов к базе данных"""
    global vectorstore

    if not vectorstore:
        await update.message.reply_text(
            "⚠️ База знаний временно не проиндексирована или пуста. "
            "Поместите документы в папку и выполните /reload."
        )
        return

    user_query = update.message.text
    status_msg = await update.message.reply_text("🔍 Обращаюсь к базе данных...")

    try:
        # Ищем 4 наиболее подходящих фрагмента текста
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # Шаблон промпта
        system_prompt = (
            "Вы — интеллектуальный бизнес-ассистент. Отвечайте на вопрос пользователя, "
            "строго опираясь на предоставленный контекст из базы данных.\n"
            "Если ответа нет в контексте, прямо ответьте, что в базе данных нет такой информации.\n\n"
            "Контекст из базы данных:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Вызов облачного API OpenAI
        response = rag_chain.invoke({"input": user_query})
        
        await status_msg.edit_text(response["answer"])

    except Exception as e:
        logger.error(f"Ошибка выполнения запроса: {e}")
        await status_msg.edit_text("❌ Ошибка при обращении к нейросети или базе данных.")


# ----------------- ЗАПУСК -----------------

def main():
    # Первичное сканирование папки с документами при старте
    if os.path.exists(DOCUMENTS_DATABASE_DIR):
        scan_and_index_database()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reload", reload_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_query))

    print("🚀 Скелет бота запущен и готов к тестированию!")
    app.run_polling()


if __name__ == "__main__":
    main()
