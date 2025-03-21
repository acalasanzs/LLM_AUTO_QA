import os
import json
import glob
import requests
import asyncio
import aiohttp
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from queue import Queue
from dotenv import load_dotenv
import re
import nltk
import tiktoken

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class OpenWebUIProcessor:
    def __init__(self):
        """
        Initialize the Open-WebUI API processor using environment variables.
        Expects the following environment variables:
          - URL_API_BASE: Base URL for the API
          - API_KEY: API key for authentication
          - INPUT_DIR: Directory containing input .txt files (default: ./input)
          - OUTPUT_DIR: Directory to save output files (default: ./output)
          - MAX_CONCURRENT_REQUESTS: Maximum number of concurrent API requests (default: 5)
          - small_MODEL_ID: Model ID for small model (for chunking)
          - medium_MODEL_ID: Model ID for medium model (for fixing JSON)
          - LARGE_MODEL_ID: Model ID for large model (for QA generation)
        """
        self.base_url = os.getenv("URL_API_BASE")
        if not self.base_url:
            raise Exception("No URL provided in the .env file for URL_API_BASE")
        self.api_key = os.getenv("API_KEY", "")
        self.input_dir = os.getenv("INPUT_DIR", "./input")
        self.output_dir = os.getenv("OUTPUT_DIR", "./output")
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))

        # Set up model IDs from environment variables
        self.models = {
            "small": os.getenv("small_MODEL_ID", "gemma-3-27b-it"),
            "medium": os.getenv("medium_MODEL_ID", "gemma-3-27b-it"),
            "large": os.getenv(
                "LARGE_MODEL_ID", "deepseek-r1-distill-qwen-32b-abliterated"
            ),
        }

        if not self.api_key:
            logger.warning("API key not found in environment variables. Set API_KEY.")

        # Set up headers for API requests
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Queue to hold text chunks
        self.chunk_queue = Queue()

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def _make_api_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        model_id: str,
        web_search: bool = False,
    ) -> Dict[str, Any]:
        """
        Make a synchronous API request to Open-WebUI.
        """
        url = f"{self.base_url}/{endpoint}"
        if web_search:
            url = f"{url}?web-search=true"

        data["model"] = model_id

        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    async def _make_async_api_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        data: Dict[str, Any],
        model_id: str,
        web_search: bool = False,
    ) -> Dict[str, Any]:
        """
        Make an asynchronous API request to Open-WebUI.
        Retries the request if a ServerDisconnectedError is encountered.
        """
        url = f"{self.base_url}/{endpoint}"
        if web_search:
            url = f"{url}?web-search=true"

        data["model"] = model_id

        max_retries = 3
        backoff_delay = 2  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                async with session.post(
                    url, headers=self.headers, json=data
                ) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.client_exceptions.ServerDisconnectedError as e:
                logger.warning(
                    f"Server disconnected error on attempt {attempt}/{max_retries}: {e}"
                )
                if attempt < max_retries:
                    await asyncio.sleep(backoff_delay)
                else:
                    logger.error("Max retries reached. Server is disconnecting.")
                    raise
            except aiohttp.ClientError as e:
                logger.error(f"Async API request failed: {e}")
                raise

    async def process_chunk_and_save(
        self, session: aiohttp.ClientSession, chunk: str, chunk_id: int
    ) -> None:
        """
        Process a single chunk:
          - Generate QA pairs with parallel cleaning.
          - Validate (and fix) the resulting JSONL data.
          - Immediately save the output to a file.

        If a repeated server disconnection error occurs, the file corresponding to the chunk is omitted.
        """
        try:
            result = await self.generate_qa_pairs_with_parallel_cleaning(
                session, chunk, chunk_id
            )
        except aiohttp.client_exceptions.ServerDisconnectedError as e:
            logger.error(
                f"Skipping chunk {chunk_id} due to repeated server disconnections: {e}"
            )
            return

        jsonl_data = result.get("jsonl_data")
        if jsonl_data:
            try:
                valid_jsonl = await self.validate_and_fix_jsonl_async(
                    session, jsonl_data, chunk_id
                )
                output_file = os.path.join(
                    self.output_dir, f"response_{chunk_id}.jsonl"
                )
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(json.dumps(valid_jsonl))
                logger.info(f"Saved result to {output_file}")
            except aiohttp.client_exceptions.ServerDisconnectedError as e:
                logger.error(
                    f"Skipping file save for chunk {chunk_id} due to server disconnection: {e}"
                )
        else:
            error = result.get("error", "unknown error")
            logger.error(f"Failed to process chunk {chunk_id}: {error}")

    async def clean_text_async(self, session: aiohttp.ClientSession, text: str) -> str:
        """
        Asynchronously clean text by removing UI elements using both model-based and regex-based approaches.
        """

        def regex_clean_text(input_text):
            """Basic cleaning using regex patterns to remove common UI elements"""
            patterns = [
                r"Cookie Policy.*?(?=\n\n|\Z)",
                r"Accept\s+(?:All)?\s*Cookies",
                r"Navigation Menu",
                r"Search\.\.\.",
                r"Share\s+(?:on)?\s+(?:Twitter|Facebook|LinkedIn)",
                r"©\s*\d{4}.*?(?=\n|\Z)",
                r"All Rights Reserved",
                r"Terms of (?:Use|Service)",
                r"Privacy Policy",
                r"\[\s*menu\s*\]",
                r"\[\s*footer\s*\]",
                r"\[\s*header\s*\]",
                r"\[\s*sidebar\s*\]",
                r"\[\s*advertisement\s*\]",
            ]
            cleaned = input_text
            for pattern in patterns:
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            cleaned = re.sub(r"\s{2,}", " ", cleaned)
            return cleaned.strip()

        cleaned_text = text
        try:
            clean_prompt = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a text processor that cleans and structures text content.",
                    },
                    {
                        "role": "user",
                        "content": f"Clean the following text by removing any website UI elements, navigation menus, footers, ads, and other non-content elements. Return ONLY the cleaned text with no explanations:\n\n{text[:4096]}",
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 5000,
            }
            try:
                logger.info("Attempting to clean text with small model")
                response = await self._make_async_api_request(
                    session, "chat/completions", clean_prompt, self.models["small"]
                )
                model_cleaned_text = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if model_cleaned_text and len(model_cleaned_text.strip()) > 100:
                    logger.info("Successfully cleaned text with small model")
                    cleaned_text = model_cleaned_text
                else:
                    logger.warning(
                        "Small model returned insufficient cleaned text. Using regex cleaning."
                    )
                    cleaned_text = regex_clean_text(text)
            except Exception as e:
                logger.warning(
                    f"Small model cleaning failed: {str(e)}. Using regex cleaning."
                )
                cleaned_text = regex_clean_text(text)
        except Exception as e:
            logger.error(f"All cleaning methods failed: {str(e)}. Using original text.")
        return cleaned_text

    async def generate_qa_pairs_with_parallel_cleaning(
        self, session: aiohttp.ClientSession, chunk: str, chunk_id: int
    ) -> Dict[str, Any]:
        """
        Generate question-answer pairs for a chunk using the large model with web search enabled,
        while cleaning the text in parallel.
        """
        cleaning_task = asyncio.create_task(self.clean_text_async(session, chunk))
        logger.info(f"Started parallel cleaning for chunk {chunk_id+1}")
        cleaned_chunk = await cleaning_task
        logger.info(
            f"Completed cleaning for chunk {chunk_id+1}, now generating QA pairs"
        )

        prompt = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Generate 30 relevant question-answer pairs about the following text.",
                },
                {
                    "role": "user",
                    "content": f"Generate 30 question-answer pairs for the following text. Format your response as a JSON array of objects where each object has a 'question' and 'answer' field: {cleaned_chunk}",
                },
            ],
            "temperature": 0.7,
            "max_tokens": 8192,
        }

        response = await self._make_async_api_request(
            session, "chat/completions", prompt, self.models["large"], web_search=True
        )
        try:
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                qa_pairs = json.loads(json_match.group(0))
            else:
                qa_pairs = json.loads(content)
            jsonl_data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]
            }
            for pair in qa_pairs:
                jsonl_data["messages"].append(
                    {"role": "user", "content": pair["question"]}
                )
                jsonl_data["messages"].append(
                    {"role": "assistant", "content": pair["answer"]}
                )
            return {"chunk_id": chunk_id, "jsonl_data": jsonl_data}
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Failed to parse QA pairs from response: {e}")
            return {
                "chunk_id": chunk_id,
                "jsonl_data": None,
                "error": str(e),
                "raw_response": content,
            }

    async def validate_and_fix_jsonl_async(
        self, session: aiohttp.ClientSession, jsonl_data: Dict[str, Any], chunk_id: int
    ) -> Dict[str, Any]:
        """
        Asynchronously validate JSONL data and, if necessary, fix it using the medium model.
        """
        try:
            json.dumps(jsonl_data)
            return jsonl_data
        except (TypeError, json.JSONDecodeError) as e:
            logger.warning(
                f"Invalid JSONL for chunk {chunk_id}: {e}. Attempting to fix..."
            )
            raw_data = str(jsonl_data)
            prompt = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a JSON validator and fixer.",
                    },
                    {
                        "role": "user",
                        "content": f"Fix this invalid JSONL data. Return only the valid JSON object: {raw_data}",
                    },
                ],
                "temperature": 0.3,
                "max_tokens": 5000,
            }
            response = await self._make_async_api_request(
                session, "chat/completions", prompt, self.models["medium"]
            )
            try:
                content = (
                    response.get("choices", [{}])
                    .pop(0)
                    .get("message", {})
                    .get("content", "")
                )
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if json_match:
                    fixed_jsonl = json.loads(json_match.group(0))
                else:
                    fixed_jsonl = json.loads(content)
                return fixed_jsonl
            except (json.JSONDecodeError, IndexError) as e:
                logger.error(f"Failed to fix JSONL: {e}")
                return {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Failed to process chunk."},
                        {
                            "role": "assistant",
                            "content": "I apologize, but I couldn't process that chunk properly.",
                        },
                    ]
                }

    def process_file(self, file_path: str) -> None:
        """
        Process a single text file:
          - Read file content.
          - Chunk the text.
          - Add each chunk to the FIFO queue.
        """
        logger.info(f"Processing file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = self.chunk_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            for chunk in chunks:
                self.chunk_queue.put(chunk)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")

    async def process_chunks_and_save(self) -> None:
        """
        Process chunks from the queue asynchronously by generating QA pairs and immediately saving each result to its own file.
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            chunk_id = 0
            while not self.chunk_queue.empty():
                chunk = self.chunk_queue.get()  # FIFO order
                task = asyncio.create_task(
                    self.process_chunk_and_save(session, chunk, chunk_id)
                )
                tasks.append(task)
                chunk_id += 1
                if len(tasks) >= self.max_concurrent_requests:
                    await asyncio.gather(*tasks)
                    tasks = []
            if tasks:
                await asyncio.gather(*tasks)

    async def run(self) -> None:
        """
        Main execution method:
          1. Fetch all .txt files from the input directory.
          2. Process each file (chunking text into semantic units).
          3. Process chunks concurrently to generate QA pairs with parallel cleaning and immediately save results.
        """
        input_files = glob.glob(os.path.join(self.input_dir, "*.txt"))
        if not input_files:
            logger.warning(f"No .txt files found in {self.input_dir}")
            return
        with ThreadPoolExecutor() as executor:
            for file_path in input_files:
                executor.submit(self.process_file, file_path)
        await self.process_chunks_and_save()
        logger.info("Processing complete!")

    def count_tokens(self, text, encoder):
        """Return the token count of text using the given encoder."""
        return len(encoder.encode(text))

    def chunk_text(self, text, max_tokens=5000):
        """
        Splits text into chunks under max_tokens WITHOUT cleaning.
        The cleaning will happen in parallel with QA generation.
        """
        try:
            encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            encoder = tiktoken.get_encoding("cl100k_base")
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception as e:
            logger.warning(
                f"Sentence tokenization failed: {e}. Falling back to simple splitting."
            )
            sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        for sentence in sentences:
            if not sentence.strip():
                continue
            try:
                sentence_tokens = self.count_tokens(sentence, encoder)
            except Exception as e:
                logger.warning(f"Token counting failed: {e}. Using character estimate.")
                sentence_tokens = len(sentence) // 4
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk = (
                    current_chunk + " " + sentence if current_chunk else sentence
                )
                current_tokens += sentence_tokens
            else:
                if sentence_tokens > max_tokens:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                        current_tokens = 0
                    try:
                        tokens = encoder.encode(sentence)
                        for i in range(0, len(tokens), max_tokens):
                            chunk_tokens = tokens[i : i + max_tokens]
                            chunk_text = encoder.decode(chunk_tokens)
                            chunks.append(chunk_text)
                    except Exception as e:
                        logger.warning(
                            f"Token-based splitting failed: {e}. Using character-based splitting."
                        )
                        for i in range(0, len(sentence), max_tokens * 4):
                            chunks.append(sentence[i : i + max_tokens * 4])
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
                    current_tokens = sentence_tokens
        if current_chunk:
            chunks.append(current_chunk)
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks


async def main():
    processor = OpenWebUIProcessor()
    await processor.run()


if __name__ == "__main__":
    nltk.download("punkt")
    asyncio.run(main())
