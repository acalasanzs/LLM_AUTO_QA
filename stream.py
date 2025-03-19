import os
import json
import glob
import requests
import asyncio
import aiohttp
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Tuple
from queue import Queue
from collections import deque
from dotenv import load_dotenv
import re
import nltk
import tiktoken
import backoff

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
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))
        self.request_timeout = int(
            os.getenv("REQUEST_TIMEOUT", "180")
        )  # 3 minutes timeout
        self.max_retries = int(os.getenv("MAX_RETRIES", "5"))

        # Standardize chunk size - consistent value throughout the code
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "6000"))

        # Set up model IDs from environment variables
        self.models = {
            "small": os.getenv("small_MODEL_ID", "qwen2.5-coder-14b-instruct-mlx"),
            "medium": os.getenv("medium_MODEL_ID", "qwen2.5-coder-14b-instruct-mlx"),
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

        # FIFO Queue for processing pipeline
        self.processing_queue = asyncio.Queue()

        # Semaphore to control concurrent API requests
        self.api_semaphore = None

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    @backoff.on_exception(
        backoff.expo,
        (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            aiohttp.client_exceptions.ClientConnectorError,
            aiohttp.client_exceptions.ServerDisconnectedError,
        ),
        max_tries=5,
        jitter=backoff.full_jitter,
    )
    async def _make_streaming_api_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        data: Dict[str, Any],
        model_id: str,
        web_search: bool = False,
    ) -> str:
        """
        Make an asynchronous streaming API request to Open-WebUI.
        Uses backoff for retries and handles disconnects gracefully.
        """
        url = f"{self.base_url}/{endpoint}"
        if web_search:
            url = f"{url}?web-search=true"

        # Add stream parameter to request
        data["model"] = model_id
        data["stream"] = True

        # Reduce token generation when possible
        if "max_tokens" in data and data["max_tokens"] > 6000:
            data["max_tokens"] = 6000

        complete_response = ""

        try:
            async with session.post(
                url, headers=self.headers, json=data, timeout=self.request_timeout
            ) as response:
                response.raise_for_status()

                # Process the streaming response
                buffer = ""
                async for line in response.content:
                    if session.closed:
                        logger.warning("Session closed during streaming. Stopping.")
                        break

                    try:
                        line_text = line.decode("utf-8").strip()

                        if not line_text:
                            continue

                        if line_text.startswith("data: "):
                            line_text = line_text[6:]  # Remove 'data: ' prefix

                        if line_text == "[DONE]":
                            break

                        # Accumulate data in case of partial JSON
                        buffer += line_text

                        try:
                            # Try to parse JSON
                            chunk = json.loads(buffer)
                            buffer = ""  # Reset buffer after successful parse

                            # Extract content from the chunk
                            if "choices" in chunk and chunk["choices"]:
                                if "delta" in chunk["choices"][0]:
                                    delta = chunk["choices"][0]["delta"]
                                    if "content" in delta:
                                        complete_response += delta["content"]
                        except json.JSONDecodeError:
                            # If JSON is incomplete, keep accumulating
                            continue
                    except UnicodeDecodeError:
                        logger.warning(
                            "Unicode decode error with streaming chunk. Skipping."
                        )
                        continue

                return complete_response

        except asyncio.CancelledError:
            logger.warning("Task cancelled during API request.")
            raise
        except Exception as e:
            logger.error(f"Error during streaming request: {str(e)}")
            raise

    async def create_session(self):
        """Create a new aiohttp session with optimized settings for stability."""
        conn = aiohttp.TCPConnector(
            limit=self.max_concurrent_requests,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            force_close=True,
            limit_per_host=2,
        )

        return aiohttp.ClientSession(
            connector=conn,
            timeout=aiohttp.ClientTimeout(
                total=self.request_timeout,
                connect=60,
                sock_connect=60,
                sock_read=self.request_timeout,
            ),
            headers=self.headers,
        )

    async def clean_text_async(
        self, session: aiohttp.ClientSession, text: str, chunk_id: int
    ) -> str:
        """
        Asynchronously clean text by removing UI elements using both model-based and regex-based approaches.
        Fallback to regex-based cleaning if model-based cleaning fails.
        """
        async with self.api_semaphore:
            logger.info(f"Started cleaning text for chunk {chunk_id}")

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

            # Use regex cleaning as a default
            cleaned_text = regex_clean_text(text)

            # Short texts don't need model-based cleaning
            if len(text) < 1000:
                logger.info(f"Chunk {chunk_id} is short, skipping model-based cleaning")
                return cleaned_text

            try:
                clean_prompt = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a text processor that cleans and structures text content.",
                        },
                        {
                            "role": "user",
                            "content": f"Clean the following text by removing any website UI elements, navigation menus, footers, ads, and other non-content elements. Return ONLY the cleaned text with no explanations:\n\n{text}",
                        },
                    ],
                    "temperature": 0.1,
                    "max_tokens": 8192,
                }

                logger.info(
                    f"Attempting to clean text with small model for chunk {chunk_id}"
                )
                response_content = await self._make_streaming_api_request(
                    session, "chat/completions", clean_prompt, self.models["small"]
                )

                model_cleaned_text = response_content.strip()
                if model_cleaned_text and len(model_cleaned_text) > 100:
                    logger.info(
                        f"Successfully cleaned text with small model for chunk {chunk_id}"
                    )
                    return model_cleaned_text
                else:
                    logger.warning(
                        f"Small model returned insufficient cleaned text for chunk {chunk_id}. Using regex cleaning."
                    )
                    return cleaned_text

            except Exception as e:
                logger.warning(
                    f"Model-based cleaning failed for chunk {chunk_id}: {str(e)}. Using regex cleaning."
                )
                return cleaned_text

    async def _generate_qa_pairs(
        self,
        session: aiohttp.ClientSession,
        chunk: str,
        chunk_id: int,
        model_id: str,
        web_search: bool,
    ) -> Dict[str, Any]:
        """
        Helper method to generate QA pairs from a chunk of text.
        """
        async with self.api_semaphore:
            logger.info(f"Started generating QA pairs for chunk {chunk_id}")

            # Limit the size of the chunk to process
            if len(chunk) > 8000:
                chunk = chunk[:8000]
                logger.info(f"Truncated chunk {chunk_id} to 6000 chars")

            prompt = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Generate relevant 30 question-answer pairs about the following text.",
                    },
                    {
                        "role": "user",
                        "content": f"Generate 30 question-answer pairs for the following text. Format your response as a JSON array of objects where each object has a 'question' and 'answer' field: {chunk}",
                    },
                ],
                "temperature": 0.7,
                "max_tokens": 8192,
            }

            try:
                response_content = await self._make_streaming_api_request(
                    session, "chat/completions", prompt, model_id, web_search=web_search
                )

                # Find JSON in the response
                json_match = re.search(r"\[.*?\]", response_content, re.DOTALL)

                if json_match:
                    try:
                        qa_pairs = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        # If JSON is malformed, try to fix it
                        fixed_json = await self._fix_json_string_async(
                            session, json_match.group(0), chunk_id
                        )
                        qa_pairs = json.loads(fixed_json)
                else:
                    # If no JSON array found, try to parse the entire response
                    try:
                        qa_pairs = json.loads(response_content)
                    except json.JSONDecodeError:
                        # If still fails, create a simple structure
                        qa_pairs = [
                            {
                                "question": "What is the main topic of this text?",
                                "answer": "The text discusses various topics that couldn't be structured into proper QA pairs.",
                            }
                        ]

                jsonl_data = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."}
                    ]
                }

                for pair in qa_pairs:
                    if (
                        isinstance(pair, dict)
                        and "question" in pair
                        and "answer" in pair
                    ):
                        jsonl_data["messages"].append(
                            {"role": "user", "content": pair["question"]}
                        )
                        jsonl_data["messages"].append(
                            {"role": "assistant", "content": pair["answer"]}
                        )

                logger.info(f"Successfully generated QA pairs for chunk {chunk_id}")
                return {"chunk_id": chunk_id, "jsonl_data": jsonl_data}

            except Exception as e:
                logger.error(f"Failed to generate QA pairs for chunk {chunk_id}: {e}")
                # Create a simple fallback response
                jsonl_data = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": "Can you summarize the key points from this text?",
                        },
                        {
                            "role": "assistant",
                            "content": f"I encountered an error processing this chunk of text. Error: {str(e)}",
                        },
                    ]
                }
                return {"chunk_id": chunk_id, "jsonl_data": jsonl_data}

    async def _fix_json_string_async(
        self, session: aiohttp.ClientSession, json_str: str, chunk_id: int
    ) -> str:
        """
        Attempt to fix JSON parsing issues using both manual fixes and model-based assistance.
        """

        def manual_fix_json(json_str):
            """Attempt to fix common JSON parsing issues."""
            # Replace single quotes with double quotes
            fixed = json_str.replace("'", '"')

            # Ensure property names are in double quotes
            fixed = re.sub(r"(\w+):", r'"\1":', fixed)

            # Fix missing commas between objects
            fixed = re.sub(r"}\s*{", "},{", fixed)

            # Fix trailing commas in arrays
            fixed = re.sub(r",\s*]", "]", fixed)

            return fixed

        # Try manual fixing first
        try:
            fixed_json = manual_fix_json(json_str)
            json.loads(fixed_json)  # Test if valid
            return fixed_json
        except (json.JSONDecodeError, Exception):
            # If manual fixing fails, try using the medium model for more complex fixes
            try:
                # Only use the model for complex cases when manual fixing fails
                fix_prompt = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a JSON repair specialist. Fix the invalid JSON and return only the corrected JSON with no additional text.",
                        },
                        {
                            "role": "user",
                            "content": f"Fix this invalid JSON and return only the corrected JSON:\n\n{json_str}",
                        },
                    ],
                    "temperature": 0.1,
                    "max_tokens": 8192,
                }

                logger.info(f"Using medium model to fix JSON for chunk {chunk_id}")
                async with self.api_semaphore:
                    response_content = await self._make_streaming_api_request(
                        session, "chat/completions", fix_prompt, self.models["medium"]
                    )

                # Try to extract valid JSON from the response
                json_match = re.search(r"\[.*?\]", response_content, re.DOTALL)
                if json_match:
                    try:
                        fixed_json = json_match.group(0)
                        json.loads(fixed_json)  # Test if valid
                        return fixed_json
                    except json.JSONDecodeError:
                        pass

                # If no valid JSON array was found, try a simpler approach
                return '[\n  {\n    "question": "What is the main topic?",\n    "answer": "The content was not properly structured due to JSON parsing issues."\n  }\n]'
            except Exception as e:
                logger.error(f"Failed to fix JSON with model for chunk {chunk_id}: {e}")
                return '[\n  {\n    "question": "What is the main topic?",\n    "answer": "The content was not properly structured due to JSON parsing issues."\n  }\n]'

    async def validate_and_fix_jsonl_async(
        self, session: aiohttp.ClientSession, jsonl_data: Dict[str, Any], chunk_id: int
    ) -> Dict[str, Any]:
        """
        Validate JSONL data and fix it if necessary.
        """
        try:
            # Try to serialize to ensure it's valid
            json.dumps(jsonl_data)
            return jsonl_data
        except (TypeError, json.JSONDecodeError) as e:
            logger.warning(
                f"Invalid JSONL for chunk {chunk_id}: {e}. Attempting to fix..."
            )

            # Create a simplified structure if jsonl_data can't be fixed
            fallback_data = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": "Can you provide information about this text?",
                    },
                    {
                        "role": "assistant",
                        "content": "I encountered an error processing this chunk of text. The JSON structure was invalid.",
                    },
                ]
            }

            try:
                # Try to fix by using a simple approach rather than asking another model
                if isinstance(jsonl_data, dict) and "messages" in jsonl_data:
                    clean_messages = []
                    for msg in jsonl_data["messages"]:
                        if isinstance(msg, dict) and "role" in msg and "content" in msg:
                            clean_messages.append(
                                {
                                    "role": str(msg["role"]),
                                    "content": str(msg["content"]),
                                }
                            )

                    if clean_messages:
                        return {"messages": clean_messages}

                return fallback_data
            except Exception:
                logger.error(f"Failed to fix JSONL manually for chunk {chunk_id}")
                return fallback_data

    async def process_pipeline(self, session: aiohttp.ClientSession):
        """
        Process chunks from the FIFO queue, handling cleaning and QA generation in parallel.
        This is the main worker that processes items from the queue.
        """
        while True:
            try:
                # Get the next item from the queue
                item = await self.processing_queue.get()

                if item is None:  # None is our signal to stop
                    self.processing_queue.task_done()
                    break

                chunk, chunk_id, output_dir = item

                try:
                    # Step 1: Clean text asynchronously
                    logger.info(
                        f"Pipeline: Starting text cleaning for chunk {chunk_id}"
                    )
                    cleaned_text = await self.clean_text_async(session, chunk, chunk_id)

                    # Step 2: Generate QA pairs from cleaned text
                    logger.info(
                        f"Pipeline: Starting QA generation for chunk {chunk_id}"
                    )
                    result = await self._generate_qa_pairs(
                        session, cleaned_text, chunk_id, self.models["large"], False
                    )

                    # Step 3: Validate and fix JSONL if needed
                    jsonl_data = result.get("jsonl_data")
                    if jsonl_data:
                        valid_jsonl = await self.validate_and_fix_jsonl_async(
                            session, jsonl_data, chunk_id
                        )

                        # Step 4: Save the result
                        output_file = os.path.join(
                            output_dir, f"response_{chunk_id}.jsonl"
                        )
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(valid_jsonl, f, ensure_ascii=False)

                        logger.info(f"Saved result to {output_file}")
                    else:
                        logger.error(f"No valid JSONL data for chunk {chunk_id}")
                        self._save_error_file(
                            chunk_id, "No valid JSONL data", output_dir
                        )

                except asyncio.TimeoutError:
                    logger.error(f"Timeout processing chunk {chunk_id}")
                    self._save_error_file(chunk_id, "Timeout error", output_dir)
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
                    self._save_error_file(chunk_id, str(e), output_dir)
                finally:
                    # Mark this task as done
                    self.processing_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Unexpected error in pipeline: {str(e)}")
                continue

    def _save_error_file(self, chunk_id, error_message, output_dir):
        """Save an error file for failed chunks."""
        error_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What happened with this chunk?"},
                {
                    "role": "assistant",
                    "content": f"Processing failed with error: {error_message}",
                },
            ]
        }

        output_file = os.path.join(output_dir, f"error_{chunk_id}.jsonl")

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(error_data, f, ensure_ascii=False)
            logger.info(f"Saved error file to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save error file: {str(e)}")

    def process_file(self, file_path: str) -> Tuple[List[str], str]:
        """
        Process a single text file: read content, chunk it, and return the chunks with output directory.
        """
        logger.info(f"Processing file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Get filename without extension
            base_filename = os.path.splitext(os.path.basename(file_path))[0]

            # Create a subdirectory for this file's chunks
            file_output_dir = os.path.join(self.output_dir, base_filename)
            os.makedirs(file_output_dir, exist_ok=True)

            chunks = self.chunk_text(text, max_tokens=self.chunk_size)
            logger.info(f"Split text into {len(chunks)} chunks")

            return chunks, file_output_dir

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return [], ""

    async def run(self) -> None:
        """
        Main execution method with improved parallel processing.
        """
        input_files = glob.glob(os.path.join(self.input_dir, "*.txt"))
        if not input_files:
            logger.warning(f"No .txt files found in {self.input_dir}")
            return

        # Initialize the semaphore for API request limiting
        self.api_semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        # Create a session for all requests
        session = await self.create_session()

        async with session:
            # Start worker tasks for parallel processing
            workers = []
            for _ in range(self.max_concurrent_requests):
                worker = asyncio.create_task(self.process_pipeline(session))
                workers.append(worker)

            # Process each file
            for file_path in input_files:
                try:
                    logger.info(f"Starting processing for file: {file_path}")
                    chunks, file_output_dir = self.process_file(file_path)

                    if not chunks or not file_output_dir:
                        logger.warning(
                            f"Skipping file {file_path} due to processing errors"
                        )
                        continue

                    # Add all chunks to the processing queue
                    for chunk_id, chunk in enumerate(chunks):
                        await self.processing_queue.put(
                            (chunk, chunk_id, file_output_dir)
                        )

                    logger.info(
                        f"Added {len(chunks)} chunks from {file_path} to processing queue"
                    )

                except Exception as e:
                    logger.error(f"Failed to process file {file_path}: {str(e)}")

            # Wait for all tasks in the queue to be processed
            await self.processing_queue.join()

            # Send stop signals to all workers
            for _ in range(len(workers)):
                await self.processing_queue.put(None)

            # Wait for all workers to finish
            await asyncio.gather(*workers, return_exceptions=True)

        logger.info("All processing complete!")

    def count_tokens(self, text, encoder):
        """Return the token count of text using the given encoder."""
        try:
            return len(encoder.encode(text))
        except Exception:
            # Fallback to character-based estimation
            return len(text) // 4

    def chunk_text(self, text, max_tokens=6000):
        """
        Split text into smaller chunks, optimized for stability.
        """
        # Try to use tiktoken for token counting
        try:
            encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            try:
                encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                encoder = None
                logger.warning(
                    "Could not load tiktoken. Using character-based estimation."
                )

        # Try to use NLTK for sentence splitting
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception as e:
            logger.warning(
                f"NLTK sentence tokenization failed: {e}. Using regex splitting."
            )
            # Simple regex-based sentence splitting
            sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Estimate token count
            if encoder:
                try:
                    sentence_tokens = self.count_tokens(sentence, encoder)
                except Exception:
                    sentence_tokens = len(sentence) // 4
            else:
                sentence_tokens = len(sentence) // 4

            # If adding this sentence keeps us under max_tokens, add it
            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk = (
                    current_chunk + " " + sentence if current_chunk else sentence
                )
                current_tokens += sentence_tokens
            else:
                # If the current sentence is too long by itself, split it
                if sentence_tokens > max_tokens:
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                        current_tokens = 0

                    # Split long sentences into smaller pieces
                    words = sentence.split()
                    temp_chunk = ""
                    temp_tokens = 0

                    for word in words:
                        word_tokens = len(word) // 4 + 1  # Rough estimate
                        if temp_tokens + word_tokens <= max_tokens:
                            temp_chunk = temp_chunk + " " + word if temp_chunk else word
                            temp_tokens += word_tokens
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk)
                            temp_chunk = word
                            temp_tokens = word_tokens

                    if temp_chunk:
                        chunks.append(temp_chunk)
                else:
                    # Save current chunk and start a new one
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
                    current_tokens = sentence_tokens

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks


async def main():
    try:
        nltk.download("punkt", quiet=True)
    except Exception:
        pass

    processor = OpenWebUIProcessor()
    await processor.run()


if __name__ == "__main__":
    asyncio.run(main())
