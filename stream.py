import os
import json
import glob
import requests
import asyncio
import aiohttp
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from queue import Queue
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
        self.max_concurrent_requests = int(
            os.getenv("MAX_CONCURRENT_REQUESTS", "3")
        )  # Reduced from 5 to 3
        self.request_timeout = int(
            os.getenv("REQUEST_TIMEOUT", "180")
        )  # 3 minutes timeout
        self.max_retries = int(os.getenv("MAX_RETRIES", "5"))
        self.chunk_size = int(
            os.getenv("CHUNK_SIZE", "4000")
        )  # Reduced from 5000 to 4000

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
        if "max_tokens" in data and data["max_tokens"] > 4000:
            data["max_tokens"] = 4000

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

    async def clean_text_async(self, session: aiohttp.ClientSession, text: str) -> str:
        """
        Asynchronously clean text by removing UI elements using both model-based and regex-based approaches.
        Fallback to regex-based cleaning if model-based cleaning fails.
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

        # Use regex cleaning as a default
        cleaned_text = regex_clean_text(text)

        # Short texts don't need model-based cleaning
        if len(text) < 1000:
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
                        "content": f"Clean the following text by removing any website UI elements, navigation menus, footers, ads, and other non-content elements. Return ONLY the cleaned text with no explanations:\n\n{text[:3000]}",
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 3500,
            }

            logger.info("Attempting to clean text with small model")
            response_content = await self._make_streaming_api_request(
                session, "chat/completions", clean_prompt, self.models["small"]
            )

            model_cleaned_text = response_content.strip()
            if model_cleaned_text and len(model_cleaned_text) > 100:
                logger.info("Successfully cleaned text with small model")
                return model_cleaned_text
            else:
                logger.warning(
                    "Small model returned insufficient cleaned text. Using regex cleaning."
                )
                return cleaned_text

        except Exception as e:
            logger.warning(
                f"Model-based cleaning failed: {str(e)}. Using regex cleaning."
            )
            return cleaned_text

    async def generate_qa_pairs_with_parallel_cleaning(
        self, session: aiohttp.ClientSession, chunk: str, chunk_id: int
    ) -> Dict[str, Any]:
        """
        Generate question-answer pairs for a chunk using the large model with web search enabled,
        while cleaning the text in parallel.
        """
        if len(chunk) > 3000:
            cleaning_task = asyncio.create_task(self.clean_text_async(session, chunk))
            logger.info(f"Started parallel cleaning for chunk {chunk_id}")
            cleaned_chunk = await cleaning_task
            logger.info(
                f"Completed cleaning for chunk {chunk_id}, now generating QA pairs"
            )
        else:
            # For small chunks, skip model-based cleaning
            cleaned_chunk = chunk
            logger.info(f"Chunk {chunk_id} is small, skipping model-based cleaning")

        # Limit the size of the chunk to process
        if len(cleaned_chunk) > 4000:
            cleaned_chunk = cleaned_chunk[:4000]
            logger.info(f"Truncated chunk {chunk_id} to 4000 chars")

        prompt = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Generate relevant question-answer pairs about the following text.",
                },
                {
                    "role": "user",
                    "content": f"Generate 3 question-answer pairs for the following text. Format your response as a JSON array of objects where each object has a 'question' and 'answer' field: {cleaned_chunk}",
                },
            ],
            "temperature": 0.7,
            "max_tokens": 3000,  # Reduced from 8192
        }

        try:
            response_content = await self._make_streaming_api_request(
                session,
                "chat/completions",
                prompt,
                self.models["large"],
                web_search=False,
            )

            # Find JSON in the response
            json_match = re.search(r"\[.*?\]", response_content, re.DOTALL)

            if json_match:
                try:
                    qa_pairs = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    # If JSON is malformed, try to fix it
                    fixed_json = self._fix_json_string(json_match.group(0))
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
                if isinstance(pair, dict) and "question" in pair and "answer" in pair:
                    jsonl_data["messages"].append(
                        {"role": "user", "content": pair["question"]}
                    )
                    jsonl_data["messages"].append(
                        {"role": "assistant", "content": pair["answer"]}
                    )

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

    def _fix_json_string(self, json_str):
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

    async def process_chunk_and_save(
        self, session: aiohttp.ClientSession, chunk: str, chunk_id: int
    ) -> None:
        """
        Process a single chunk with improved error handling and save the result.
        """
        try:
            # Set a timeout for the entire processing of this chunk
            result = await asyncio.wait_for(
                self.generate_qa_pairs_with_parallel_cleaning(session, chunk, chunk_id),
                timeout=self.request_timeout * 1.5,
            )

            jsonl_data = result.get("jsonl_data")
            if jsonl_data:
                valid_jsonl = await self.validate_and_fix_jsonl_async(
                    session, jsonl_data, chunk_id
                )

                # Save the result
                output_file = os.path.join(
                    self.output_dir, f"response_{chunk_id}.jsonl"
                )

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(valid_jsonl, f, ensure_ascii=False)

                logger.info(f"Saved result to {output_file}")
            else:
                logger.error(f"No valid JSONL data for chunk {chunk_id}")

        except asyncio.TimeoutError:
            logger.error(f"Timeout processing chunk {chunk_id}")
            self._save_error_file(chunk_id, "Timeout error")
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
            self._save_error_file(chunk_id, str(e))

    def _save_error_file(self, chunk_id, error_message):
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

        output_file = os.path.join(self.output_dir, f"error_{chunk_id}.jsonl")

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(error_data, f, ensure_ascii=False)
            logger.info(f"Saved error file to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save error file: {str(e)}")

    def process_file(self, file_path: str) -> None:
        """
        Process a single text file: read content, chunk it, and add to the queue.
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

            # Clear the queue before adding new chunks
            while not self.chunk_queue.empty():
                self.chunk_queue.get()

            for chunk in chunks:
                self.chunk_queue.put((chunk, file_output_dir))

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")

    async def process_chunks_with_semaphore(self, file_output_dir) -> None:
        """
        Process chunks using a semaphore to limit concurrent requests.
        Each file is processed separately to avoid memory issues.
        """
        # Create a semaphore with a lower limit to prevent overloading
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def process_with_semaphore(session, chunk, chunk_id, output_dir):
            async with semaphore:
                # Store the original output directory
                original_output_dir = self.output_dir

                try:
                    # Set the output directory to the file-specific directory
                    self.output_dir = output_dir

                    # Process the chunk with a timeout
                    await asyncio.wait_for(
                        self.process_chunk_and_save(session, chunk, chunk_id),
                        timeout=self.request_timeout * 1.5,
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Timeout in processing chunk {chunk_id}")
                    self._save_error_file(chunk_id, "Processing timeout")
                except Exception as e:
                    logger.error(f"Error in processing chunk {chunk_id}: {str(e)}")
                    self._save_error_file(chunk_id, str(e))
                finally:
                    # Restore the original output directory
                    self.output_dir = original_output_dir

        session = await self.create_session()  # Await the coroutine to get the session
        async with session:  # Now use the session in an async context manager
            chunk_id = 0
            tasks = []

            # Process up to self.max_concurrent_requests chunks at a time
            batch_size = self.max_concurrent_requests

            while not self.chunk_queue.empty():
                current_batch = []

                # Create a batch of tasks
                for _ in range(min(batch_size, self.chunk_queue.qsize())):
                    if self.chunk_queue.empty():
                        break

                    chunk, output_dir = self.chunk_queue.get()
                    task = asyncio.create_task(
                        process_with_semaphore(session, chunk, chunk_id, output_dir)
                    )
                    current_batch.append(task)
                    chunk_id += 1

                if current_batch:
                    # Wait for the current batch to complete before starting the next batch
                    await asyncio.gather(*current_batch, return_exceptions=True)

                    # Small delay between batches to let the system recover
                    await asyncio.sleep(1)

            # Wait for any remaining tasks
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def run(self) -> None:
        """
        Main execution method with improved error handling and per-file processing.
        """
        input_files = glob.glob(os.path.join(self.input_dir, "*.txt"))
        if not input_files:
            logger.warning(f"No .txt files found in {self.input_dir}")
            return

        # Process each file separately to avoid memory issues
        for file_path in input_files:
            try:
                logger.info(f"Starting processing for file: {file_path}")
                self.process_file(file_path)

                # Get filename without extension for the output directory
                base_filename = os.path.splitext(os.path.basename(file_path))[0]
                file_output_dir = os.path.join(self.output_dir, base_filename)

                # Process chunks for this file
                await self.process_chunks_with_semaphore(file_output_dir)

                logger.info(f"Completed processing file: {file_path}")

                # Small delay between files to let the system recover
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {str(e)}")

        logger.info("All processing complete!")

    def count_tokens(self, text, encoder):
        """Return the token count of text using the given encoder."""
        try:
            return len(encoder.encode(text))
        except Exception:
            # Fallback to character-based estimation
            return len(text) // 4

    def chunk_text(self, text, max_tokens=4000):
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
