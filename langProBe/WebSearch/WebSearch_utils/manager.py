import logging
import re
import time
import traceback
from typing import Optional


class ResponseManager:
    """Manages the processing of responses and web search operations."""

    def __init__(self):
        self.client = None
        self.max_retries = 5
        self.retry_delay = 2
        self.initialization_timeout = 30
        
        # 配置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def initialize_client(self):
        """延迟初始化客户端，添加重试机制"""
        if self.client is None:
            start_time = time.time()
            for attempt in range(self.max_retries):
                try:
                    from .synced_mcp_client import SyncedMcpClient
                    self.client = SyncedMcpClient(server_url="http://47.243.19.78:33333/sse")
                    
                    # 验证客户端是否正常工作
                    tools = self.client.list_tools()
                    self.logger.info(f"Successfully initialized client with {len(tools)} available tools")
                    return
                    
                except Exception as e:
                    self.logger.error(f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                    self.logger.debug(traceback.format_exc())
                    
                    if time.time() - start_time > self.initialization_timeout:
                        self.logger.error("Initialization timeout exceeded")
                        break
                        
                    if attempt < self.max_retries - 1:
                        self.logger.info(f"Waiting {self.retry_delay} seconds before next attempt...")
                        time.sleep(self.retry_delay)
                    else:
                        self.logger.error("All initialization attempts failed")
                        self.client = None

    def extract_search_query(self, response: str) -> Optional[str]:
        """Extract search query from response if it contains WebSearch tags.

        Args:
            response: The response string to process

        Returns:
            The search query if found, None otherwise
        """
        search_match = re.search(r'<WebSearch>(.*?)</WebSearch>', response)
        return search_match.group(1) if search_match else None

    def perform_web_search(self, query: str) -> str:
        """Perform web search using the provided query.

        Args:
            query: The search query

        Returns:
            The search results
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # 确保客户端已初始化
                if self.client is None:
                    self.initialize_client()
                
                if self.client is None:
                    return f"Error: Search client could not be initialized. Using default response for '{query}'."

                # 执行搜索
                result = self.client.call_tool("FirecrawlInternetSearch", {"keyword": query})
                return result
                
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Web search attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                self.logger.debug(traceback.format_exc())
                
                if time.time() - start_time > self.initialization_timeout:
                    self.logger.error("Search timeout exceeded")
                    break
                    
                if attempt < self.max_retries - 1:
                    self.logger.info(f"Waiting {self.retry_delay} seconds before next attempt...")
                    time.sleep(self.retry_delay)
                    # 重置客户端以重新初始化
                    self.client = None
                else:
                    self.logger.error("All web search attempts failed")

        return f"Error performing web search for '{query}': {last_error}"

