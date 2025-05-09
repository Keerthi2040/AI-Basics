from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
from bs4 import BeautifulSoup
#fetch webcontent

# Logging and Configuration
import logging
import argparse
import sys

class SearchSummarizationAgent:
    def __init__(self, model="llama3.2:1b"):
        # Initialize DuckDuckGo search
        self.search = DuckDuckGoSearchResults(backend="news")
        
        # Initialize Ollama LLM
        self.llm = ChatOllama(model=model, temperature=0.3)
        
        # Summarization prompt
        self.summarization_prompt = PromptTemplate.from_template(
            """You are an expert summarizer. Synthesize the following web search results 
            into a comprehensive, concise summary that captures the key points:

            Search Query: {query}

            Web Search Results:
            {search_results}

            Summary Guidelines:
            - Provide a clear, objective overview
            - Highlight the most important information
            - Be concise but thorough
            - Use a neutral, informative tone

            Detailed Summary:"""
        )

    def search_web(self, query, num_results=5):
        """Perform web search using DuckDuckGo"""
        try:
            # Invoke search and parse results
            raw_results = self.search.invoke(query)
            
            # Parse the results into a list of dictionaries
            results = []
            for result in raw_results.split('\n'):
                if result.strip():
                    parts = result.split(', ')
                    result_dict = {}
                    for part in parts:
                        if ':' in part:
                            key, value = part.split(':', 1)
                            result_dict[key.strip().lower()] = value.strip()
                    
                    if result_dict:
                        results.append(result_dict)
                
                if len(results) == num_results:
                    break
            
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def fetch_web_content(self, urls):
        """Fetch and clean web page content"""
        all_content = []
        
        for url in urls:
            try:
                # Fetch webpage
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract main content (you might need to adjust this)
                content = soup.get_text(separator=' ', strip=True)
                
                # Basic content cleaning
                content = ' '.join(content.split())
                all_content.append(content)
            
            except Exception as e:
                print(f"Error fetching {url}: {e}")
        
        return ' '.join(all_content)

    def summarize_results(self, query, search_results):
        """Summarize search results"""
        # Extract URLs from search results
        urls = [result.get('link', '') for result in search_results if 'link' in result]
        
        # Fetch web content
        web_content = self.fetch_web_content(urls)
        
        # Create summarization chain
        summarization_chain = (
            self.summarization_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Generate summary
        summary = summarization_chain.invoke({
            "query": query,
            "search_results": web_content
        })
        
        return summary

    def process_query(self, query):
        """Main method to process a user query"""
        # Step 1: Web Search
        search_results = self.search_web(query)
        
        # Step 2: Summarize Results
        summary = self.summarize_results(query, search_results)
        
        return {
            "query": query,
            "search_results": search_results,
            "summary": summary
        }

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler('search_agent.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="AI-Powered Web Search Agent")
    parser.add_argument(
        '--model', 
        default='llama3', 
        help='Ollama model to use for summarization'
    )
    parser.add_argument(
        '--results', 
        type=int, 
        default=5, 
        help='Number of search results to retrieve'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true', 
        help='Enable verbose output'
    )
    return parser.parse_args()

def interactive_search_agent(agent, args):
    """Interactive search agent interface"""
    print("\n🌐 AI Search Agent Activated 🔍")
    print("Type your query or 'quit' to exit.\n")

    while True:
        try:
            # Get user query
            query = input("🔎 Enter search query: ").strip()
            
            # Exit conditions
            if query.lower() in ['quit', 'exit', 'q', ':q']:
                print("\nThank you for using AI Search Agent. Goodbye! 👋")
                break
            
            # Validate query
            if not query:
                print("❌ Please enter a valid search query.")
                continue
            
            # Perform search and summarization
            result = agent.process_query(query)
            
            # Display results
            print("\n" + "="*50)
            print(f"🔍 Query: {result['query']}")
            print("="*50)
            
            # Verbose mode: Show search result links
            if args.verbose:
                print("\n📌 Top Search Results:")
                for i, res in enumerate(result['search_results'], 1):
                    print(f"{i}. {res.get('link', 'N/A')}")
            
            # Display summary
            print("\n📝 Summary:")
            print(result['summary'])
            print("\n" + "="*50 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n🛑 Search interrupted. Returning to main menu.")
        
        except Exception as e:
            print(f"❌ An error occurred: {e}")

def main():
    """Main application entry point"""
    # Setup logging
    logger = setup_logging()
    
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Initialize search agent
        agent = SearchSummarizationAgent(model=args.model)
        
        # Start interactive search
        interactive_search_agent(agent, args)
    
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"An error occurred: {e}")

# Ensure script is run directly
if __name__ == "__main__":
    main()

        
