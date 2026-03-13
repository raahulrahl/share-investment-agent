# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""share-investment-agent - A Bindu AI Agent for Share Investment Analysis."""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, cast

from bindu.penguin.bindufy import bindufy
from dotenv import load_dotenv

from .agents import run_share_investment_analysis

# Load environment variables from .env file
load_dotenv()

# Global agent instance
agent: Any = None
_initialized = False
_init_lock = asyncio.Lock()

# Setup logging
_logger = logging.getLogger(__name__)


def load_config() -> dict[str, Any]:
    """Load agent config from `agent_config.json` or return defaults."""
    config_path = Path(__file__).parent / "agent_config.json"

    if config_path.exists():
        try:
            with open(config_path) as f:
                return cast(dict[str, Any], json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            _logger.warning("Failed to load config from %s", config_path, exc_info=exc)

    return {
        "name": "share-investment-agent",
        "description": "AI-powered share investment analysis assistant",
        "deployment": {
            "url": "http://127.0.0.1:3774",
            "expose": True,
            "protocol_version": "1.0.0",
        },
    }


class ShareInvestmentAgent:
    """Share Investment Agent wrapper following the spatial-agent pattern."""

    def __init__(self, model_name: str = "gpt-4o"):
        """Initialize share investment agent with model name."""
        self.model_name = model_name

        # Get API key from environment (only supports OpenRouter)
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        if openrouter_api_key:
            print(f"✅ Using OpenRouter model: {model_name}")
        else:
            error_msg = (
                "No API key provided. Set OPENROUTER_API_KEY environment variable.\n"
                "Get your key from: https://openrouter.ai/keys"
            )
            raise ValueError(error_msg)

    async def arun(self, messages: list[dict[str, str]]) -> str:
        """Run the agent with the given messages - matches spatial-agent pattern."""
        # Extract investment request from messages
        investment_request = ""
        ticker = ""

        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", "")
                investment_request = content

                # Extract ticker from message (enhanced extraction for various formats)
                import re

                # Look for 6-digit numbers in various formats: (000001), 000001, etc.
                ticker_patterns = [
                    r"\((\d{6})\)",  # (000001)
                    r"\b(\d{6})\b",  # 000001
                    r"股票代码[:]\s*(\d{6})",  # 股票代码:000001
                    r"ticker[:\s]+(\d{6})",  # ticker: 000001
                    r"stock[:\s]+(\d{6})",  # stock: 000001
                ]

                for pattern in ticker_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        ticker = match.group(1)
                        break
                break

        if not ticker:
            return "Error: No stock ticker provided. Please include a 6-digit Chinese stock code (e.g., 000001)."

        try:
            # Run the share investment analysis pipeline
            final_analysis = await run_share_investment_analysis(
                ticker=ticker, analysis_request=investment_request, model_name=self.model_name
            )

            return final_analysis
        except Exception as e:
            error_msg = f"Error during share investment analysis: {e!s}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return error_msg


async def initialize_agent() -> None:
    """Initialize share investment agent with proper model."""
    global agent

    # Get API key and model from environment
    model_name = os.getenv("MODEL_NAME", "gpt-4o")

    # Get API key from environment (only supports OpenRouter)
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if openrouter_api_key:
        agent = ShareInvestmentAgent(model_name)
        print(f"✅ Using model: {model_name}")
    else:
        error_msg = (
            "No API key provided. Set OPENROUTER_API_KEY environment variable.\n"
            "Get your key from: https://openrouter.ai/keys"
        )
        raise ValueError(error_msg)

    print("✅ Share Investment Agent initialized")


async def run_agent(messages: list[dict[str, str]]) -> Any:
    """Run the agent with the given messages - matches spatial-agent pattern."""
    global agent
    if not agent:
        error_msg = "Agent not initialized"
        raise RuntimeError(error_msg)

    # Run the agent and get response - matches spatial-agent pattern
    return await agent.arun(messages)


async def handler(messages: list[dict[str, str]]) -> Any:
    """Handle incoming agent messages with lazy initialization - matches spatial-agent pattern."""
    global _initialized

    # Lazy initialization on first call
    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing Share Investment Agent...")
            await initialize_agent()
            _initialized = True

    # Run the async agent
    result = await run_agent(messages)
    return result


async def initialize_all() -> None:
    """Initialize all agent components - alias for initialize_agent for test compatibility."""
    await initialize_agent()


async def cleanup() -> None:
    """Clean up any resources."""
    print("🧹 Cleaning up Share Investment Agent resources...")


def main():
    """Run the main entry point for the Share Investment Agent."""
    parser = argparse.ArgumentParser(description="Bindu Share Investment Analysis Agent")
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "gpt-4o"),
        help="Model name to use (env: MODEL_NAME, default: gpt-4o)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to agent_config.json (optional)",
    )
    args = parser.parse_args()

    # Set environment variables if provided via CLI
    if args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key
    if args.model:
        os.environ["MODEL_NAME"] = args.model

    print("🤖 Share Investment Analysis Agent - AI Investment Research")
    print("📊 Capabilities: Technical Analysis, Fundamental Analysis, Sentiment Analysis, Valuation Analysis")
    print("👥 Specialist Team: Market Data, Technical, Fundamental, Sentiment, Valuation, Portfolio Management")

    # Load configuration
    config = load_config()

    try:
        # Bindufy and start the agent server
        print("🚀 Starting Bindu Share Investment Agent server...")
        print(f"🌐 Server will run on: {config.get('deployment', {}).get('url', 'http://127.0.0.1:3774')}")
        bindufy(config, handler)
    except KeyboardInterrupt:
        print("\n🛑 Share Investment Agent stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup on exit
        asyncio.run(cleanup())


# Bindufy and start the agent server
if __name__ == "__main__":
    main()
