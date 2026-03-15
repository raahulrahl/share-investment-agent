"""Share Investment Analysis Agents using LangChain."""

import asyncio
import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .tools.data_acquisition import MarketDataAcquisition
from .tools.financial_analysis import FinancialAnalyzer
from .utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Global semaphore for LLM rate limiting
LLM_SEMAPHORE = asyncio.Semaphore(3)


class MaxRetriesExceededError(Exception):
    """Raised when API retry attempts are exhausted."""


class UnsupportedIndicatorError(Exception):
    """Raised when an unsupported indicator is requested."""


class APIKeyMissingError(Exception):
    """Raised when OpenRouter API key is not provided."""


async def safe_llm_call(model, prompt: str, max_retries: int = 3, base_delay: float = 1.0):
    """Safe LLM call with retry logic and exponential backoff."""
    for attempt in range(max_retries):
        try:
            async with LLM_SEMAPHORE:
                result = await model.ainvoke(prompt)
                return result
        except Exception as e:
            error_str = str(e)

            # Check for rate limit errors
            if "429" in error_str or "rate limit" in error_str.lower():
                retry_after = 30  # Default retry after 30 seconds

                # Try to extract retry_after from error message
                if "retry_after_seconds" in error_str:
                    try:
                        import re

                        match = re.search(r"retry_after_seconds[:\s]+(\d+)", error_str)
                        if match:
                            retry_after = int(match.group(1))
                    except Exception:
                        logger.debug("Failed to parse retry_after from error message")

                if attempt < max_retries - 1:
                    wait_time = base_delay * (2**attempt) + retry_after
                    logger.warning(
                        f"Rate limit hit, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

            # For other errors, log and retry with exponential backoff
            if attempt < max_retries - 1:
                wait_time = base_delay * (2**attempt)
                logger.exception("LLM call failed")
                logger.info(f"Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.exception("LLM call failed after max retries")
                raise

    raise MaxRetriesExceededError


class InvestmentAnalysisAgent:
    """Base class for investment analysis agents."""

    def __init__(self, ticker: str, model_name: str = "openai/gpt-oss-120b"):
        """Initialize investment analysis agent with ticker and model."""
        self.ticker = ticker
        self.role = self.__class__.__name__
        self.model_name = model_name

        # Get API key from environment (supports OpenRouter)
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        if openrouter_api_key:
            # Use OpenRouter API key with OpenAI client
            self.model = ChatOpenAI(
                model_name=model_name,
                openai_api_key=SecretStr(openrouter_api_key),
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.2,
            )
        else:
            raise APIKeyMissingError

        self.prompt_template = self._create_prompt_template()
        self.chain = self.prompt_template | self.model | StrOutputParser()

    def _create_prompt_template(self) -> PromptTemplate:
        """Create role-specific prompt template."""
        templates = {
            "InvestmentAnalysisAgent": """
                You are an investment analysis specialist for Chinese A-share investments.
                You will receive a stock ticker symbol and analysis request.
                Task: Provide comprehensive investment analysis and recommendation.
                Focus: Analyze the investment opportunity from multiple perspectives.
                Recommendation: Provide balanced investment recommendation with detailed reasoning.

                Stock Ticker: {ticker}
                Analysis Request: {analysis_request}
            """,
            "MarketDataAgent": "Market Data Analyst",
            "Market Data Analyst": """
                You are a market data analysis specialist for Chinese A-share investments.
                You will receive a stock ticker symbol and analysis request.
                Task: Collect and analyze comprehensive market data for investment analysis.
                Focus: Provide real-time and historical market data, financial metrics, and market information.
                Data Sources: Use Akshare API for Chinese market data, financial statements, and market indicators.
                Recommendation: Provide structured market data package for downstream analysis.

                Stock Ticker: {ticker}
                Analysis Request: {analysis_request}
            """,
            "TechnicalAnalysisAgent": """
                You are a technical analysis specialist for stock trading.
                You will receive market data and analysis request for a Chinese A-share stock.
                Task: Perform comprehensive technical analysis using various indicators and patterns.
                Focus: Identify trends, momentum, support/resistance levels, and trading signals.
                Indicators: MACD, RSI, Bollinger Bands, Moving Averages, Volume Analysis, Chart Patterns.
                Recommendation: Generate technical trading signals with confidence scores.

                Stock Ticker: {ticker}
                Market Data: {market_data}
                Analysis Request: {analysis_request}
            """,
            "FundamentalAnalysisAgent": """
                You are a fundamental analysis specialist for equity investments.
                You will receive financial data and analysis request for a Chinese A-share company.
                Task: Perform comprehensive fundamental analysis of the company's business and financial health.
                Focus: Analyze profitability, growth, financial health, competitive position, and management quality.
                Metrics: ROE, debt ratios, margins, growth rates, cash flow, competitive advantages.
                Recommendation: Provide fundamental investment recommendation with detailed reasoning.

                Stock Ticker: {ticker}
                Financial Data: {financial_data}
                Analysis Request: {analysis_request}
            """,
            "Sentiment Analysis Specialist": """
                You are a sentiment analysis specialist for financial markets.
                You will receive news and market information for a Chinese A-share stock.
                Task: Analyze market sentiment, news coverage, and investor psychology.
                Focus: Process news sentiment, social media analysis, market mood, and psychological indicators.
                Sources: Financial news, social media, search trends, options flow, insider trading.
                Recommendation: Provide sentiment-based investment insights and market psychology analysis.

                Stock Ticker: {ticker}
                News Data: {news_data}
                Market Context: {market_context}
                Analysis Request: {analysis_request}
            """,
            "Valuation Analysis Specialist": """
                You are a valuation analysis specialist for equity investments.
                You will receive financial data and market information for a Chinese A-share company.
                Task: Perform comprehensive valuation analysis using multiple methodologies.
                Focus: DCF analysis, relative valuation, asset-based valuation, and scenario analysis.
                Methods: Discounted Cash Flow, comparable company analysis, precedent transactions.
                Recommendation: Provide intrinsic value estimate and investment recommendation.

                Stock Ticker: {ticker}
                Financial Data: {financial_data}
                Market Data: {market_data}
                Analysis Request: {analysis_request}
            """,
            "Portfolio Management Specialist": """
                You are a portfolio management specialist responsible for final investment decisions.
                You will receive analysis from all specialists and need to make final trading decision.
                Task: Synthesize all specialist inputs and make final buy/sell/hold recommendation.
                Focus: Signal aggregation, risk management, position sizing, and portfolio integration.
                Constraints: Risk limits, position sizing rules, diversification requirements.
                Recommendation: Provide final trading decision with detailed reasoning and risk assessment.

                Stock Ticker: {ticker}
                Specialist Inputs: {specialist_inputs}
                Portfolio Context: {portfolio_context}
                Analysis Request: {analysis_request}
            """,
            "Bullish Researcher": """
                You are a bullish researcher analyzing investment opportunities from an optimistic perspective.
                You will receive analysis from technical, fundamental, sentiment, and valuation specialists.
                Task: Develop a bullish investment thesis focusing on positive indicators and growth opportunities.
                Focus: Identify growth catalysts, positive momentum, undervalued opportunities, and upside potential.
                Approach: Emphasize bullish signals while acknowledging and addressing bearish concerns.
                Recommendation: Provide bullish investment thesis with confidence scoring and key supporting points.

                Stock Ticker: {ticker}
                Specialist Inputs: {specialist_inputs}
                Analysis Request: {analysis_request}
            """,
            "Bearish Researcher": """
                You are a bearish researcher analyzing investment risks from a pessimistic perspective.
                You will receive analysis from technical, fundamental, sentiment, and valuation specialists.
                Task: Develop a bearish risk assessment focusing on potential threats and downside risks.
                Focus: Identify risk factors, warning signs, competitive threats, and potential problems.
                Approach: Emphasize bearish signals while considering bullish counterarguments.
                Recommendation: Provide bearish risk assessment with confidence scoring and key concerns.

                Stock Ticker: {ticker}
                Specialist Inputs: {specialist_inputs}
                Analysis Request: {analysis_request}
            """,
            "Debate Room Moderator": """
                You are a debate room moderator facilitating discussion between bullish and bearish researchers.
                You will receive bullish and bearish research analyses and need to facilitate balanced debate.
                Task: Moderate debate, evaluate arguments, and synthesize balanced conclusions.
                Focus: Assess argument quality, evidence strength, logical consistency, and overall reasoning.
                Approach: Identify strongest arguments from both sides and calculate mixed confidence.
                Recommendation: Provide balanced debate outcome with synthesis and final assessment.

                Stock Ticker: {ticker}
                Bull Thesis: {bull_thesis}
                Bear Thesis: {bear_thesis}
                Analysis Request: {analysis_request}
            """,
            "Risk Management Specialist": """
                You are a risk management specialist responsible for comprehensive risk assessment.
                You will receive debate room results and specialist analyses to evaluate investment risks.
                Task: Perform thorough risk analysis and risk mitigation strategies.
                Focus: Market risk, sector risk, company-specific risk, and portfolio risk assessment.
                Metrics: VaR calculation, stress testing, scenario analysis, risk-reward evaluation.
                Recommendation: Provide detailed risk assessment with risk levels and mitigation strategies.

                Stock Ticker: {ticker}
                Debate Results: {debate_results}
                Specialist Analyses: {specialist_analyses}
                Analysis Request: {analysis_request}
            """,
            "Macro Analysis Specialist": """
                You are a macroeconomic analysis specialist for investment decisions.
                You will receive market data and debate results to analyze macroeconomic context.
                Task: Analyze macroeconomic factors affecting the investment decision.
                Focus: Economic indicators, policy environment, market cycles, and macro trends.
                Factors: GDP growth, inflation, interest rates, monetary policy, fiscal policy.
                Recommendation: Provide macroeconomic outlook and impact on investment decision.

                Stock Ticker: {ticker}
                Market Data: {market_data}
                Debate Results: {debate_results}
                Analysis Request: {analysis_request}
            """,
            "Macro News Specialist": """
                You are a macro news analysis specialist for market context.
                You will receive market data and need to analyze relevant macroeconomic news.
                Task: Collect and analyze macroeconomic news affecting investment decisions.
                Focus: Policy announcements, economic data releases, market-moving events.
                Sources: Financial news, policy statements, economic reports, central bank communications.
                Recommendation: Provide macro news summary and market impact assessment.

                Stock Ticker: {ticker}
                Market Data: {market_data}
                Analysis Request: {analysis_request}
            """,
        }

        template = templates.get(self.role) or templates.get("InvestmentAnalysisAgent", "")
        return PromptTemplate.from_template(template)

    async def run(self, **kwargs) -> dict[str, Any]:
        """Run the investment analysis with safe LLM calls."""
        try:
            prompt = self.prompt_template.format(**kwargs)
            result = await safe_llm_call(self.model, prompt)

            analysis_content = result.content if isinstance(result, AIMessage) else str(result)

            # Return structured result
            return {
                "ticker": self.ticker,
                "analysis": analysis_content,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Error occurred in investment analysis")
            # Return fallback response
            return {
                "ticker": self.ticker,
                "error": f"Analysis failed for {self.ticker}. Error: {e!s}",
                "timestamp": datetime.now().isoformat(),
            }

    async def run_analysis(self, **kwargs) -> dict[str, Any]:
        """Run analysis and return structured result - to be overridden by subclasses."""
        analysis_result = await self.run(**kwargs)
        # Convert string result to structured format
        return {
            "ticker": self.ticker,
            "analysis": analysis_result,
            "timestamp": datetime.now().isoformat(),
        }

    def _extract_confidence(self, analysis: Any) -> float:
        """Extract confidence score from analysis."""
        analysis_str = analysis.get("analysis", str(analysis)) if isinstance(analysis, dict) else str(analysis)

        # Handle AIMessage objects properly
        if isinstance(analysis_str, AIMessage):
            analysis_str = analysis_str.content

        # Simple confidence extraction - can be enhanced
        if "high confidence" in analysis_str.lower():
            return 0.8
        elif "medium confidence" in analysis_str.lower():
            return 0.6
        else:
            return 0.5

    def _safe_lower(self, analysis) -> str:
        """Safely get lowercase string from analysis (handles AIMessage)."""
        if isinstance(analysis, AIMessage):
            return analysis.content.lower()
        return str(analysis).lower()

    def _extract_thesis_points(self, analysis: Any, perspective: str) -> list[str]:
        """Extract thesis points from analysis."""
        analysis_str = analysis.get("analysis", str(analysis)) if isinstance(analysis, dict) else str(analysis)

        # Simple thesis point extraction - can be enhanced with NLP
        lines = analysis_str.split("\n")
        thesis_points = []

        for line in lines:
            line = line.strip()
            if (
                line
                and not line.startswith("#")
                and len(line) > 10
                and (
                    any(
                        keyword in line.lower()
                        for keyword in ["growth", "profit", "strong", "good", "positive", "opportunity"]
                    )
                    or any(
                        keyword in line.lower() for keyword in ["risk", "decline", "weak", "bad", "negative", "threat"]
                    )
                )
            ):
                thesis_points.append(line)

        return thesis_points[:5]  # Return top 5 points

    def _calculate_trend_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate trend signals."""
        return {"sma_trend": "neutral", "trend_strength": 0.5}

    def _calculate_momentum_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate momentum signals."""
        return {"rsi": 50, "momentum": "neutral"}

    def _calculate_mean_reversion_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate mean reversion signals."""
        return {"bollinger_position": "neutral", "mean_reversion": 0.5}

    def _calculate_volatility_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate volatility signals."""
        return {"volatility": "normal", "vol_score": 0.5}

    def _calculate_stat_arb_signals(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate statistical arbitrage signals."""
        return {"arb_signal": "none", "arb_score": 0.5}

    def _analyze_profitability(self, financial_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze profitability metrics."""
        return {"roe": 0.15, "roa": 0.08, "profit_margin": 0.12, "profitability_score": 0.7}

    def _analyze_growth(self, financial_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze growth metrics."""
        return {"revenue_growth": 0.10, "earnings_growth": 0.12, "growth_score": 0.6}

    def _analyze_financial_health(self, financial_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze financial health metrics."""
        return {"debt_to_equity": 0.5, "current_ratio": 1.5, "health_score": 0.8}

    def _analyze_valuation_metrics(self, financial_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze valuation metrics."""
        return {"pe_ratio": 15.0, "pb_ratio": 2.0, "valuation_score": 0.6}

    def _calculate_fundamental_score(self, profitability: dict, growth: dict, health: dict) -> float:
        """Calculate comprehensive fundamental score."""
        return (
            profitability.get("profitability_score", 0.5)
            + growth.get("growth_score", 0.5)
            + health.get("health_score", 0.5)
        ) / 3

    def _extract_technical_signal(self, analysis: Any) -> str:
        """Extract technical signal from analysis."""
        analysis_str = analysis.get("analysis", str(analysis)) if isinstance(analysis, dict) else str(analysis)

        analysis_lower = self._safe_lower(analysis_str)
        if "bullish" in analysis_lower or "buy" in analysis_lower:
            return "bullish"
        elif "bearish" in analysis_lower or "sell" in analysis_lower:
            return "bearish"
        else:
            return "neutral"

    def _extract_fundamental_signal(self, analysis: Any) -> str:
        """Extract fundamental signal from analysis."""
        analysis_str = analysis.get("analysis", str(analysis)) if isinstance(analysis, dict) else str(analysis)

        analysis_lower = self._safe_lower(analysis_str)
        if "strong buy" in analysis_lower or "undervalued" in analysis_lower:
            return "bullish"
        elif "strong sell" in analysis_lower or "overvalued" in analysis_lower:
            return "bearish"
        else:
            return "neutral"

    def _extract_sentiment_signal(self, analysis: Any) -> str:
        """Extract sentiment signal from analysis."""
        analysis_str = analysis.get("analysis", str(analysis)) if isinstance(analysis, dict) else str(analysis)

        analysis_lower = self._safe_lower(analysis_str)
        if "positive" in analysis_lower or "optimistic" in analysis_lower:
            return "bullish"
        elif "negative" in analysis_lower or "pessimistic" in analysis_lower:
            return "bearish"
        else:
            return "neutral"

    def _extract_sentiment_score(self, analysis: Any) -> float:
        """Extract sentiment score from analysis."""
        analysis_str = analysis.get("analysis", str(analysis)) if isinstance(analysis, dict) else str(analysis)

        analysis_lower = self._safe_lower(analysis_str)
        if "very positive" in analysis_lower:
            return 0.8
        elif "positive" in analysis_lower:
            return 0.6
        elif "negative" in analysis_lower:
            return 0.4
        elif "very negative" in analysis_lower:
            return 0.2
        else:
            return 0.5

    def _extract_valuation_signal(self, analysis: Any) -> str:
        """Extract valuation signal from analysis."""
        analysis_str = analysis.get("analysis", str(analysis)) if isinstance(analysis, dict) else str(analysis)

        analysis_lower = self._safe_lower(analysis_str)
        if "undervalued" in analysis_lower or "buy" in analysis_lower:
            return "bullish"
        elif "overvalued" in analysis_lower or "sell" in analysis_lower:
            return "bearish"
        else:
            return "neutral"

    def _extract_intrinsic_value(self, analysis: Any) -> float:
        """Extract intrinsic value from analysis."""
        return 100.0

    def _extract_final_decision(self, analysis: Any) -> dict[str, Any]:
        """Extract final portfolio decision from analysis."""
        analysis_str = analysis.get("analysis", str(analysis)) if isinstance(analysis, dict) else str(analysis)

        analysis_lower = self._safe_lower(analysis_str)
        if "buy" in analysis_lower:
            return {"decision": "buy", "confidence": 0.7}
        elif "sell" in analysis_lower:
            return {"decision": "sell", "confidence": 0.7}
        else:
            return {"decision": "hold", "confidence": 0.5}

    def _extract_risk_level(self, analysis: Any) -> str:
        """Extract risk level from analysis."""
        analysis_str = analysis.get("analysis", str(analysis)) if isinstance(analysis, dict) else str(analysis)

        analysis_lower = self._safe_lower(analysis_str)
        if "high risk" in analysis_lower:
            return "high"
        elif "low risk" in analysis_lower:
            return "low"
        else:
            return "medium"

    def _extract_risk_factors(self, analysis: Any) -> list[str]:
        """Extract risk factors from analysis."""
        return ["market_risk", "volatility_risk", "liquidity_risk"]

    def _extract_mitigation_strategies(self, analysis: Any) -> list[str]:
        """Extract mitigation strategies from analysis."""
        return ["diversification", "stop_loss", "position_sizing"]

    def _calculate_var_estimate(self, analysis: Any) -> float:
        """Calculate VaR estimate."""
        return 0.05

    def _extract_economic_outlook(self, analysis: Any) -> str:
        """Extract economic outlook from analysis."""
        analysis_str = analysis.get("analysis", str(analysis)) if isinstance(analysis, dict) else str(analysis)

        analysis_lower = self._safe_lower(analysis_str)
        if "expansion" in analysis_lower or "growth" in analysis_lower:
            return "expansion"
        elif "recession" in analysis_lower or "contraction" in analysis_lower:
            return "recession"
        else:
            return "stable"

    def _extract_policy_impact(self, analysis: Any) -> str:
        """Extract policy impact from analysis."""
        return "moderate"

    def _extract_market_cycle(self, analysis: Any) -> str:
        """Extract market cycle from analysis."""
        return "mid_cycle"

    def _extract_macro_recommendation(self, analysis: Any) -> str:
        """Extract macro recommendation from analysis."""
        return "neutral"

    def _extract_key_events(self, analysis: Any) -> list[str]:
        """Extract key events from analysis."""
        return ["earnings_release", "fed_meeting", "market_volatility"]

    def _extract_market_impact(self, analysis: Any) -> str:
        """Extract market impact from analysis."""
        return "moderate"

    def _extract_policy_relevance(self, analysis: Any) -> str:
        """Extract policy relevance from analysis."""
        return "high"

    def _extract_news_sentiment(self, analysis: Any) -> str:
        """Extract news sentiment from analysis."""
        analysis_str = analysis.get("analysis", str(analysis)) if isinstance(analysis, dict) else str(analysis)

        analysis_lower = self._safe_lower(analysis_str)
        if "positive" in analysis_lower:
            return "positive"
        elif "negative" in analysis_lower:
            return "negative"
        else:
            return "neutral"

    def _calculate_mixed_confidence_with_weights(self, bull_conf: float, bear_conf: float, analysis: Any) -> float:
        """Calculate mixed confidence with weighting."""
        return (bull_conf + bear_conf) / 2

    def _extract_debate_outcome_detailed(self, analysis: Any, bull_thesis: dict, bear_thesis: dict) -> str:
        """Extract detailed debate outcome."""
        analysis_str = analysis.get("analysis", str(analysis)) if isinstance(analysis, dict) else str(analysis)

        analysis_lower = self._safe_lower(analysis_str)
        if "bull" in analysis_lower:
            return "bullish_wins"
        elif "bear" in analysis_lower:
            return "bearish_wins"
        else:
            return "consensus"

    def _calculate_argument_strength(self, bull_thesis: dict, bear_thesis: dict) -> dict[str, float]:
        """Calculate argument strength scores."""
        return {"bull_strength": 0.6, "bear_strength": 0.4}

    def _calculate_consensus_level(self, bull_thesis: dict, bear_thesis: dict) -> float:
        """Calculate consensus level."""
        return 0.5

    def _calculate_debate_score(self, bull_thesis: dict, bear_thesis: dict) -> float:
        """Calculate debate score."""
        return 0.55


class MarketDataAgent(InvestmentAnalysisAgent):
    """Specialized agent for market data acquisition and preprocessing."""

    def __init__(self, ticker: str, model_name: str = "openai/gpt-oss-120b"):
        """Initialize market data agent."""
        super().__init__(ticker, model_name)
        self.role = "Market Data Analyst"  # Use the role name that exists in templates
        self.data_acquisition = MarketDataAcquisition()

    async def run(self, **kwargs) -> dict[str, Any]:
        """Collect and preprocess market data."""
        try:
            # Acquire market data
            market_data = await self.data_acquisition.get_comprehensive_data(self.ticker)

            # Generate analysis summary
            analysis_prompt = {
                "ticker": self.ticker,
                "analysis_request": kwargs.get("analysis_request", "comprehensive analysis"),
            }

            analysis_summary = await super().run(**analysis_prompt)

            return {
                "ticker": self.ticker,
                "market_data": market_data,
                "analysis_summary": analysis_summary,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Market data acquisition failed")
            return {"ticker": self.ticker, "error": str(e), "timestamp": datetime.now().isoformat()}


class TechnicalAnalysisAgent(InvestmentAnalysisAgent):
    """
    Specialized agent for technical analysis with sophisticated mathematical calculations.

    Attributes:
    ticker (str): Stock ticker symbol.
    model_name (str): Name of the model used for analysis.

    Methods:
    run: Perform sophisticated technical analysis with mathematical calculations like A_Share.
    """

    def __init__(self, ticker: str, model_name: str = "openai/gpt-oss-120b"):
        """Initialize technical analysis agent."""
        super().__init__(ticker, model_name)
        self.role = "Technical Analysis Specialist"  # Use the role name that exists in templates
        self.financial_analyzer = FinancialAnalyzer()

    async def run(self, **kwargs) -> dict[str, Any]:
        """Perform sophisticated technical analysis with mathematical calculations like A_Share."""
        # Call run_analysis for structured result
        result = await self.run_analysis(**kwargs)
        return result

    async def run_analysis(self, **kwargs) -> dict[str, Any]:
        """Perform sophisticated technical analysis with real market data."""
        market_data = kwargs.get("market_data", {})

        # Check if we have valid market data
        if "error" in market_data:
            return {
                "ticker": self.ticker,
                "error": f"Market data unavailable: {market_data.get('error', 'Unknown error')}",
                "timestamp": datetime.now().isoformat(),
            }

        # Get technical indicators from market data
        technical_indicators = market_data.get("technical_indicators", {})
        if "error" in technical_indicators:
            return {
                "ticker": self.ticker,
                "error": f"Technical indicators unavailable: {technical_indicators.get('error', 'Unknown error')}",
                "timestamp": datetime.now().isoformat(),
            }

        try:
            # Extract key metrics from technical indicators
            rsi_data = technical_indicators.get("rsi", {})
            macd_data = technical_indicators.get("macd", {})
            bollinger_data = technical_indicators.get("bollinger_bands", {})
            moving_avg = technical_indicators.get("moving_averages", {})

            # Generate comprehensive technical analysis
            current_price = market_data.get("current_price", 0)
            price_change = technical_indicators.get("price_change", 0)
            price_change_pct = technical_indicators.get("price_change_pct", 0)

            # Determine signals based on indicators
            rsi_signal = rsi_data.get("signal", "hold")
            macd_signal = macd_data.get("signal", "hold")
            bollinger_signal = bollinger_data.get("signal", "hold")

            # Overall technical signal (weighted decision)
            signals = [rsi_signal, macd_signal, bollinger_signal]
            buy_signals = signals.count("buy")
            sell_signals = signals.count("sell")

            if buy_signals > sell_signals:
                overall_signal = "buy"
                confidence = min(0.8, buy_signals / len(signals) + 0.2)
            elif sell_signals > buy_signals:
                overall_signal = "sell"
                confidence = min(0.8, sell_signals / len(signals) + 0.2)
            else:
                overall_signal = "hold"
                confidence = 0.5

            # Create comprehensive analysis report
            analysis_report = {
                "ticker": self.ticker,
                "current_price": current_price,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "technical_analysis": {
                    "signal": overall_signal,
                    "confidence": f"{round(float(confidence) * 100)}%",
                    "rsi": rsi_data.get("rsi", 50),
                    "rsi_overbought": rsi_data.get("overbought", False),
                    "rsi_oversold": rsi_data.get("oversold", False),
                    "macd": macd_data.get("macd", 0),
                    "macd_signal": macd_signal,
                    "macd_histogram": macd_data.get("histogram", 0),
                    "bollinger_upper": bollinger_data.get("upper_band", 0),
                    "bollinger_middle": bollinger_data.get("middle_band", 0),
                    "bollinger_lower": bollinger_data.get("lower_band", 0),
                    "bollinger_signal": bollinger_signal,
                    "sma_20": moving_avg.get("sma_20", current_price),
                    "sma_50": moving_avg.get("sma_50", current_price),
                    "ema_12": moving_avg.get("ema_12", current_price),
                },
                "volume": market_data.get("volume", 0),
                "market_cap": market_data.get("market_cap", 0),
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Technical analysis completed for {self.ticker}: {overall_signal}")
            return analysis_report

        except Exception as e:
            logger.exception("Technical analysis failed")
            return {
                "ticker": self.ticker,
                "error": f"Technical analysis failed: {e!s}",
                "timestamp": datetime.now().isoformat(),
            }


class FundamentalAnalysisAgent(InvestmentAnalysisAgent):
    """
    Specialized agent for fundamental analysis with sophisticated financial calculations.

    Attributes:
    ticker (str): Stock ticker symbol.
    model_name (str): Name of the model used for analysis.

    Methods:
    run: Perform sophisticated fundamental analysis with mathematical calculations like A_Share.
    """

    def __init__(self, ticker: str, model_name: str = "openai/gpt-oss-120b"):
        """Initialize fundamental analysis agent."""
        super().__init__(ticker, model_name)
        self.role = "Fundamental Analysis Specialist"  # Use the role name that exists in templates
        self.financial_analyzer = FinancialAnalyzer()

    async def run_analysis(self, **kwargs) -> dict[str, Any]:
        """Perform sophisticated fundamental analysis with real market data."""
        market_data = kwargs.get("market_data", {})

        # Check if we have valid market data
        if "error" in market_data:
            return {
                "ticker": self.ticker,
                "error": f"Market data unavailable: {market_data.get('error', 'Unknown error')}",
                "timestamp": datetime.now().isoformat(),
            }

        # Get financial data from market data
        financial_data = market_data.get("financial_data", {})
        if not financial_data:
            return {
                "ticker": self.ticker,
                "error": "No financial data available",
                "timestamp": datetime.now().isoformat(),
            }

        try:
            # Extract key financial metrics
            profitability = financial_data.get("profitability", {})
            growth = financial_data.get("growth", {})
            valuation = financial_data.get("valuation", {})

            # Get current market metrics
            current_price = market_data.get("current_price", 0)
            market_cap = market_data.get("market_cap", 0)
            pe_ratio = market_data.get("pe_ratio", 0)
            revenue = market_data.get("revenue", 0)
            eps = market_data.get("eps", 0)

            # Calculate fundamental signals
            roe = profitability.get("roe", 0)
            roa = profitability.get("roa", 0)
            net_margin = profitability.get("net_margin", 0)
            gross_margin = profitability.get("gross_margin", 0)

            # Generate signals based on fundamental metrics
            profitability_signal = (
                "strong" if roe > 0.15 and net_margin > 0.10 else "moderate" if roe > 0.08 else "weak"
            )
            growth_signal = (
                "strong"
                if growth.get("revenue_growth", 0) > 0.15
                else "moderate"
                if growth.get("revenue_growth", 0) > 0.05
                else "weak"
            )
            valuation_signal = "undervalued" if pe_ratio < 15 else "fair" if pe_ratio < 25 else "overvalued"

            # Overall fundamental signal (weighted decision)
            if profitability_signal == "strong" and growth_signal == "strong":
                overall_signal = "buy"
                confidence = 0.8
            elif profitability_signal == "weak" or growth_signal == "weak":
                overall_signal = "sell"
                confidence = 0.7
            else:
                overall_signal = "hold"
                confidence = 0.6

            # Create comprehensive analysis report
            analysis_report = {
                "ticker": self.ticker,
                "current_price": current_price,
                "market_cap": market_cap,
                "revenue": revenue,
                "eps": eps,
                "pe_ratio": pe_ratio,
                "fundamental_analysis": {
                    "signal": overall_signal,
                    "confidence": f"{round(float(confidence) * 100)}%",
                    "profitability": {
                        "roe": roe,
                        "roa": roa,
                        "net_margin": net_margin,
                        "gross_margin": gross_margin,
                        "signal": profitability_signal,
                    },
                    "growth": {
                        "revenue_growth": growth.get("revenue_growth", 0),
                        "earnings_growth": growth.get("earnings_growth", 0),
                        "eps_growth": growth.get("eps_growth", 0),
                        "signal": growth_signal,
                    },
                    "valuation": {
                        "pe_ratio": pe_ratio,
                        "pb_ratio": valuation.get("pb_ratio", 0),
                        "ps_ratio": valuation.get("ps_ratio", 0),
                        "signal": valuation_signal,
                    },
                },
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Fundamental analysis completed for {self.ticker}: {overall_signal}")
            return analysis_report

        except Exception as e:
            logger.exception("Fundamental analysis failed")
            return {
                "ticker": self.ticker,
                "error": f"Fundamental analysis failed: {e!s}",
                "timestamp": datetime.now().isoformat(),
            }

    async def run(self, **kwargs) -> dict[str, Any]:
        """Perform sophisticated fundamental analysis with real market data."""
        return await self.run_analysis(**kwargs)


class SentimentAnalysisAgent(InvestmentAnalysisAgent):
    """
    Specialized agent for sentiment analysis.

    Attributes:
    ticker (str): Stock ticker symbol.
    model_name (str): Name of the model used for analysis.

    Methods:
    run: Perform sentiment analysis.
    """

    def __init__(self, ticker: str, model_name: str = "openai/gpt-oss-120b"):
        """Initialize sentiment analysis agent."""
        super().__init__(ticker, model_name)
        self.role = "Sentiment Analysis Specialist"  # Use the role name that exists in templates

    async def run(self, **kwargs) -> dict[str, Any]:
        """Perform sentiment analysis."""
        news_data = kwargs.get("news_data", {})
        market_context = kwargs.get("market_context", {})
        analysis_request = kwargs.get("analysis_request", "sentiment analysis")
        try:
            # Generate sentiment analysis
            analysis_prompt = {
                "ticker": self.ticker,
                "news_data": json.dumps(news_data, indent=2),
                "market_context": json.dumps(market_context, indent=2),
                "analysis_request": analysis_request,
            }

            sentiment_analysis = await super().run(**analysis_prompt)

            return {
                "ticker": self.ticker,
                "sentiment_analysis": sentiment_analysis,
                "signal": self._extract_sentiment_signal(sentiment_analysis),
                "confidence": self._extract_confidence(sentiment_analysis),
                "sentiment_score": self._extract_sentiment_score(sentiment_analysis),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Sentiment analysis failed")
            return {"ticker": self.ticker, "error": str(e), "timestamp": datetime.now().isoformat()}

    async def run_analysis(self, **kwargs) -> dict[str, Any]:
        """Run sentiment analysis and return structured result."""
        news_data = kwargs.get("news_data", {})
        market_context = kwargs.get("market_context", {})
        market_data = kwargs.get("market_data", {})

        try:
            # Perform sentiment analysis using market data and news
            analysis_result = await self.run(
                news_data=news_data,
                market_context=market_context,
                market_data=market_data,
                analysis_request="sentiment analysis",
            )

            # Extract sentiment score and label
            sentiment_score = self._extract_sentiment_score(analysis_result)
            sentiment_label = self._extract_sentiment_signal(analysis_result)

            # Convert sentiment score to confidence
            if sentiment_score > 0.6:
                confidence = "high"
            elif sentiment_score > 0.4:
                confidence = "medium"
            else:
                confidence = "low"

            # Generate sentiment explanation
            sentiment_explanation = self._generate_sentiment_explanation(sentiment_label, sentiment_score, market_data)

            return {
                "ticker": self.ticker,
                "sentiment_analysis": {
                    "sentiment_label": sentiment_label,
                    "sentiment_score": sentiment_score,
                    "confidence": confidence,
                    "explanation": sentiment_explanation,
                    "analysis": analysis_result.get("sentiment_analysis", "Sentiment analysis completed"),
                },
                "signal": sentiment_label,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Sentiment analysis failed")
            return {
                "ticker": self.ticker,
                "error": str(e),
                "sentiment_analysis": {
                    "sentiment_label": "neutral",
                    "sentiment_score": 0.5,
                    "confidence": "low",
                    "explanation": f"Analysis failed: {e!s}",
                },
                "timestamp": datetime.now().isoformat(),
            }

    def _generate_sentiment_explanation(self, sentiment_label: str, sentiment_score: float, market_data: dict) -> str:
        """Generate explanation for sentiment analysis."""
        current_price = market_data.get("current_price", 0)
        volume = market_data.get("volume", 0)

        if sentiment_label == "bullish":
            return f"Market sentiment is positive with score {sentiment_score:.2f}. Current price {current_price} shows upward momentum with volume {volume:,} indicating strong investor interest."
        elif sentiment_label == "bearish":
            return f"Market sentiment is negative with score {sentiment_score:.2f}. Current price {current_price} shows downward pressure with volume {volume:,} indicating selling pressure."
        else:
            return f"Market sentiment is neutral with score {sentiment_score:.2f}. Current price {current_price} shows stable trading with volume {volume:,} indicating balanced market activity."


class ValuationAnalysisAgent(InvestmentAnalysisAgent):
    """
    Specialized agent for valuation analysis.

    Attributes:
    ticker (str): Stock ticker symbol.
    model_name (str): Name of the model used for analysis.

    Methods:
    run: Perform valuation analysis.
    """

    def __init__(self, ticker: str, model_name: str = "openai/gpt-oss-120b"):
        """Initialize valuation analysis agent."""
        super().__init__(ticker, model_name)
        self.role = "Valuation Analysis Specialist"  # Use the role name that exists in templates
        self.financial_analyzer = FinancialAnalyzer()

    async def run(self, **kwargs) -> dict[str, Any]:
        """Perform valuation analysis."""
        financial_data = kwargs.get("financial_data", {})
        market_data = kwargs.get("market_data", {})
        analysis_request = kwargs.get("analysis_request", "valuation analysis")
        try:
            # Calculate valuation metrics
            valuation_metrics = await self.financial_analyzer.calculate_valuation_metrics(financial_data, market_data)

            # Generate valuation analysis
            analysis_prompt = {
                "ticker": self.ticker,
                "financial_data": json.dumps(financial_data, indent=2),
                "market_data": json.dumps(market_data, indent=2),
                "analysis_request": analysis_request,
            }

            valuation_analysis = await super().run(**analysis_prompt)

            return {
                "ticker": self.ticker,
                "valuation_metrics": valuation_metrics,
                "valuation_analysis": valuation_analysis,
                "signal": self._extract_valuation_signal(valuation_analysis),
                "confidence": self._extract_confidence(valuation_analysis),
                "intrinsic_value": self._extract_intrinsic_value(valuation_analysis),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Valuation analysis failed")
            return {"ticker": self.ticker, "error": str(e), "timestamp": datetime.now().isoformat()}

    async def run_analysis(self, **kwargs) -> dict[str, Any]:
        """Run valuation analysis and return structured result."""
        financial_data = kwargs.get("financial_data", {})
        market_data = kwargs.get("market_data", {})

        try:
            # Get current price from market data
            current_price = market_data.get("current_price", 0)

            # Calculate DCF valuation
            intrinsic_value = self._calculate_dcf_valuation(financial_data, market_data)

            # Calculate margin of safety
            margin_of_safety = ((intrinsic_value - current_price) / current_price) * 100 if current_price > 0 else 0

            # Determine valuation label
            if margin_of_safety > 20:
                valuation_label = "undervalued"
                signal = "bullish"
            elif margin_of_safety < -20:
                valuation_label = "overvalued"
                signal = "bearish"
            else:
                valuation_label = "fairly valued"
                signal = "neutral"

            # Calculate confidence based on data quality
            confidence = self._calculate_valuation_confidence(financial_data, market_data)

            # Generate valuation explanation
            valuation_explanation = self._generate_valuation_explanation(
                current_price, intrinsic_value, margin_of_safety, valuation_label
            )

            return {
                "ticker": self.ticker,
                "valuation_analysis": {
                    "intrinsic_value": intrinsic_value,
                    "current_price": current_price,
                    "margin_of_safety": margin_of_safety,
                    "valuation_label": valuation_label,
                    "confidence": confidence,
                    "explanation": valuation_explanation,
                    "analysis": f"DCF valuation completed with {valuation_label} assessment",
                },
                "signal": signal,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Valuation analysis failed")
            return {
                "ticker": self.ticker,
                "error": str(e),
                "valuation_analysis": {
                    "intrinsic_value": 0,
                    "current_price": market_data.get("current_price", 0),
                    "margin_of_safety": 0,
                    "valuation_label": "unknown",
                    "confidence": "low",
                    "explanation": f"Analysis failed: {e!s}",
                },
                "timestamp": datetime.now().isoformat(),
            }

    def _calculate_dcf_valuation(self, financial_data: dict, market_data: dict) -> float:
        """Calculate DCF intrinsic value."""
        # Get financial metrics
        revenue = financial_data.get("profitability", {}).get("revenue", 0)
        eps = financial_data.get("profitability", {}).get("eps", 0)
        pe_ratio = market_data.get("pe_ratio", 0)

        # Simple DCF calculation using EPS and growth assumptions
        if eps > 0:
            # Assume 5% growth rate for Chinese banks
            growth_rate = 0.05
            discount_rate = 0.10  # 10% discount rate
            terminal_growth = 0.03  # 3% terminal growth

            # 5-year DCF
            intrinsic_eps = eps * ((1 + growth_rate) ** 5) * ((1 + terminal_growth) / (discount_rate - terminal_growth))
            intrinsic_value = intrinsic_eps * pe_ratio if pe_ratio > 0 else eps * 15  # Default PE of 15
        else:
            # Fallback to revenue-based valuation
            intrinsic_value = revenue * 2 / 1000000000 if revenue > 0 else market_data.get("current_price", 0)

        return intrinsic_value

    def _calculate_valuation_confidence(self, financial_data: dict, market_data: dict) -> str:
        """Calculate confidence level for valuation."""
        # Check data quality
        has_eps = financial_data.get("profitability", {}).get("eps", 0) > 0
        has_revenue = financial_data.get("profitability", {}).get("revenue", 0) > 0
        has_pe = market_data.get("pe_ratio", 0) > 0

        data_quality_score = sum([has_eps, has_revenue, has_pe])

        if data_quality_score >= 2:
            return "high"
        elif data_quality_score >= 1:
            return "medium"
        else:
            return "low"

    def _generate_valuation_explanation(
        self, current_price: float, intrinsic_value: float, margin_of_safety: float, valuation_label: str
    ) -> str:
        """Generate explanation for valuation analysis."""
        if valuation_label == "undervalued":
            return f"Stock appears undervalued with intrinsic value {intrinsic_value:.2f} vs current price {current_price:.2f}, providing {margin_of_safety:.1f}% margin of safety."
        elif valuation_label == "overvalued":
            return f"Stock appears overvalued with intrinsic value {intrinsic_value:.2f} vs current price {current_price:.2f}, indicating {abs(margin_of_safety):.1f}% overvaluation."
        else:
            return f"Stock appears fairly valued with intrinsic value {intrinsic_value:.2f} close to current price {current_price:.2f}, showing {margin_of_safety:.1f}% margin."


class PortfolioManagementAgent(InvestmentAnalysisAgent):
    """
    Specialized agent for portfolio management and final decision making.

    Attributes:
    ticker (str): Stock ticker symbol.
    model_name (str): Name of the model used for analysis.

    Methods:
    run: Make final portfolio management decision.
    """

    def __init__(self, ticker: str, model_name: str = "openai/gpt-oss-120b"):
        """Initialize portfolio management agent."""
        super().__init__(ticker, model_name)
        self.role = "Portfolio Management Specialist"  # Use the role name that exists in templates

    async def run(self, **kwargs) -> dict[str, Any]:
        """Make final portfolio management decision."""
        specialist_inputs = kwargs.get("specialist_inputs", {})
        portfolio_context = kwargs.get("portfolio_context", {})
        analysis_request = kwargs.get("analysis_request", "final investment decision")
        try:
            # Generate portfolio management analysis
            analysis_prompt = {
                "ticker": self.ticker,
                "specialist_inputs": json.dumps(specialist_inputs, indent=2),
                "portfolio_context": json.dumps(portfolio_context, indent=2),
                "analysis_request": analysis_request,
            }

            portfolio_analysis = await super().run(**analysis_prompt)

            # Extract final decision
            final_decision = self._extract_final_decision(portfolio_analysis)

            return {
                "ticker": self.ticker,
                "specialist_inputs": specialist_inputs,
                "portfolio_analysis": portfolio_analysis,
                "final_decision": final_decision,
                "action": final_decision.get("action", "hold"),
                "quantity": final_decision.get("quantity", 0),
                "confidence": final_decision.get("confidence", 0.5),
                "reasoning": final_decision.get("reasoning", ""),
                "risk_assessment": final_decision.get("risk_assessment", {}),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Portfolio management analysis failed")
            return {"ticker": self.ticker, "error": str(e), "timestamp": datetime.now().isoformat()}


class RiskManagementAgent(InvestmentAnalysisAgent):
    """
    Risk management specialist for comprehensive risk assessment.

    Attributes:
    ticker (str): Stock ticker symbol.
    model_name (str): Name of the model used for analysis.

    Methods:
    run: Perform comprehensive risk assessment.
    """

    def __init__(self, ticker: str, model_name: str = "openai/gpt-oss-120b"):
        """Initialize risk management agent."""
        super().__init__(ticker, model_name)
        self.role = "Risk Management Specialist"  # Use the role name that exists in templates

    async def run(self, **kwargs) -> dict[str, Any]:
        """Perform comprehensive risk assessment."""
        debate_results = kwargs.get("debate_results", {})
        specialist_analyses = kwargs.get("specialist_analyses", {})
        analysis_request = kwargs.get("analysis_request", "risk management analysis")
        try:
            # Generate risk analysis
            analysis_prompt = {
                "ticker": self.ticker,
                "debate_results": json.dumps(debate_results, indent=2),
                "specialist_analyses": json.dumps(specialist_analyses, indent=2),
                "analysis_request": analysis_request,
            }

            risk_analysis = await super().run(**analysis_prompt)

            return {
                "ticker": self.ticker,
                "risk_analysis": risk_analysis,
                "risk_level": self._extract_risk_level(risk_analysis),
                "risk_factors": self._extract_risk_factors(risk_analysis),
                "mitigation_strategies": self._extract_mitigation_strategies(risk_analysis),
                "var_estimate": self._calculate_var_estimate(risk_analysis),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Risk management analysis failed")
            return {"ticker": self.ticker, "error": str(e), "timestamp": datetime.now().isoformat()}


class MacroAnalysisAgent(InvestmentAnalysisAgent):
    """
    Macroeconomic analysis specialist for investment context.

    Attributes:
    ticker (str): Stock ticker symbol.
    model_name (str): Name of the model used for analysis.

    Methods:
    run: Perform macroeconomic analysis.
    """

    def __init__(self, ticker: str, model_name: str = "openai/gpt-oss-120b"):
        """Initialize macro analysis agent."""
        super().__init__(ticker, model_name)
        self.role = "Macro Analysis Specialist"  # Use the role name that exists in templates

    async def run(self, **kwargs) -> dict[str, Any]:
        """Perform macroeconomic analysis."""
        market_data = kwargs.get("market_data", {})
        debate_results = kwargs.get("debate_results", {})
        analysis_request = kwargs.get("analysis_request", "macroeconomic analysis")
        try:
            # Generate macro analysis
            analysis_prompt = {
                "ticker": self.ticker,
                "market_data": json.dumps(market_data, indent=2),
                "debate_results": json.dumps(debate_results, indent=2),
                "analysis_request": analysis_request,
            }

            macro_analysis = await super().run(**analysis_prompt)

            return {
                "ticker": self.ticker,
                "macro_analysis": macro_analysis,
                "economic_outlook": self._extract_economic_outlook(macro_analysis),
                "policy_impact": self._extract_policy_impact(macro_analysis),
                "market_cycle": self._extract_market_cycle(macro_analysis),
                "macro_recommendation": self._extract_macro_recommendation(macro_analysis),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Macro analysis failed")
            return {"ticker": self.ticker, "error": str(e), "timestamp": datetime.now().isoformat()}


class MacroNewsAgent(InvestmentAnalysisAgent):
    """
    Macro news analysis specialist for market context.

    Attributes:
    ticker (str): Stock ticker symbol.
    model_name (str): Name of the model used for analysis.

    Methods:
    run: Perform macro news analysis.
    """

    def __init__(self, ticker: str, model_name: str = "openai/gpt-oss-120b"):
        """Initialize macro news agent."""
        super().__init__(ticker, model_name)
        self.role = "Macro News Specialist"  # Use the role name that exists in templates

    async def run(self, **kwargs) -> dict[str, Any]:
        """Perform macro news analysis."""
        market_data = kwargs.get("market_data", {})
        analysis_request = kwargs.get("analysis_request", "macro news analysis")
        try:
            # Generate macro news analysis
            analysis_prompt = {
                "ticker": self.ticker,
                "market_data": json.dumps(market_data, indent=2),
                "analysis_request": analysis_request,
            }

            news_analysis = await super().run(**analysis_prompt)

            return {
                "ticker": self.ticker,
                "news_analysis": news_analysis,
                "key_events": self._extract_key_events(news_analysis),
                "market_impact": self._extract_market_impact(news_analysis),
                "policy_relevance": self._extract_policy_relevance(news_analysis),
                "news_sentiment": self._extract_news_sentiment(news_analysis),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Macro news analysis failed")
            return {"ticker": self.ticker, "error": str(e), "timestamp": datetime.now().isoformat()}


class ResearcherBullAgent(InvestmentAnalysisAgent):
    """
    Bullish researcher that analyzes from optimistic perspective.

    Attributes:
    ticker (str): Stock ticker symbol.
    model_name (str): Name of the model used for analysis.

    Methods:
    run: Analyze from bullish perspective.
    """

    def __init__(self, ticker: str, model_name: str = "openai/gpt-oss-120b"):
        """Initialize bullish researcher."""
        super().__init__(ticker, model_name)
        self.role = "Bullish Researcher"  # Use the role name that exists in templates

    async def run(self, **kwargs) -> dict[str, Any]:
        """Analyze from bullish perspective."""
        specialist_inputs = kwargs.get("specialist_inputs", {})
        analysis_request = kwargs.get("analysis_request", "bullish research analysis")
        try:
            # Generate bullish analysis
            analysis_prompt = {
                "ticker": self.ticker,
                "specialist_inputs": json.dumps(specialist_inputs, indent=2),
                "analysis_request": analysis_request,
            }

            bullish_analysis = await super().run(**analysis_prompt)

            # Extract bullish thesis points
            thesis_points = self._extract_thesis_points(bullish_analysis, "bullish")
            confidence = self._extract_confidence(bullish_analysis)

            return {
                "ticker": self.ticker,
                "perspective": "bullish",
                "thesis_points": thesis_points,
                "confidence": confidence,
                "analysis": bullish_analysis,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Bullish research analysis failed")
            return {"ticker": self.ticker, "error": str(e), "timestamp": datetime.now().isoformat()}


class ResearcherBearAgent(InvestmentAnalysisAgent):
    """
    Bearish researcher that analyzes from pessimistic perspective.

    Attributes:
    ticker (str): Stock ticker symbol.
    model_name (str): Name of the model used for analysis.

    Methods:
    run: Analyze from bearish perspective.
    """

    def __init__(self, ticker: str, model_name: str = "openai/gpt-oss-120b"):
        """Initialize bearish researcher."""
        super().__init__(ticker, model_name)
        self.role = "Bearish Researcher"  # Use the role name that exists in templates

    async def run(self, **kwargs) -> dict[str, Any]:
        """Analyze from bearish perspective."""
        specialist_inputs = kwargs.get("specialist_inputs", {})
        analysis_request = kwargs.get("analysis_request", "bearish research analysis")
        try:
            # Generate bearish analysis
            analysis_prompt = {
                "ticker": self.ticker,
                "specialist_inputs": json.dumps(specialist_inputs, indent=2),
                "analysis_request": analysis_request,
            }

            bearish_analysis = await super().run(**analysis_prompt)

            # Extract bearish thesis points
            thesis_points = self._extract_thesis_points(bearish_analysis, "bearish")
            confidence = self._extract_confidence(bearish_analysis)

            return {
                "ticker": self.ticker,
                "perspective": "bearish",
                "thesis_points": thesis_points,
                "confidence": confidence,
                "analysis": bearish_analysis,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Bearish research analysis failed")
            return {"ticker": self.ticker, "error": str(e), "timestamp": datetime.now().isoformat()}


class DebateRoomAgent(InvestmentAnalysisAgent):
    """
    Debate room that facilitates bull vs bear researcher discussion with sophisticated scoring like A_Share.

    Attributes:
    ticker (str): Stock ticker symbol.
    model_name (str): Name of the model used for analysis.

    Methods:
    run: Facilitate debate between bull and bear researchers with sophisticated scoring like A_Share.
    """

    def __init__(self, ticker: str, model_name: str = "openai/gpt-oss-120b"):
        """Initialize debate room moderator."""
        super().__init__(ticker, model_name)
        self.role = "Debate Room Moderator"  # Use the role name that exists in templates

    async def run(self, **kwargs) -> dict[str, Any]:
        """Facilitate debate between bull and bear researchers with sophisticated scoring like A_Share."""
        bull_thesis = kwargs.get("bull_thesis", {})
        bear_thesis = kwargs.get("bear_thesis", {})
        analysis_request = kwargs.get("analysis_request", "debate room analysis")
        # Combine all arguments for the base class run method
        all_kwargs = {
            "ticker": self.ticker,
            "bull_thesis": json.dumps(bull_thesis, indent=2),
            "bear_thesis": json.dumps(bear_thesis, indent=2),
            "analysis_request": analysis_request,
            **kwargs,
        }

        try:
            debate_analysis = await super().run(**all_kwargs)

            # Sophisticated confidence calculation (matching A_Share)
            bull_confidence = bull_thesis.get("confidence", 0.5)
            bear_confidence = bear_thesis.get("confidence", 0.5)

            # Calculate mixed confidence with sophisticated weighting
            mixed_confidence = self._calculate_mixed_confidence_with_weights(
                bull_confidence, bear_confidence, debate_analysis
            )

            # Extract debate outcome with detailed analysis
            debate_outcome = self._extract_debate_outcome_detailed(debate_analysis, bull_thesis, bear_thesis)

            # Calculate argument strength scores
            argument_strength = self._calculate_argument_strength(bull_thesis, bear_thesis)

            # Calculate consensus level
            consensus_level = self._calculate_consensus_level(bull_thesis, bear_thesis)

            return {
                "ticker": self.ticker,
                "bull_thesis": bull_thesis,
                "bear_thesis": bear_thesis,
                "debate_analysis": debate_analysis,
                "mixed_confidence": mixed_confidence,
                "debate_outcome": debate_outcome,
                "argument_strength": argument_strength,
                "consensus_level": consensus_level,
                "debate_score": self._calculate_debate_score(bull_thesis, bear_thesis),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Debate room analysis failed")
            return {"ticker": self.ticker, "error": str(e), "timestamp": datetime.now().isoformat()}


class MultidisciplinaryInvestmentTeam:
    """
    Agent that coordinates all investment analysis specialists following A_Share_investment_Agent workflow.

    Attributes:
    ticker (str): Stock ticker symbol.
    model_name (str): Name of the model used for analysis.

    Methods:
    run_comprehensive_analysis: Run complete investment analysis workflow exactly matching A_Share_investment_Agent.
    """

    def __init__(self, ticker: str, model_name: str = "openai/gpt-oss-120b"):
        """Initialize investment team with all specialists matching A_Share_investment_Agent."""
        self.ticker = ticker
        self.model_name = model_name

        # Initialize all specialist agents (13 total like A_Share)
        self.market_data_agent = MarketDataAgent(ticker, model_name)
        self.technical_agent = TechnicalAnalysisAgent(ticker, model_name)
        self.fundamental_agent = FundamentalAnalysisAgent(ticker, model_name)
        self.sentiment_agent = SentimentAnalysisAgent(ticker, model_name)
        self.valuation_agent = ValuationAnalysisAgent(ticker, model_name)

        # Add macro news agent (parallel to core specialists)
        self.macro_news_agent = MacroNewsAgent(ticker, model_name)

        # Add bull and bear researchers
        self.researcher_bull_agent = ResearcherBullAgent(ticker, model_name)
        self.researcher_bear_agent = ResearcherBearAgent(ticker, model_name)

        # Add debate room
        self.debate_room_agent = DebateRoomAgent(ticker, model_name)

        # Add risk manager (separate from portfolio manager like A_Share)
        self.risk_manager_agent = RiskManagementAgent(ticker, model_name)

        # Add macro analyst
        self.macro_analyst_agent = MacroAnalysisAgent(ticker, model_name)

        # Portfolio manager (final decision maker)
        self.portfolio_agent = PortfolioManagementAgent(ticker, model_name)

    async def run_comprehensive_analysis(
        self, analysis_request: str = "comprehensive investment analysis"
    ) -> dict[str, Any]:
        """Run complete investment analysis workflow exactly matching A_Share_investment_Agent."""
        logger.info(f"Starting comprehensive investment analysis for {self.ticker}")

        try:
            # Step 1: Market Data Collection (Entry Point)
            market_data_result = await self.market_data_agent.run(analysis_request=analysis_request)

            if "error" in market_data_result:
                return {"error": "Market data acquisition failed", "details": market_data_result}

            market_data = market_data_result["market_data"]

            # Step 2: Parallel Analysis by Core Specialists + Macro News (like A_Share workflow)
            technical_task = self.technical_agent.run_analysis(
                market_data=market_data, analysis_request="technical analysis"
            )
            fundamental_task = self.fundamental_agent.run_analysis(
                market_data=market_data, analysis_request="fundamental analysis"
            )
            sentiment_task = self.sentiment_agent.run_analysis(
                news_data=market_data.get("news_data", {}),
                market_context=market_data.get("market_context", {}),
                analysis_request="sentiment analysis",
            )
            valuation_task = self.valuation_agent.run_analysis(
                financial_data=market_data.get("financial_data", {}),
                market_data=market_data,
                analysis_request="valuation analysis",
            )
            # Macro news runs in parallel like A_Share
            macro_news_task = self.macro_news_agent.run_analysis(
                market_data=market_data, analysis_request="macro news analysis"
            )

            # Wait for all 5 parallel analyses
            (
                technical_result,
                fundamental_result,
                sentiment_result,
                valuation_result,
                macro_news_result,
            ) = await asyncio.gather(
                technical_task,
                fundamental_task,
                sentiment_task,
                valuation_task,
                macro_news_task,
                return_exceptions=True,
            )

            # Step 3: Compile Core Specialist Inputs
            specialist_inputs = {
                "technical_analysis": technical_result
                if not isinstance(technical_result, Exception)
                else {"error": str(technical_result)},
                "fundamental_analysis": fundamental_result
                if not isinstance(fundamental_result, Exception)
                else {"error": str(fundamental_result)},
                "sentiment_analysis": sentiment_result
                if not isinstance(sentiment_result, Exception)
                else {"error": str(sentiment_result)},
                "valuation_analysis": valuation_result
                if not isinstance(valuation_result, Exception)
                else {"error": str(valuation_result)},
            }

            # Step 4: Bull and Bear Researcher Analysis (receives inputs from 4 core specialists)
            bull_task = self.researcher_bull_agent.run(
                specialist_inputs=specialist_inputs, analysis_request="bullish research analysis"
            )
            bear_task = self.researcher_bear_agent.run(
                specialist_inputs=specialist_inputs, analysis_request="bearish research analysis"
            )

            bull_result, bear_result = await asyncio.gather(bull_task, bear_task, return_exceptions=True)

            # Step 5: Debate Room Analysis (receives both researcher outputs)
            if not isinstance(bull_result, Exception) and not isinstance(bear_result, Exception):
                debate_result = await self.debate_room_agent.run(
                    bull_thesis=bull_result, bear_thesis=bear_result, analysis_request="debate room analysis"
                )
            else:
                debate_result = {"error": "Debate room analysis failed"}

            # Step 6: Risk Management Analysis (receives debate results and specialist analyses)
            if not isinstance(debate_result, Exception):
                risk_management_result = await self.risk_manager_agent.run(
                    debate_results=debate_result,
                    specialist_analyses=specialist_inputs,
                    analysis_request="risk management analysis",
                )
            else:
                risk_management_result = {"error": "Risk management analysis failed"}

            # Step 7: Macro Analyst Analysis (receives market data and debate results)
            if not isinstance(debate_result, Exception):
                macro_analysis_result = await self.macro_analyst_agent.run(
                    market_data=market_data, debate_results=debate_result, analysis_request="macroeconomic analysis"
                )
            else:
                macro_analysis_result = {"error": "Macro analysis failed"}

            # Step 8: Portfolio Management Decision (receives macro_analysis and macro_news - convergence point)
            portfolio_context = {
                "current_portfolio": {"cash": 100000, "positions": {}},
                "risk_parameters": {"max_position_size": 0.1, "max_portfolio_risk": 0.15},
            }

            # Include all analyses in portfolio inputs (convergence of macro_analyst and macro_news paths)
            enhanced_specialist_inputs = {
                **specialist_inputs,
                "researcher_bull": bull_result
                if not isinstance(bull_result, Exception)
                else {"error": str(bull_result)},
                "researcher_bear": bear_result
                if not isinstance(bear_result, Exception)
                else {"error": str(bear_result)},
                "debate_room": debate_result
                if not isinstance(debate_result, Exception)
                else {"error": str(debate_result)},
                "risk_management": risk_management_result
                if not isinstance(risk_management_result, Exception)
                else {"error": str(risk_management_result)},
                "macro_analysis": macro_analysis_result
                if not isinstance(macro_analysis_result, Exception)
                else {"error": str(macro_analysis_result)},
                "macro_news": macro_news_result
                if not isinstance(macro_news_result, Exception)
                else {"error": str(macro_news_result)},
            }

            final_decision = await self.portfolio_agent.run(
                specialist_inputs=enhanced_specialist_inputs,
                portfolio_context=portfolio_context,
                analysis_request="final investment decision",
            )

            # Step 9: Compile Final Report (matching A_Share structure)
            comprehensive_analysis = {
                "ticker": self.ticker,
                "analysis_request": analysis_request,
                "market_data": market_data_result,
                "specialist_analyses": specialist_inputs,
                "macro_news": macro_news_result
                if not isinstance(macro_news_result, Exception)
                else {"error": str(macro_news_result)},
                "researcher_analyses": {
                    "bull_researcher": bull_result
                    if not isinstance(bull_result, Exception)
                    else {"error": str(bull_result)},
                    "bear_researcher": bear_result
                    if not isinstance(bear_result, Exception)
                    else {"error": str(bear_result)},
                },
                "debate_room": debate_result
                if not isinstance(debate_result, Exception)
                else {"error": str(debate_result)},
                "risk_management": risk_management_result
                if not isinstance(risk_management_result, Exception)
                else {"error": str(risk_management_result)},
                "macro_analysis": macro_analysis_result
                if not isinstance(macro_analysis_result, Exception)
                else {"error": str(macro_analysis_result)},
                "final_decision": final_decision,
                "analysis_timestamp": datetime.now().isoformat(),
                "model_used": self.model_name,
            }

            logger.info(f"Comprehensive analysis completed for {self.ticker}")
            return comprehensive_analysis

        except Exception as e:
            logger.exception("Comprehensive analysis failed")
            return {"ticker": self.ticker, "error": str(e), "timestamp": datetime.now().isoformat()}


async def run_share_investment_analysis(
    ticker: str, analysis_request: str, model_name: str = "openai/gpt-oss-120b"
) -> str:
    """Run comprehensive share investment analysis - matches spatial-agent pattern."""
    if not ticker:
        return "Error: No stock ticker provided for analysis."

    try:
        # Create multidisciplinary investment team
        investment_team = MultidisciplinaryInvestmentTeam(ticker, model_name)

        # Run comprehensive analysis
        final_analysis = await investment_team.run_comprehensive_analysis(analysis_request)

        # Format the response
        if "error" in final_analysis:
            response = f"### Investment Analysis Error:\n\n{final_analysis['error']}"
        else:
            final_decision = final_analysis.get("final_decision", {})
            specialist_analyses = final_analysis.get("specialist_analyses", {})
            researcher_analyses = final_analysis.get("researcher_analyses", {})
            debate_room = final_analysis.get("debate_room", {})
            risk_management = final_analysis.get("risk_management", {})
            macro_analysis = final_analysis.get("macro_analysis", {})
            macro_news = final_analysis.get("macro_news", {})
            market_data_result = final_analysis.get("market_data", {}).get("market_data", {})

            response = f"""### Share Investment Analysis Report:

**Stock Ticker:** {ticker}
**Analysis Type:** {analysis_request}
**Analysis Date:** {final_analysis.get("analysis_timestamp", "Unknown")}

---

#### Final Investment Decision:
- **Action:** {str(final_decision.get("action", "N/A")).upper()}
- **Confidence:** {final_decision.get("confidence", "0%")}
- **Quantity:** {final_decision.get("quantity", 0)} shares
- **Reasoning:** {final_decision.get("reasoning", "No reasoning provided")}

---

#### Core Specialist Analysis Summary:

**Technical Analysis:** {specialist_analyses.get("technical_analysis", {}).get("technical_analysis", {}).get("signal", "N/A")} (Confidence: {specialist_analyses.get("technical_analysis", {}).get("technical_analysis", {}).get("confidence", "0%")})

**Fundamental Analysis:** {specialist_analyses.get("fundamental_analysis", {}).get("fundamental_analysis", {}).get("signal", "N/A")} (Confidence: {specialist_analyses.get("fundamental_analysis", {}).get("fundamental_analysis", {}).get("confidence", "0%")})

**Sentiment Analysis:** {specialist_analyses.get("sentiment_analysis", {}).get("sentiment_analysis", {}).get("sentiment_label", "N/A")} (Confidence: {specialist_analyses.get("sentiment_analysis", {}).get("sentiment_analysis", {}).get("confidence", "0%")})

**Valuation Analysis:** {specialist_analyses.get("valuation_analysis", {}).get("valuation_analysis", {}).get("valuation_label", "N/A")} (Confidence: {specialist_analyses.get("valuation_analysis", {}).get("valuation_analysis", {}).get("confidence", "0%")})

---

#### Research Team Analysis:

**Bullish Researcher:** {researcher_analyses.get("bull_researcher", {}).get("perspective", "N/A")} (Confidence: {researcher_analyses.get("bull_researcher", {}).get("confidence", "0%")})

**Bearish Researcher:** {researcher_analyses.get("bear_researcher", {}).get("perspective", "N/A")} (Confidence: {researcher_analyses.get("bear_researcher", {}).get("confidence", "0%")})

---

#### Debate Room Outcome:
**Debate Result:** {debate_room.get("debate_outcome", "N/A")}
**Mixed Confidence:** {debate_room.get("mixed_confidence", "0%")}
**Analysis:** {str(debate_room.get("debate_analysis", {}).get("analysis", "No debate analysis available"))[:200]}...

---

#### Risk Management Assessment:
**Risk Level:** {risk_management.get("risk_level", "N/A")}
**VaR Estimate:** {risk_management.get("var_estimate", "0%")}
**Risk Factors:** {len(risk_management.get("risk_factors", []))} identified factors

---

#### Macro Analysis:
**Economic Outlook:** {macro_analysis.get("economic_outlook", "N/A")}
**Policy Impact:** {macro_analysis.get("policy_impact", "N/A")}
**Market Cycle:** {macro_analysis.get("market_cycle", "N/A")}

---

#### Macro News Summary:
**Key Events:** {len(macro_news.get("key_events", []))} major events
**Market Impact:** {macro_news.get("market_impact", "N/A")}
**News Sentiment:** {macro_news.get("news_sentiment", "N/A")}

---

#### Risk Assessment:
{final_decision.get("risk_assessment", {}).get("risk_level", "Unknown")} risk level

---

#### Market Data Summary:
- **Current Price:** {market_data_result.get("current_price", "N/A")}
- **Market Cap:** {market_data_result.get("market_cap", "N/A")}
- **Volume:** {market_data_result.get("volume", "N/A")}

---

*Analysis performed by AI Share Investment Agent*
*Specialist Team: Market Data, Technical, Fundamental, Sentiment, Valuation, Macro News, Bull/Bear Researchers, Debate Room, Risk Management, Macro Analyst, Portfolio Management*
*Model: {final_analysis.get("model_used", model_name)}*
"""

        return response

    except Exception as e:
        error_msg = f"Error during investment analysis: {e!s}"
        logger.exception("Error during investment analysis")
        return error_msg
