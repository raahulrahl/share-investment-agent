"""Share Investment Analysis Agents using LangChain."""

import asyncio
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from .tools.data_acquisition import MarketDataAcquisition
from .tools.financial_analysis import FinancialAnalyzer
from .utils.logging_config import setup_logger

logger = setup_logger(__name__)


class InvestmentAnalysisAgent:
    """Base class for investment analysis agents."""

    def __init__(self, ticker: str, role: str, model_name: str = "gpt-4o"):
        """Initialize investment analysis agent with ticker, role, and model."""
        self.ticker = ticker
        self.role = role
        self.model_name = model_name

        # Get API key from environment (supports OpenRouter)
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        if openrouter_api_key:
            # Use OpenRouter API key with OpenAI client
            self.model = ChatOpenAI(
                model_name=model_name,
                openai_api_key=SecretStr(openrouter_api_key),
                openai_api_base="https://api.openai.com/v1",
                temperature=0,
            )
        else:
            raise ValueError("No API key provided. Set OPENROUTER_API_KEY environment variable.")
        self.prompt_template = self._create_prompt_template()
        self.chain = self.prompt_template | self.model | StrOutputParser()

    def _create_prompt_template(self) -> PromptTemplate:
        """Create role-specific prompt template."""
        templates = {
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
            "Technical Analysis Specialist": """
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
            "Fundamental Analysis Specialist": """
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

        return PromptTemplate.from_template(templates[self.role])

    async def run(self, **kwargs) -> Any:
        """Run the agent analysis."""
        logger.info(f"{self.role} is running analysis for {self.ticker}...")

        try:
            # Create prompt template
            prompt_template = self._create_prompt_template()

            # Format prompt with provided kwargs
            prompt = prompt_template.format(**kwargs)

            # Get LLM response
            response = await self.model.ainvoke(prompt)

            return response
        except Exception as e:
            logger.exception("Error occurred in {self.role}")
            return f"Error: {e!s}"

    def _extract_confidence(self, analysis: str) -> float:
        """Extract confidence score from analysis."""
        # Simple confidence extraction - can be enhanced
        if "high confidence" in analysis.lower():
            return 0.8
        elif "medium confidence" in analysis.lower():
            return 0.6
        else:
            return 0.5

    def _extract_thesis_points(self, analysis: str, perspective: str) -> list[str]:
        """Extract thesis points from analysis."""
        # Simple thesis point extraction - can be enhanced with NLP
        lines = analysis.split("\n")
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

    def _extract_signal(self, analysis: str) -> str:
        """Extract trading signal from analysis."""
        analysis_lower = analysis.lower()
        if "strong buy" in analysis_lower or "bullish" in analysis_lower:
            return "bullish"
        elif "strong sell" in analysis_lower or "bearish" in analysis_lower:
            return "bearish"
        else:
            return "neutral"


class MarketDataAgent(InvestmentAnalysisAgent):
    """Specialized agent for market data acquisition and preprocessing."""

    def __init__(self, ticker: str, model_name: str = "gpt-4o"):
        """Initialize market data acquisition agent."""
        super().__init__(ticker, "Market Data Analyst", model_name)
        self.data_acquisition = MarketDataAcquisition()

    async def run(self, **kwargs) -> dict[str, Any]:
        """Collect and preprocess market data."""
        analysis_request = kwargs.get("analysis_request", "comprehensive analysis")

        try:
            # Acquire market data
            market_data = await self.data_acquisition.get_comprehensive_data(self.ticker)

            # Generate analysis summary
            analysis_prompt = {"ticker": self.ticker, "analysis_request": analysis_request}

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
    """Specialized agent for technical analysis with sophisticated mathematical calculations."""

    def __init__(self, ticker: str, model_name: str = "gpt-4o"):
        """Initialize technical analysis agent."""
        super().__init__(ticker, "Technical Analysis Specialist", model_name)
        self.financial_analyzer = FinancialAnalyzer()

    async def run(self, **kwargs) -> dict[str, Any]:
        """Perform sophisticated technical analysis with mathematical calculations like A_Share."""
        market_data = kwargs.get("market_data", {})
        analysis_request = kwargs.get("analysis_request", "technical analysis")

        try:
            # Get price history for calculations
            price_history = market_data.get("historical_data", [])

            if not price_history or len(price_history) < 50:
                return {
                    "ticker": self.ticker,
                    "error": "Insufficient price history data",
                    "timestamp": datetime.now().isoformat(),
                }

            # Convert to DataFrame for calculations
            df = pd.DataFrame(price_history)
            if "close" not in df.columns:
                return {"error": "Price data missing 'close' column"}

            # closes is available for calculations but not used in this method

            # Sophisticated mathematical calculations (matching A_Share)
            trend_signals = self._calculate_trend_signals(df)
            momentum_signals = self._calculate_momentum_signals(df)
            mean_reversion_signals = self._calculate_mean_reversion_signals(df)
            volatility_signals = self._calculate_volatility_signals(df)
            stat_arb_signals = self._calculate_stat_arb_signals(df)

            # Create comprehensive analysis report
            analysis_report = {
                "ticker": self.ticker,
                "trend": {
                    "signal": trend_signals["signal"],
                    "confidence": f"{round(trend_signals['confidence'] * 100)}%",
                    "metrics": trend_signals["metrics"],
                },
                "momentum": {
                    "signal": momentum_signals["signal"],
                    "confidence": f"{round(momentum_signals['confidence'] * 100)}%",
                    "metrics": momentum_signals["metrics"],
                },
                "mean_reversion": {
                    "signal": mean_reversion_signals["signal"],
                    "confidence": f"{round(mean_reversion_signals['confidence'] * 100)}%",
                    "metrics": mean_reversion_signals["metrics"],
                },
                "volatility": {
                    "signal": volatility_signals["signal"],
                    "confidence": f"{round(volatility_signals['confidence'] * 100)}%",
                    "metrics": volatility_signals["metrics"],
                },
                "statistical_arbitrage": {
                    "signal": stat_arb_signals["signal"],
                    "confidence": f"{round(stat_arb_signals['confidence'] * 100)}%",
                    "metrics": stat_arb_signals["metrics"],
                },
            }

            # Generate LLM analysis for interpretation
            analysis_prompt = {
                "ticker": self.ticker,
                "market_data": json.dumps(analysis_report, indent=2),
                "analysis_request": analysis_request,
            }

            technical_analysis = await super().run(**analysis_prompt)

            return {
                "ticker": self.ticker,
                "technical_indicators": analysis_report,
                "technical_analysis": technical_analysis,
                "signal": self._extract_trading_signal(technical_analysis),
                "confidence": self._extract_confidence(technical_analysis),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Technical analysis failed")
            return {"ticker": self.ticker, "error": str(e), "timestamp": datetime.now().isoformat()}

    def _calculate_trend_signals(self, prices_df):
        """Advanced trend following strategy using multiple timeframes and indicators (matching A_Share)."""
        # Calculate EMAs for multiple timeframes
        ema_8 = self._calculate_ema(prices_df, 8)
        ema_21 = self._calculate_ema(prices_df, 21)
        ema_55 = self._calculate_ema(prices_df, 55)

        # Calculate ADX for trend strength
        adx = self._calculate_adx(prices_df, 14)

        # Trend strength scoring
        trend_score = 0
        if ema_8.iloc[-1] > ema_21.iloc[-1] > ema_55.iloc[-1]:
            trend_score += 0.3
        if adx.iloc[-1] > 25:  # Strong trend
            trend_score += 0.4
        if adx.iloc[-1] > 40:  # Very strong trend
            trend_score += 0.3

        # Generate signal
        if trend_score >= 0.7:
            signal = "bullish"
            confidence = min(trend_score, 1.0)
        elif trend_score <= 0.3:
            signal = "bearish"
            confidence = min(1 - trend_score, 1.0)
        else:
            signal = "neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "ema_8": float(ema_8.iloc[-1]),
                "ema_21": float(ema_21.iloc[-1]),
                "ema_55": float(ema_55.iloc[-1]),
                "adx": float(adx.iloc[-1]),
                "trend_score": float(trend_score),
            },
        }

    def _calculate_momentum_signals(self, prices_df):
        """Multi-factor momentum strategy with conservative settings (matching A_Share)."""
        # Price momentum with adjusted min_periods
        returns = prices_df["close"].pct_change()
        mom_1m = returns.rolling(21, min_periods=5).sum()
        mom_3m = returns.rolling(63, min_periods=42).sum()
        mom_6m = returns.rolling(126, min_periods=63).sum()

        # Volume momentum
        volume_ma = prices_df["volume"].rolling(21, min_periods=10).mean()
        volume_momentum = prices_df["volume"] / volume_ma

        # Handle NaN values
        mom_1m = mom_1m.fillna(0)
        mom_3m = mom_3m.fillna(mom_1m)
        mom_6m = mom_6m.fillna(mom_3m)

        # Calculate momentum score with more weight on longer timeframes
        momentum_score = 0.2 * mom_1m + 0.3 * mom_3m + 0.5 * mom_6m

        # Volume confirmation
        volume_confirmation = volume_momentum.iloc[-1] > 1.2

        # Generate signal
        if momentum_score.iloc[-1] > 0.05 and volume_confirmation:
            signal = "bullish"
            confidence = min(abs(momentum_score.iloc[-1]) * 10, 1.0)
        elif momentum_score.iloc[-1] < -0.05 and not volume_confirmation:
            signal = "bearish"
            confidence = min(abs(momentum_score.iloc[-1]) * 10, 1.0)
        else:
            signal = "neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "momentum_1m": float(mom_1m.iloc[-1]),
                "momentum_3m": float(mom_3m.iloc[-1]),
                "momentum_6m": float(mom_6m.iloc[-1]),
                "volume_momentum": float(volume_momentum.iloc[-1]),
                "momentum_score": float(momentum_score.iloc[-1]),
            },
        }

    def _calculate_mean_reversion_signals(self, prices_df):
        """Mean reversion strategy using Bollinger Bands and RSI (matching A_Share)."""
        # Calculate RSI
        rsi_14 = self._calculate_rsi(prices_df, 14)
        rsi_28 = self._calculate_rsi(prices_df, 28)

        # Calculate Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(prices_df)

        # Calculate Z-score
        bb_middle = (bb_upper + bb_lower) / 2
        bb_std = (bb_upper - bb_lower) / 4
        z_score = (prices_df["close"] - bb_middle) / bb_std

        # Price vs Bollinger Bands position
        price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

        # Combine signals
        if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
            signal = "bullish"
            confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
        elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
            signal = "bearish"
            confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
        else:
            signal = "neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "z_score": float(z_score.iloc[-1]),
                "price_vs_bb": float(price_vs_bb),
                "rsi_14": float(rsi_14.iloc[-1]),
                "rsi_28": float(rsi_28.iloc[-1]),
            },
        }

    def _calculate_volatility_signals(self, prices_df):
        """Volatility analysis with regime detection (matching A_Share)."""
        returns = prices_df["close"].pct_change()
        daily_vol = returns.std()
        volatility = daily_vol * (252**0.5)

        # Calculate volatility regime
        rolling_std = returns.rolling(window=120).std() * (252**0.5)
        volatility_mean = rolling_std.mean()
        volatility_std = rolling_std.std()
        volatility_percentile = (volatility - volatility_mean) / volatility_std

        # Calculate ATR
        high_low = prices_df["high"] - prices_df["low"]
        high_close = abs(prices_df["high"] - prices_df["close"].shift())
        low_close = abs(prices_df["low"] - prices_df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()
        atr_ratio = atr / prices_df["close"]

        # Handle NaN values
        if pd.isna(volatility_percentile.iloc[-1]):
            volatility_percentile.iloc[-1] = 0.0

        # Generate signal based on volatility regime
        current_vol_regime = volatility
        vol_z = volatility_percentile.iloc[-1]

        if current_vol_regime < volatility_mean * 0.8 and vol_z < -1:
            signal = "bullish"  # Low vol regime, potential for expansion
            confidence = min(abs(vol_z) / 3, 1.0)
        elif current_vol_regime > volatility_mean * 1.2 and vol_z > 1:
            signal = "bearish"  # High vol regime, potential for contraction
            confidence = min(abs(vol_z) / 3, 1.0)
        else:
            signal = "neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "historical_volatility": float(volatility),
                "volatility_regime": float(current_vol_regime),
                "volatility_z_score": float(vol_z),
                "atr_ratio": float(atr_ratio.iloc[-1]),
            },
        }

    def _calculate_stat_arb_signals(self, prices_df):
        """Optimized statistical arbitrage signals with shorter lookback periods (matching A_Share)."""
        returns = prices_df["close"].pct_change()

        # Calculate price distribution statistics
        skew = returns.rolling(42, min_periods=21).skew()
        kurt = returns.rolling(42, min_periods=21).kurt()

        # Calculate Hurst exponent
        hurst = self._calculate_hurst_exponent(prices_df["close"], max_lag=10)

        # Handle NaN values
        if pd.isna(skew.iloc[-1]):
            skew.iloc[-1] = 0.0
        if pd.isna(kurt.iloc[-1]):
            kurt.iloc[-1] = 3.0

        # Generate statistical arbitrage signal
        stat_score = 0
        if skew.iloc[-1] > 0.5:  # Positive skew
            stat_score += 0.3
        if kurt.iloc[-1] > 4:  # Fat tails
            stat_score += 0.3
        if hurst < 0.45:  # Mean reverting
            stat_score += 0.4

        if stat_score >= 0.7:
            signal = "bullish"
            confidence = min(stat_score, 1.0)
        elif stat_score <= 0.3:
            signal = "bearish"
            confidence = min(1 - stat_score, 1.0)
        else:
            signal = "neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "skew": float(skew.iloc[-1]),
                "kurtosis": float(kurt.iloc[-1]),
                "hurst_exponent": float(hurst),
                "stat_score": float(stat_score),
            },
        }

    def _calculate_ema(self, prices_df, period):
        """Calculate Exponential Moving Average."""
        return prices_df["close"].ewm(span=period).mean()

    def _calculate_adx(self, prices_df, period):
        """Calculate Average Directional Index."""
        # Calculate True Range
        high_low = prices_df["high"] - prices_df["low"]
        high_close = abs(prices_df["high"] - prices_df["close"].shift())
        low_close = abs(prices_df["low"] - prices_df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate +DM and -DM
        up_move = prices_df["high"] - prices_df["high"].shift()
        down_move = prices_df["low"].shift() - prices_df["low"]

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # Calculate ADX
        tr_smooth = tr.rolling(window=period).mean()
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()

        plus_di = 100 * (plus_dm_smooth / tr_smooth).ewm(span=period).mean()
        minus_di = 100 * (minus_dm_smooth / tr_smooth).ewm(span=period).mean()

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period).mean()

        return adx

    def _calculate_rsi(self, prices_df, period):
        """Calculate Relative Strength Index."""
        delta = prices_df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(self, prices_df, period=20, std_dev=2):
        """Calculate Bollinger Bands."""
        sma = prices_df["close"].rolling(window=period).mean()
        std = prices_df["close"].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

    def _calculate_hurst_exponent(self, ts, max_lag=20):
        """Calculate Hurst exponent for mean reversion detection."""
        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

        # Linear regression in log-log space
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] * 2.0

        return hurst

    def _extract_trading_signal(self, analysis: str) -> str:
        """Extract trading signal from analysis."""
        analysis_lower = analysis.lower()
        if "strong buy" in analysis_lower or "bullish" in analysis_lower:
            return "bullish"
        elif "strong sell" in analysis_lower or "bearish" in analysis_lower:
            return "bearish"
        else:
            return "neutral"


class FundamentalAnalysisAgent(InvestmentAnalysisAgent):
    """Specialized agent for fundamental analysis with sophisticated financial calculations."""

    def __init__(self, ticker: str, model_name: str = "gpt-4o"):
        """Initialize fundamental analysis agent."""
        super().__init__(ticker, "Fundamental Analysis Specialist", model_name)
        self.financial_analyzer = FinancialAnalyzer()

    async def run(self, **kwargs) -> dict[str, Any]:
        """Perform sophisticated fundamental analysis with mathematical calculations like A_Share."""
        market_data = kwargs.get("market_data", {})
        analysis_request = kwargs.get("analysis_request", "fundamental analysis")

        try:
            # Get financial data
            financial_data = market_data.get("financial_data", {})

            if not financial_data:
                return {
                    "ticker": self.ticker,
                    "error": "No financial data available",
                    "timestamp": datetime.now().isoformat(),
                }

            # Sophisticated fundamental analysis calculations (matching A_Share)
            profitability_analysis = self._analyze_profitability(financial_data)
            growth_analysis = self._analyze_growth(financial_data)
            financial_health_analysis = self._analyze_financial_health(financial_data)
            valuation_metrics = self._analyze_valuation_metrics(financial_data)

            # Calculate comprehensive fundamental score
            fundamental_score = self._calculate_fundamental_score(
                profitability_analysis, growth_analysis, financial_health_analysis
            )

            # Create comprehensive fundamental analysis report
            analysis_report = {
                "ticker": self.ticker,
                "profitability": {
                    "signal": profitability_analysis["signal"],
                    "confidence": f"{round(profitability_analysis['confidence'] * 100)}%",
                    "metrics": profitability_analysis["metrics"],
                },
                "growth": {
                    "signal": growth_analysis["signal"],
                    "confidence": f"{round(growth_analysis['confidence'] * 100)}%",
                    "metrics": growth_analysis["metrics"],
                },
                "financial_health": {
                    "signal": financial_health_analysis["signal"],
                    "confidence": f"{round(financial_health_analysis['confidence'] * 100)}%",
                    "metrics": financial_health_analysis["metrics"],
                },
                "valuation": {
                    "signal": valuation_metrics["signal"],
                    "confidence": f"{round(valuation_metrics['confidence'] * 100)}%",
                    "metrics": valuation_metrics["metrics"],
                },
                "fundamental_score": fundamental_score,
            }

            # Generate LLM analysis for interpretation
            analysis_prompt = {
                "ticker": self.ticker,
                "financial_data": json.dumps(analysis_report, indent=2),
                "analysis_request": analysis_request,
            }

            fundamental_analysis = await super().run(**analysis_prompt)

            return {
                "ticker": self.ticker,
                "financial_metrics": analysis_report,
                "fundamental_analysis": fundamental_analysis,
                "signal": self._extract_fundamental_signal(fundamental_analysis),
                "confidence": self._extract_confidence(fundamental_analysis),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Fundamental analysis failed")
            return {"ticker": self.ticker, "error": str(e), "timestamp": datetime.now().isoformat()}

    def _analyze_profitability(self, financial_data):
        """Analyze profitability metrics with sophisticated calculations (matching A_Share)."""
        metrics = financial_data.get("profitability_metrics", {})

        # Get key profitability ratios
        return_on_equity = metrics.get("return_on_equity", 0)
        net_margin = metrics.get("net_margin", 0)
        operating_margin = metrics.get("operating_margin", 0)
        gross_margin = metrics.get("gross_margin", 0)
        roic = metrics.get("return_on_invested_capital", 0)
        roa = metrics.get("return_on_assets", 0)

        # Sophisticated profitability scoring
        thresholds = [
            (return_on_equity, 0.15),  # Strong ROE above 15%
            (net_margin, 0.20),  # Healthy profit margins
            (operating_margin, 0.15),  # Strong operating efficiency
            (gross_margin, 0.30),  # Good gross margins
            (roic, 0.12),  # Good capital efficiency
            (roa, 0.08),  # Good asset utilization
        ]

        profitability_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds) / len(
            thresholds
        )

        # Generate signal based on profitability score
        if profitability_score >= 0.6:
            signal = "bullish"
            confidence = min(profitability_score + 0.2, 1.0)
        elif profitability_score <= 0.2:
            signal = "bearish"
            confidence = min((1 - profitability_score) + 0.2, 1.0)
        else:
            signal = "neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "return_on_equity": float(return_on_equity),
                "net_margin": float(net_margin),
                "operating_margin": float(operating_margin),
                "gross_margin": float(gross_margin),
                "roic": float(roic),
                "roa": float(roa),
                "profitability_score": float(profitability_score),
            },
        }

    def _analyze_growth(self, financial_data):
        """Analyze growth metrics with sophisticated calculations (matching A_Share)."""
        metrics = financial_data.get("growth_metrics", {})

        # Get key growth metrics
        revenue_growth = metrics.get("revenue_growth_rate", 0)
        earnings_growth = metrics.get("earnings_growth_rate", 0)
        eps_growth = metrics.get("eps_growth_rate", 0)
        book_value_growth = metrics.get("book_value_growth_rate", 0)

        # Growth consistency analysis
        growth_consistency = metrics.get("growth_consistency_score", 0.5)

        # Sophisticated growth scoring
        growth_metrics = [
            (revenue_growth, 0.10),  # 10% revenue growth
            (earnings_growth, 0.15),  # 15% earnings growth
            (eps_growth, 0.12),  # 12% EPS growth
            (book_value_growth, 0.08),  # 8% book value growth
        ]

        growth_score = sum(metric is not None and metric > threshold for metric, threshold in growth_metrics) / len(
            growth_metrics
        )

        # Adjust for consistency
        growth_score = growth_score * growth_consistency

        # Generate signal based on growth score
        if growth_score >= 0.6:
            signal = "bullish"
            confidence = min(growth_score + 0.15, 1.0)
        elif growth_score <= 0.2:
            signal = "bearish"
            confidence = min((1 - growth_score) + 0.15, 1.0)
        else:
            signal = "neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "revenue_growth": float(revenue_growth),
                "earnings_growth": float(earnings_growth),
                "eps_growth": float(eps_growth),
                "book_value_growth": float(book_value_growth),
                "growth_consistency": float(growth_consistency),
                "growth_score": float(growth_score),
            },
        }

    def _analyze_financial_health(self, financial_data):
        """Analyze financial health with sophisticated calculations (matching A_Share)."""
        metrics = financial_data.get("financial_health_metrics", {})

        # Get key financial health metrics
        debt_to_equity = metrics.get("debt_to_equity", 0)
        current_ratio = metrics.get("current_ratio", 0)
        quick_ratio = metrics.get("quick_ratio", 0)
        interest_coverage = metrics.get("interest_coverage_ratio", 0)
        debt_service_coverage = metrics.get("debt_service_coverage", 0)
        cash_ratio = metrics.get("cash_ratio", 0)

        # Sophisticated financial health scoring
        health_metrics = [
            (debt_to_equity, 0.6, "max"),  # Debt to equity should be low
            (current_ratio, 1.5, "min"),  # Current ratio should be high
            (quick_ratio, 1.0, "min"),  # Quick ratio should be adequate
            (interest_coverage, 3.0, "min"),  # Interest coverage should be strong
            (debt_service_coverage, 1.2, "min"),  # DSCR should be adequate
            (cash_ratio, 0.2, "min"),  # Cash ratio should be reasonable
        ]

        health_score = 0
        for metric, threshold, comparison in health_metrics:
            if metric is not None:
                if comparison == "max":
                    if metric <= threshold:
                        health_score += 1 / len(health_metrics)
                else:  # 'min'
                    if metric >= threshold:
                        health_score += 1 / len(health_metrics)

        # Generate signal based on health score
        if health_score >= 0.7:
            signal = "bullish"
            confidence = min(health_score, 1.0)
        elif health_score <= 0.3:
            signal = "bearish"
            confidence = min(1 - health_score, 1.0)
        else:
            signal = "neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "debt_to_equity": float(debt_to_equity),
                "current_ratio": float(current_ratio),
                "quick_ratio": float(quick_ratio),
                "interest_coverage": float(interest_coverage),
                "debt_service_coverage": float(debt_service_coverage),
                "cash_ratio": float(cash_ratio),
                "health_score": float(health_score),
            },
        }

    def _analyze_valuation_metrics(self, financial_data):
        """Analyze valuation metrics with sophisticated calculations (matching A_Share)."""
        metrics = financial_data.get("valuation_metrics", {})

        # Get key valuation metrics
        pe_ratio = metrics.get("price_to_earnings", 0)
        pb_ratio = metrics.get("price_to_book", 0)
        ps_ratio = metrics.get("price_to_sales", 0)
        ev_ebitda = metrics.get("ev_to_ebitda", 0)
        dividend_yield = metrics.get("dividend_yield", 0)

        # Industry averages for comparison (simplified)
        industry_pe_avg = 15.0
        industry_pb_avg = 2.0
        industry_ps_avg = 3.0

        # Sophisticated valuation scoring
        valuations = [
            (pe_ratio, industry_pe_avg, "inverse"),  # Lower P/E is better
            (pb_ratio, industry_pb_avg, "inverse"),  # Lower P/B is better
            (ps_ratio, industry_ps_avg, "inverse"),  # Lower P/S is better
            (dividend_yield, 0.03, "normal"),  # Higher dividend yield is better
        ]

        valuation_score = 0
        for metric, industry_avg, comparison in valuations:
            if metric is not None and industry_avg > 0:
                if comparison == "inverse":
                    if metric <= industry_avg * 0.8:  # Significantly undervalued
                        valuation_score += 0.25
                    elif metric <= industry_avg:  # Reasonably valued
                        valuation_score += 0.15
                else:  # 'normal'
                    if metric >= industry_avg * 1.2:  # Significantly overvalued
                        valuation_score -= 0.25
                    elif metric >= industry_avg:  # Overvalued
                        valuation_score -= 0.15

        # Normalize score to 0-1 range
        valuation_score = max(0, min(1, valuation_score + 0.5))

        # Generate signal based on valuation score
        if valuation_score >= 0.6:
            signal = "bullish"  # Undervalued
            confidence = min(valuation_score, 1.0)
        elif valuation_score <= 0.3:
            signal = "bearish"  # Overvalued
            confidence = min(1 - valuation_score, 1.0)
        else:
            signal = "neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "pe_ratio": float(pe_ratio),
                "pb_ratio": float(pb_ratio),
                "ps_ratio": float(ps_ratio),
                "ev_ebitda": float(ev_ebitda),
                "dividend_yield": float(dividend_yield),
                "valuation_score": float(valuation_score),
            },
        }

    def _calculate_fundamental_score(self, profitability, growth, financial_health):
        """Calculate comprehensive fundamental score."""
        weights = {"profitability": 0.3, "growth": 0.3, "financial_health": 0.4}

        score = (
            weights["profitability"] * profitability["confidence"]
            + weights["growth"] * growth["confidence"]
            + weights["financial_health"] * financial_health["confidence"]
        )

        return round(score, 3)

    def _extract_fundamental_signal(self, analysis: str) -> str:
        """Extract fundamental signal from analysis."""
        analysis_lower = analysis.lower()
        if "strong fundamentals" in analysis_lower or "buy" in analysis_lower:
            return "bullish"
        elif "weak fundamentals" in analysis_lower or "sell" in analysis_lower:
            return "bearish"
        else:
            return "neutral"


class SentimentAnalysisAgent(InvestmentAnalysisAgent):
    """Specialized agent for sentiment analysis."""

    def __init__(self, ticker: str, model_name: str = "gpt-4o"):
        """Initialize sentiment analysis agent."""
        super().__init__(ticker, "Sentiment Analysis Specialist", model_name)

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

    def _extract_sentiment_signal(self, analysis: str) -> str:
        """Extract sentiment signal from analysis."""
        analysis_lower = analysis.lower()
        if "positive sentiment" in analysis_lower or "bullish" in analysis_lower:
            return "bullish"
        elif "negative sentiment" in analysis_lower or "bearish" in analysis_lower:
            return "bearish"
        else:
            return "neutral"

    def _extract_sentiment_score(self, analysis: str) -> float:
        """Extract sentiment score from analysis."""
        # Simple sentiment score extraction - can be enhanced
        if "strongly positive" in analysis.lower():
            return 0.8
        elif "positive" in analysis.lower():
            return 0.6
        elif "negative" in analysis.lower():
            return -0.6
        elif "strongly negative" in analysis.lower():
            return -0.8
        else:
            return 0.0


class ValuationAnalysisAgent(InvestmentAnalysisAgent):
    """Specialized agent for valuation analysis."""

    def __init__(self, ticker: str, model_name: str = "gpt-4o"):
        """Initialize valuation analysis agent."""
        super().__init__(ticker, "Valuation Analysis Specialist", model_name)
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
                "financial_data": json.dumps(valuation_metrics, indent=2),
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

    def _extract_valuation_signal(self, analysis: str) -> str:
        """Extract valuation signal from analysis."""
        analysis_lower = analysis.lower()
        if "undervalued" in analysis_lower or "buy" in analysis_lower:
            return "bullish"
        elif "overvalued" in analysis_lower or "sell" in analysis_lower:
            return "bearish"
        else:
            return "neutral"

    def _extract_intrinsic_value(self, analysis: str) -> float:
        """Extract intrinsic value from analysis."""
        # Simple intrinsic value extraction - can be enhanced
        # This would typically involve parsing DCF or other valuation results
        return 0.0  # Placeholder


class PortfolioManagementAgent(InvestmentAnalysisAgent):
    """Specialized agent for portfolio management and final decision making."""

    def __init__(self, ticker: str, model_name: str = "gpt-4o"):
        """Initialize portfolio management agent."""
        super().__init__(ticker, "Portfolio Management Specialist", model_name)

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
                "portfolio_context": portfolio_context,
                "portfolio_analysis": portfolio_analysis,
                "final_decision": final_decision,
                "action": final_decision.get("action", "hold"),
                "confidence": final_decision.get("confidence", 0.5),
                "reasoning": final_decision.get("reasoning", ""),
                "risk_assessment": final_decision.get("risk_assessment", {}),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Portfolio management analysis failed")
            return {"ticker": self.ticker, "error": str(e), "timestamp": datetime.now().isoformat()}

    def _extract_final_decision(self, analysis: str) -> dict[str, Any]:
        """Extract final decision from analysis."""
        # Simple decision extraction - can be enhanced with structured parsing
        analysis_lower = analysis.lower()

        action = "hold"
        if "buy" in analysis_lower or "strong buy" in analysis_lower:
            action = "buy"
        elif "sell" in analysis_lower or "strong sell" in analysis_lower:
            action = "sell"

        return {
            "action": action,
            "quantity": 100,  # Default quantity - should be calculated based on risk
            "confidence": 0.6,
            "reasoning": analysis,
            "risk_assessment": {"risk_level": "medium"},
        }


class DebateRoomAgent(InvestmentAnalysisAgent):
    """Debate room that facilitates bull vs bear researcher discussion with sophisticated scoring like A_Share."""

    def __init__(self, ticker: str, model_name: str = "gpt-4o"):
        """Initialize debate room agent."""
        super().__init__(ticker, "Debate Room Moderator", model_name)

    async def run(self, **kwargs) -> dict[str, Any]:
        """Facilitate debate between bull and bear researchers with sophisticated scoring like A_Share."""
        bull_thesis = kwargs.get("bull_thesis", {})
        bear_thesis = kwargs.get("bear_thesis", {})
        analysis_request = kwargs.get("analysis_request", "debate room analysis")

        try:
            # Generate debate analysis
            analysis_prompt = {
                "ticker": self.ticker,
                "bull_thesis": json.dumps(bull_thesis, indent=2),
                "bear_thesis": json.dumps(bear_thesis, indent=2),
                "analysis_request": analysis_request,
            }

            debate_analysis = await super().run(**analysis_prompt)

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
                "debate_outcome": debate_outcome,
                "mixed_confidence": mixed_confidence,
                "argument_strength": argument_strength,
                "consensus_level": consensus_level,
                "debate_score": self._calculate_debate_score(bull_thesis, bear_thesis),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.exception("Debate room analysis failed")
            return {"ticker": self.ticker, "error": str(e), "timestamp": datetime.now().isoformat()}

    def _calculate_mixed_confidence_with_weights(self, bull_confidence, bear_confidence, debate_analysis):
        """Calculate mixed confidence with sophisticated weighting like A_Share."""
        # Base mixed confidence
        confidence_diff = abs(bull_confidence - bear_confidence)
        avg_confidence = (bull_confidence + bear_confidence) / 2

        # Weight based on confidence difference
        if confidence_diff > 0.4:
            # One side much more confident - weight towards stronger side
            mixed_confidence = max(bull_confidence, bear_confidence)
        elif confidence_diff < 0.1:
            # Similar confidence levels - use average
            mixed_confidence = avg_confidence
        else:
            # Moderate difference - use weighted average
            stronger_confidence = max(bull_confidence, bear_confidence)
            weaker_confidence = min(bull_confidence, bear_confidence)
            mixed_confidence = stronger_confidence * 0.6 + weaker_confidence * 0.4

        # Adjust based on debate analysis quality
        analysis_quality = self._assess_debate_quality(debate_analysis)
        mixed_confidence = mixed_confidence * (0.8 + 0.2 * analysis_quality)

        return min(max(mixed_confidence, 0.1), 1.0)

    def _extract_debate_outcome_detailed(self, debate_analysis, bull_thesis, bear_thesis):
        """Extract detailed debate outcome with sophisticated analysis like A_Share."""
        analysis_lower = debate_analysis.lower()
        bull_thesis_points = bull_thesis.get("thesis_points", [])
        bear_thesis_points = bear_thesis.get("thesis_points", [])

        # Count argument quality indicators
        bull_quality_indicators = self._count_quality_indicators(bull_thesis_points)
        bear_quality_indicators = self._count_quality_indicators(bear_thesis_points)

        # Determine outcome based on analysis and argument strength
        if "bullish case stronger" in analysis_lower or "bullish wins" in analysis_lower:
            outcome = "bullish_wins"
        elif "bearish case stronger" in analysis_lower or "bearish wins" in analysis_lower:
            outcome = "bearish_wins"
        elif "balanced" in analysis_lower or "inconclusive" in analysis_lower:
            outcome = "balanced"
        elif bull_quality_indicators > bear_quality_indicators + 1:
            outcome = "bullish_wins"
        elif bear_quality_indicators > bull_quality_indicators + 1:
            outcome = "bearish_wins"
        else:
            outcome = "balanced"

        return outcome

    def _calculate_argument_strength(self, bull_thesis, bear_thesis):
        """Calculate argument strength with sophisticated scoring like A_Share."""
        bull_thesis_points = bull_thesis.get("thesis_points", [])
        bear_thesis_points = bear_thesis.get("thesis_points", [])

        # Calculate strength based on thesis point quality and quantity
        bull_strength = self._calculate_thesis_strength_score(bull_thesis_points)
        bear_strength = self._calculate_thesis_strength_score(bear_thesis_points)

        return {
            "bull_strength": bull_strength,
            "bear_strength": bear_strength,
            "strength_difference": abs(bull_strength - bear_strength),
        }

    def _calculate_thesis_strength_score(self, thesis_points):
        """Calculate thesis strength score based on point quality."""
        if not thesis_points:
            return 0.0

        strength_score = 0.0
        for point in thesis_points:
            # Score based on point characteristics
            point_length = len(point)
            has_data = any(keyword in point.lower() for keyword in ["data", "evidence", "metric", "ratio"])
            has_analysis = any(keyword in point.lower() for keyword in ["analysis", "indicates", "suggests", "shows"])

            # Base score for having a point
            point_score = 0.2

            # Bonus for length (more detailed analysis)
            if point_length > 50:
                point_score += 0.1
            elif point_length > 100:
                point_score += 0.2

            # Bonus for data-driven arguments
            if has_data:
                point_score += 0.2

            # Bonus for analytical content
            if has_analysis:
                point_score += 0.2

            # Cap individual point score
            point_score = min(point_score, 0.8)

            strength_score += point_score

        # Normalize by number of points
        return min(strength_score / len(thesis_points), 1.0)

    def _count_quality_indicators(self, thesis_points):
        """Count quality indicators in thesis points."""
        if not thesis_points:
            return 0

        quality_indicators = 0
        for point in thesis_points:
            point_lower = point.lower()
            # Count quality indicators
            if any(
                indicator in point_lower
                for indicator in [
                    "evidence",
                    "data",
                    "metric",
                    "ratio",
                    "analysis",
                    "trend",
                    "growth",
                    "risk",
                    "opportunity",
                    "strength",
                    "weakness",
                    "potential",
                    "outlook",
                ]
            ):
                quality_indicators += 1

        return quality_indicators

    def _assess_debate_quality(self, debate_analysis):
        """Assess the quality of debate analysis."""
        analysis_lower = debate_analysis.lower()

        quality_indicators = [
            "balanced",
            "comprehensive",
            "detailed",
            "thorough",
            "well-reasoned",
            "logical",
            "structured",
            "analytical",
            "objective",
            "nuanced",
        ]

        quality_count = sum(1 for indicator in quality_indicators if indicator in analysis_lower)

        # Normalize to 0-1 range
        return min(quality_count / len(quality_indicators), 1.0)

    def _calculate_consensus_level(self, bull_thesis, bear_thesis):
        """Calculate consensus level between researchers."""
        bull_confidence = bull_thesis.get("confidence", 0.5)
        bear_confidence = bear_thesis.get("confidence", 0.5)

        # Calculate consensus based on confidence similarity
        confidence_similarity = 1 - abs(bull_confidence - bear_confidence)

        # Adjust for thesis alignment
        bull_thesis_points = bull_thesis.get("thesis_points", [])
        bear_thesis_points = bear_thesis.get("thesis_points", [])

        thesis_alignment = self._calculate_thesis_alignment(bull_thesis_points, bear_thesis_points)

        # Combine confidence similarity and thesis alignment
        consensus_level = confidence_similarity * 0.6 + thesis_alignment * 0.4

        return consensus_level

    def _calculate_thesis_alignment(self, bull_points, bear_points):
        """Calculate alignment between thesis points."""
        if not bull_points or not bear_points:
            return 0.5

        # Simple alignment calculation based on keyword overlap
        bull_keywords = set()
        bear_keywords = set()

        for point in bull_points:
            words = point.lower().split()
            bull_keywords.update([word for word in words if len(word) > 3])

        for point in bear_points:
            words = point.lower().split()
            bear_keywords.update([word for word in words if len(word) > 3])

        # Calculate Jaccard similarity
        intersection = bull_keywords.intersection(bear_keywords)
        union = bull_keywords.union(bear_keywords)

        if not union:
            return 0.5

        return len(intersection) / len(union)

    def _calculate_debate_score(self, bull_thesis, bear_thesis):
        """Calculate overall debate score."""
        bull_confidence = bull_thesis.get("confidence", 0.5)
        bear_confidence = bear_thesis.get("confidence", 0.5)

        # Calculate debate score based on confidence levels and argument strength
        avg_confidence = (bull_confidence + bear_confidence) / 2

        # Adjust for consensus
        consensus_level = self._calculate_consensus_level(bull_thesis, bear_thesis)

        debate_score = avg_confidence * consensus_level

        return debate_score


class RiskManagementAgent(InvestmentAnalysisAgent):
    """Specialized agent for risk assessment and management."""

    def __init__(self, ticker: str, model_name: str = "gpt-4o"):
        """Initialize risk management agent."""
        super().__init__(ticker, "Risk Management Specialist", model_name)

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

    def _extract_risk_level(self, analysis: str) -> str:
        """Extract risk level from analysis."""
        analysis_lower = analysis.lower()
        if "high risk" in analysis_lower or "very risky" in analysis_lower:
            return "high"
        elif "low risk" in analysis_lower or "minimal risk" in analysis_lower:
            return "low"
        else:
            return "medium"

    def _extract_risk_factors(self, analysis: str) -> list[str]:
        """Extract risk factors from analysis."""
        lines = analysis.split("\n")
        risk_factors = []

        for line in lines:
            line = line.strip()
            if line and any(keyword in line.lower() for keyword in ["risk", "threat", "concern", "vulnerability"]):
                risk_factors.append(line)

        return risk_factors[:5]  # Return top 5 risk factors

    def _extract_mitigation_strategies(self, analysis: str) -> list[str]:
        """Extract mitigation strategies from analysis."""
        lines = analysis.split("\n")
        strategies = []

        for line in lines:
            line = line.strip()
            if line and any(
                keyword in line.lower() for keyword in ["mitigate", "reduce", "hedge", "protect", "diversify"]
            ):
                strategies.append(line)

        return strategies[:5]  # Return top 5 strategies

    def _calculate_var_estimate(self, analysis: str) -> float:
        """Calculate Value at Risk estimate."""
        # Simple VaR calculation based on risk level mentioned in analysis
        if "high" in self._extract_risk_level(analysis).lower():
            return 0.15  # 15% VaR
        elif "low" in self._extract_risk_level(analysis).lower():
            return 0.05  # 5% VaR
        else:
            return 0.10  # 10% VaR


class MacroAnalysisAgent(InvestmentAnalysisAgent):
    """Specialized agent for macroeconomic analysis."""

    def __init__(self, ticker: str, model_name: str = "gpt-4o"):
        """Initialize macro analysis agent."""
        super().__init__(ticker, "Macro Analysis Specialist", model_name)

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

    def _extract_economic_outlook(self, analysis: str) -> str:
        """Extract economic outlook from analysis."""
        analysis_lower = analysis.lower()
        if "positive outlook" in analysis_lower or "bullish" in analysis_lower:
            return "positive"
        elif "negative outlook" in analysis_lower or "bearish" in analysis_lower:
            return "negative"
        else:
            return "neutral"

    def _extract_policy_impact(self, analysis: str) -> str:
        """Extract policy impact from analysis."""
        analysis_lower = analysis.lower()
        if "supportive policy" in analysis_lower or "favorable" in analysis_lower:
            return "positive"
        elif "restrictive policy" in analysis_lower or "unfavorable" in analysis_lower:
            return "negative"
        else:
            return "neutral"

    def _extract_market_cycle(self, analysis: str) -> str:
        """Extract market cycle from analysis."""
        analysis_lower = analysis.lower()
        if "bull market" in analysis_lower or "expansion" in analysis_lower:
            return "bull"
        elif "bear market" in analysis_lower or "recession" in analysis_lower:
            return "bear"
        else:
            return "neutral"

    def _extract_macro_recommendation(self, analysis: str) -> str:
        """Extract macro recommendation from analysis."""
        analysis_lower = analysis.lower()
        if "buy" in analysis_lower or "invest" in analysis_lower:
            return "buy"
        elif "sell" in analysis_lower or "avoid" in analysis_lower:
            return "sell"
        else:
            return "hold"


class MacroNewsAgent(InvestmentAnalysisAgent):
    """Specialized agent for macro news analysis."""

    def __init__(self, ticker: str, model_name: str = "gpt-4o"):
        """Initialize macro news agent."""
        super().__init__(ticker, "Macro News Specialist", model_name)

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

    def _extract_key_events(self, analysis: str) -> list[str]:
        """Extract key events from analysis."""
        lines = analysis.split("\n")
        events = []

        for line in lines:
            line = line.strip()
            if line and any(
                keyword in line.lower() for keyword in ["announcement", "release", "policy", "decision", "report"]
            ):
                events.append(line)

        return events[:5]  # Return top 5 events

    def _extract_market_impact(self, analysis: str) -> str:
        """Extract market impact from analysis."""
        analysis_lower = analysis.lower()
        if "positive impact" in analysis_lower or "bullish impact" in analysis_lower:
            return "positive"
        elif "negative impact" in analysis_lower or "bearish impact" in analysis_lower:
            return "negative"
        else:
            return "neutral"

    def _extract_policy_relevance(self, analysis: str) -> str:
        """Extract policy relevance from analysis."""
        analysis_lower = analysis.lower()
        if "high policy relevance" in analysis_lower or "policy driven" in analysis_lower:
            return "high"
        elif "low policy relevance" in analysis_lower or "market driven" in analysis_lower:
            return "low"
        else:
            return "medium"

    def _extract_news_sentiment(self, analysis: str) -> str:
        """Extract news sentiment from analysis."""
        analysis_lower = analysis.lower()
        if "positive sentiment" in analysis_lower or "optimistic" in analysis_lower:
            return "positive"
        elif "negative sentiment" in analysis_lower or "pessimistic" in analysis_lower:
            return "negative"
        else:
            return "neutral"


class ResearcherBullAgent(InvestmentAnalysisAgent):
    """Specialized agent for bullish research analysis."""

    def __init__(self, ticker: str, model_name: str = "gpt-4o"):
        """Initialize bullish research agent."""
        super().__init__(ticker, "Bullish Researcher", model_name)

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
    """Specialized agent for bearish research analysis."""

    def __init__(self, ticker: str, model_name: str = "gpt-4o"):
        """Initialize bearish research agent."""
        super().__init__(ticker, "Bearish Researcher", model_name)

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


class MultidisciplinaryInvestmentTeam:
    """Agent that coordinates all investment analysis specialists following A_Share_investment_Agent workflow."""

    def __init__(self, ticker: str, model_name: str = "gpt-4o"):
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
            technical_task = self.technical_agent.run(market_data=market_data, analysis_request="technical analysis")
            fundamental_task = self.fundamental_agent.run(
                market_data=market_data, analysis_request="fundamental analysis"
            )
            sentiment_task = self.sentiment_agent.run(
                news_data=market_data.get("news_data", {}),
                market_context=market_data.get("market_context", {}),
                analysis_request="sentiment analysis",
            )
            valuation_task = self.valuation_agent.run(
                financial_data=market_data.get("financial_data", {}),
                market_data=market_data,
                analysis_request="valuation analysis",
            )
            # Macro news runs in parallel like A_Share
            macro_news_task = self.macro_news_agent.run(market_data=market_data, analysis_request="macro news analysis")

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


async def run_share_investment_analysis(ticker: str, analysis_request: str, model_name: str = "gpt-4o") -> str:
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

            response = f"""### Share Investment Analysis Report:

**Stock Ticker:** {ticker}
**Analysis Type:** {analysis_request}
**Analysis Date:** {final_analysis.get("analysis_timestamp", "Unknown")}

---

#### Final Investment Decision:
- **Action:** {final_decision.get("action", "N/A").upper()}
- **Confidence:** {final_decision.get("confidence", 0):.1%}
- **Quantity:** {final_decision.get("quantity", 0)} shares
- **Reasoning:** {final_decision.get("reasoning", "No reasoning provided")}

---

#### Core Specialist Analysis Summary:

**Technical Analysis:** {specialist_analyses.get("technical_analysis", {}).get("signal", "N/A")} (Confidence: {specialist_analyses.get("technical_analysis", {}).get("confidence", 0):.1%})

**Fundamental Analysis:** {specialist_analyses.get("fundamental_analysis", {}).get("signal", "N/A")} (Confidence: {specialist_analyses.get("fundamental_analysis", {}).get("confidence", 0):.1%})

**Sentiment Analysis:** {specialist_analyses.get("sentiment_analysis", {}).get("signal", "N/A")} (Confidence: {specialist_analyses.get("sentiment_analysis", {}).get("confidence", 0):.1%})

**Valuation Analysis:** {specialist_analyses.get("valuation_analysis", {}).get("signal", "N/A")} (Confidence: {specialist_analyses.get("valuation_analysis", {}).get("confidence", 0):.1%})

---

#### Research Team Analysis:

**Bullish Researcher:** {researcher_analyses.get("bull_researcher", {}).get("perspective", "N/A")} (Confidence: {researcher_analyses.get("bull_researcher", {}).get("confidence", 0):.1%})

**Bearish Researcher:** {researcher_analyses.get("bear_researcher", {}).get("perspective", "N/A")} (Confidence: {researcher_analyses.get("bear_researcher", {}).get("confidence", 0):.1%})

---

#### Debate Room Outcome:
**Debate Result:** {debate_room.get("debate_outcome", "N/A")}
**Mixed Confidence:** {debate_room.get("mixed_confidence", 0):.1%}
**Analysis:** {debate_room.get("debate_analysis", "No debate analysis available")[:200]}...

---

#### Risk Management Assessment:
**Risk Level:** {risk_management.get("risk_level", "N/A")}
**VaR Estimate:** {risk_management.get("var_estimate", 0):.1%}
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
- **Current Price:** {final_analysis.get("market_data", {}).get("market_data", {}).get("current_price", "N/A")}
- **Market Cap:** {final_analysis.get("market_data", {}).get("market_data", {}).get("market_cap", "N/A")}
- **Volume:** {final_analysis.get("market_data", {}).get("market_data", {}).get("volume", "N/A")}

---

*Analysis performed by AI Share Investment Agent*
*Specialist Team: Market Data, Technical, Fundamental, Sentiment, Valuation, Macro News, Bull/Bear Researchers, Debate Room, Risk Management, Macro Analyst, Portfolio Management*
*Model: {model_name}*
"""

        return response
    except Exception as e:
        error_msg = f"Error during investment analysis: {e!s}"
        logger.exception("Error during investment analysis")
        return error_msg
