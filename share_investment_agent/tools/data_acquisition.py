"""Data acquisition tools for Chinese A-share market data."""

from datetime import datetime, timedelta
from typing import Any

import requests

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class MarketDataAcquisition:
    """Handles acquisition of market data for Chinese A-shares."""

    def __init__(self):
        """Initialize data acquisition with API endpoints."""
        self.base_urls = {
            "akshare": "https://akshare.akfamily.xyz/data",
            "sina": "https://hq.sinajs.cn",
            "eastmoney": "https://push2.eastmoney.com/api/qt",
        }
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

    async def get_comprehensive_data(self, ticker: str) -> dict[str, Any]:
        """Get comprehensive market data for a ticker."""
        try:
            logger.info(f"Acquiring comprehensive market data for {ticker}")

            # Get real-time data
            real_time_data = await self.get_real_time_data(ticker)

            # Get historical data
            historical_data = await self.get_historical_data(ticker)

            # Get financial data
            financial_data = await self.get_financial_data(ticker)

            # Get news data
            news_data = await self.get_news_data(ticker)

            # Compile comprehensive data package
            comprehensive_data = {
                "ticker": ticker,
                "real_time_data": real_time_data,
                "historical_data": historical_data,
                "financial_data": financial_data,
                "news_data": news_data,
                "market_context": self._get_market_context(),
                "acquisition_timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Successfully acquired comprehensive data for {ticker}")
            return comprehensive_data
        except Exception as e:
            logger.exception("Failed to acquire comprehensive data for {ticker}")
            return {"ticker": ticker, "error": str(e)}

    async def get_real_time_data(self, ticker: str) -> dict[str, Any]:
        """Get real-time market data."""
        try:
            # Use Sina finance API for real-time data
            url = f"https://hq.sinajs.cn/list={ticker}"

            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            # Parse Sina finance response
            data_text = response.text
            if not data_text or "var hq_str_" not in data_text:
                raise ValueError("Invalid response")

            # Extract data
            data_part = data_text.split('"')[1]
            data_fields = data_part.split(",")

            if len(data_fields) < 32:
                raise ValueError("Incomplete data")

            real_time_data = {
                "name": data_fields[0],
                "open": float(data_fields[1]),
                "close_prev": float(data_fields[2]),
                "current": float(data_fields[3]),
                "high": float(data_fields[4]),
                "low": float(data_fields[5]),
                "volume": int(data_fields[8]),
                "amount": float(data_fields[9]),
                "date": data_fields[30] + " " + data_fields[31] if len(data_fields) > 31 else data_fields[30],
                "timestamp": datetime.now().isoformat(),
            }

            # Calculate additional metrics
            real_time_data["change"] = real_time_data["current"] - real_time_data["close_prev"]
            real_time_data["change_percent"] = (real_time_data["change"] / real_time_data["close_prev"]) * 100

            return real_time_data
        except Exception as e:
            logger.exception("Failed to get real-time data for {ticker}")
            return {"error": str(e)}

    async def get_historical_data(self, ticker: str, days: int = 250) -> list[dict[str, Any]]:
        """Get historical price data."""
        try:
            # For demo purposes, generate sample historical data
            # In production, this would use Akshare or similar API
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            historical_data = []
            current_date = start_date

            # Generate sample data (replace with real API call)
            base_price = 10.0
            for i in range(days):
                if current_date.weekday() < 5:  # Weekdays only
                    # Simulate price movement
                    price_change = (hash(f"{ticker}{i}") % 100 - 50) / 1000
                    base_price += price_change

                    historical_data.append({
                        "date": current_date.strftime("%Y-%m-%d"),
                        "open": base_price - 0.1,
                        "high": base_price + 0.2,
                        "low": base_price - 0.15,
                        "close": base_price,
                        "volume": (hash(f"{ticker}{i}") % 1000000) + 100000,
                        "amount": base_price * ((hash(f"{ticker}{i}") % 1000000) + 100000),
                    })

                current_date += timedelta(days=1)

            return historical_data[-100:]  # Return last 100 trading days

        except Exception as e:
            logger.exception("Failed to get historical data for {ticker}")
            return [{"error": str(e)}]

    async def get_financial_data(self, ticker: str) -> dict[str, Any]:
        """Get financial metrics and statements."""
        try:
            # For demo purposes, generate sample financial data
            # In production, this would use real financial data APIs

            financial_data = {
                "basic_info": {
                    "ticker": ticker,
                    "company_name": f"Company {ticker}",
                    "industry": "Technology",
                    "market_cap": (hash(ticker) % 1000000000) + 100000000,
                    "pe_ratio": 15.5 + (hash(ticker) % 20),
                    "pb_ratio": 2.1 + (hash(ticker) % 3),
                    "dividend_yield": 0.02 + (hash(ticker) % 30) / 1000,
                },
                "profitability": {
                    "revenue": (hash(ticker) % 1000000000) + 500000000,
                    "net_income": (hash(ticker) % 100000000) + 50000000,
                    "gross_margin": 0.25 + (hash(ticker) % 20) / 100,
                    "net_margin": 0.10 + (hash(ticker) % 15) / 100,
                    "roe": 0.12 + (hash(ticker) % 18) / 100,
                    "roa": 0.06 + (hash(ticker) % 10) / 100,
                },
                "growth": {
                    "revenue_growth": 0.05 + (hash(ticker) % 25) / 100,
                    "earnings_growth": 0.08 + (hash(ticker) % 30) / 100,
                    "eps_growth": 0.07 + (hash(ticker) % 28) / 100,
                },
                "financial_health": {
                    "current_ratio": 1.5 + (hash(ticker) % 100) / 100,
                    "debt_to_equity": 0.3 + (hash(ticker) % 70) / 100,
                    "interest_coverage": 5.0 + (hash(ticker) % 15),
                    "free_cash_flow": (hash(ticker) % 200000000) + 50000000,
                },
                "timestamp": datetime.now().isoformat(),
            }

            return financial_data
        except Exception as e:
            logger.exception(f"Failed to get financial data for {ticker}: {e}")
            return {"error": str(e)}

    async def get_news_data(self, ticker: str, days: int = 7) -> list[dict[str, Any]]:
        """Get recent news and sentiment data."""
        try:
            # For demo purposes, generate sample news data
            # In production, this would use real news APIs

            news_data = []
            news_templates = [
                f"{ticker} reports strong quarterly earnings, exceeding analyst expectations",
                f"{ticker} announces new product launch in Chinese market",
                f"Analysts upgrade {ticker} rating citing growth prospects",
                f"{ticker} faces regulatory challenges in new expansion",
                f"Market sentiment positive on {ticker} technical breakout",
                f"{ticker} secures major partnership deal",
                f"Economic conditions favor {ticker} sector performance",
            ]

            for i in range(min(10, days)):  # Get up to 10 news items
                news_date = datetime.now() - timedelta(days=i)
                news_text = news_templates[i % len(news_templates)]

                # Simple sentiment classification
                positive_words = ["strong", "exceeding", "upgrade", "positive", "secures", "favor"]
                negative_words = ["challenges", "regulatory", "faces"]

                sentiment_score = 0
                for word in positive_words:
                    if word in news_text.lower():
                        sentiment_score += 1
                for word in negative_words:
                    if word in news_text.lower():
                        sentiment_score -= 1

                sentiment = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"

                news_data.append({
                    "title": news_text,
                    "date": news_date.strftime("%Y-%m-%d"),
                    "source": "Financial News",
                    "sentiment": sentiment,
                    "sentiment_score": max(-1, min(1, sentiment_score / 3)),
                    "url": f"https://example.com/news/{ticker}/{i}",
                })

            return news_data
        except Exception as e:
            logger.exception(f"Failed to get news data for {ticker}: {e}")
            return [{"error": str(e)}]

    def _get_market_context(self) -> dict[str, Any]:
        """Get overall market context."""
        try:
            # For demo purposes, generate sample market context
            market_context = {
                "market_indices": {
                    "shanghai_composite": {
                        "value": 3200.0 + (hash("market") % 500),
                        "change": (hash("market") % 100 - 50) / 10,
                        "change_percent": (hash("market") % 200 - 100) / 1000,
                    },
                    "shenzhen_component": {
                        "value": 11000.0 + (hash("sz") % 1000),
                        "change": (hash("sz") % 150 - 75) / 10,
                        "change_percent": (hash("sz") % 300 - 150) / 1000,
                    },
                },
                "market_sentiment": "neutral",
                "volatility_index": 15.0 + (hash("vol") % 10),
                "sector_performance": {
                    "technology": 0.05 + (hash("tech") % 80) / 1000,
                    "finance": 0.03 + (hash("fin") % 60) / 1000,
                    "healthcare": -0.02 + (hash("health") % 40) / 1000,
                },
                "timestamp": datetime.now().isoformat(),
            }

            return market_context
        except Exception as e:
            logger.exception(f"Failed to get market context: {e}")
            return {"error": str(e)}

    async def get_specific_indicator(self, ticker: str, indicator: str) -> Any:
        """Get specific technical indicator data."""
        try:
            # This would integrate with specialized APIs for different indicators
            indicator_methods = {
                "macd": self._calculate_macd,
                "rsi": self._calculate_rsi,
                "bollinger": self._calculate_bollinger_bands,
                "volume": self._get_volume_analysis,
            }

            if indicator in indicator_methods:
                return await indicator_methods[indicator](ticker)
            else:
                raise ValueError(f"Unsupported indicator: {indicator}")

        except Exception as e:
            logger.exception(f"Failed to get {indicator} for {ticker}: {e}")
            return {"error": str(e)}

    async def _calculate_macd(self, ticker: str) -> dict[str, Any]:
        """Calculate MACD indicator."""
        try:
            historical_data = await self.get_historical_data(ticker, 100)
            if len(historical_data) < 26:
                return {"error": "Insufficient data for MACD calculation"}

            closes = [float(d["close"]) for d in historical_data if "close" in d]

            # Simple MACD calculation (replace with proper implementation)
            ema12 = self._calculate_ema(closes, 12)
            ema26 = self._calculate_ema(closes, 26)
            macd_line = [e12 - e26 for e12, e26 in zip(ema12[-len(ema26) :], ema26, strict=False)]
            signal_line = self._calculate_ema(macd_line, 9)

            return {
                "macd": macd_line[-1] if macd_line else 0,
                "signal": signal_line[-1] if signal_line else 0,
                "histogram": (macd_line[-1] - signal_line[-1]) if macd_line and signal_line else 0,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": str(e)}

    async def _calculate_rsi(self, ticker: str, period: int = 14) -> dict[str, Any]:
        """Calculate RSI indicator."""
        try:
            historical_data = await self.get_historical_data(ticker, period + 10)
            if len(historical_data) < period + 1:
                return {"error": "Insufficient data for RSI calculation"}

            closes = [float(d["close"]) for d in historical_data if "close" in d]

            # Simple RSI calculation
            gains = []
            losses = []

            for i in range(1, len(closes)):
                change = closes[i] - closes[i - 1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            if len(gains) < period:
                return {"error": "Insufficient data points"}

            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period

            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            return {"rsi": rsi, "overbought": rsi > 70, "oversold": rsi < 30, "timestamp": datetime.now().isoformat()}

        except Exception as e:
            return {"error": str(e)}

    async def _calculate_bollinger_bands(self, ticker: str, period: int = 20, std_dev: float = 2) -> dict[str, Any]:
        """Calculate Bollinger Bands."""
        try:
            historical_data = await self.get_historical_data(ticker, period + 10)
            if len(historical_data) < period:
                return {"error": "Insufficient data for Bollinger Bands"}

            closes = [float(d["close"]) for d in historical_data if "close" in d]
            recent_closes = closes[-period:]

            if len(recent_closes) < period:
                return {"error": "Insufficient data points"}

            sma = sum(recent_closes) / period
            variance = sum((price - sma) ** 2 for price in recent_closes) / period
            std = variance**0.5

            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)
            current_price = recent_closes[-1]

            return {
                "upper_band": upper_band,
                "middle_band": sma,
                "lower_band": lower_band,
                "current_price": current_price,
                "position": "above_upper"
                if current_price > upper_band
                else "below_lower"
                if current_price < lower_band
                else "between",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": str(e)}

    async def _get_volume_analysis(self, ticker: str) -> dict[str, Any]:
        """Get volume analysis."""
        try:
            historical_data = await self.get_historical_data(ticker, 30)
            if len(historical_data) < 10:
                return {"error": "Insufficient data for volume analysis"}

            volumes = [int(d["volume"]) for d in historical_data if "volume" in d]
            recent_volumes = volumes[-10:]

            avg_volume = sum(recent_volumes) / len(recent_volumes)
            current_volume = recent_volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            return {
                "current_volume": current_volume,
                "average_volume": avg_volume,
                "volume_ratio": volume_ratio,
                "high_volume": volume_ratio > 1.5,
                "low_volume": volume_ratio < 0.7,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": str(e)}

    def _calculate_ema(self, prices: list[float], period: int) -> list[float]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices

        multiplier = 2 / (period + 1)
        ema = [sum(prices[:period]) / period]  # Start with SMA

        for price in prices[period:]:
            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))

        return ema
