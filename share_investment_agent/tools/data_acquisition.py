"""Data acquisition tools for Chinese A-share market data."""

import asyncio
from datetime import datetime
from typing import Any

import requests
import yfinance as yf

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Global semaphore for API rate limiting
API_SEMAPHORE = asyncio.Semaphore(2)


class MaxRetriesExceededError(Exception):
    """Raised when API retry attempts are exhausted."""


class UnsupportedIndicatorError(Exception):
    """Raised when an unsupported indicator is requested."""


async def safe_api_call(url: str, headers: dict[str, str] | None = None, timeout: int = 10, max_retries: int = 3):
    """Safe API call with retry logic and exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers or {}, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.exception("API call failed after max retries")
                raise

    raise MaxRetriesExceededError


class MarketDataAcquisition:
    """Handles acquisition of market data for Chinese A-shares."""

    def __init__(self):
        """Initialize data acquisition with API endpoints."""
        self.session = requests.Session()
        # Enhanced headers to avoid 403 errors
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Referer": "https://finance.sina.com.cn/",
            "Cache-Control": "max-age=0",
        })

    async def get_comprehensive_data(self, ticker: str) -> dict[str, Any]:
        """Get comprehensive market data for a ticker using Yahoo Finance."""
        try:
            logger.info(f"Acquiring comprehensive market data for {ticker}")

            # Convert ticker to Yahoo Finance format
            yahoo_ticker = self._convert_to_yahoo_ticker(ticker)
            logger.info(f"Converted ticker {ticker} to Yahoo Finance format: {yahoo_ticker}")

            # Get real market data using Yahoo Finance
            market_data = await self._get_yahoo_finance_data(yahoo_ticker)

            if "error" in market_data:
                logger.error(f"Failed to retrieve market data for {ticker}: {market_data['error']}")
                return {"ticker": ticker, "error": market_data["error"]}

            logger.info(f"Successfully retrieved market data for {ticker}: {list(market_data.keys())}")

            # Get additional technical indicators
            technical_indicators = await self._calculate_technical_indicators(market_data)

            # Compile comprehensive data package
            comprehensive_data = {
                "ticker": ticker,
                "current_price": market_data.get("current_price"),
                "market_cap": market_data.get("market_cap"),
                "volume": market_data.get("volume"),
                "pe_ratio": market_data.get("pe_ratio"),
                "revenue": market_data.get("revenue"),
                "eps": market_data.get("eps"),
                "historical_prices": market_data.get("historical_prices", []),
                "technical_indicators": technical_indicators,
                "real_time_data": market_data,
                "financial_data": market_data.get("financial_data", {}),
                "news_data": market_data.get("news_data", []),
                "market_context": self._get_market_context(),
                "acquisition_timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Successfully acquired comprehensive data for {ticker}")
            return comprehensive_data

        except Exception as e:
            logger.exception(f"Failed to acquire comprehensive data for {ticker}")
            return {"ticker": ticker, "error": str(e)}

    def _convert_to_yahoo_ticker(self, ticker: str) -> str:
        """Convert Chinese ticker to Yahoo Finance format."""
        # Handle different Chinese stock exchanges
        if len(ticker) == 6 and ticker.isdigit():
            # 6-digit numeric ticker - add .SS for Shanghai or .SZ for Shenzhen
            if ticker.startswith(("000", "001", "002", "003", "300")):
                return f"{ticker}.SZ"  # Shenzhen Stock Exchange
            else:
                return f"{ticker}.SS"  # Shanghai Stock Exchange
        elif "." in ticker:
            return ticker  # Already in correct format
        else:
            return f"{ticker}.SS"  # Default to Shanghai

    async def _get_yahoo_finance_data(self, yahoo_ticker: str) -> dict[str, Any]:
        """Get comprehensive market data using Yahoo Finance API."""
        try:
            logger.info(f"Fetching Yahoo Finance data for {yahoo_ticker}")

            # Use yfinance to get stock data
            stock = yf.Ticker(yahoo_ticker)

            # Get basic info
            info = stock.info
            logger.info(f"Retrieved stock info for {yahoo_ticker}, keys: {list(info.keys())[:10] if info else 'None'}")

            if not info or "regularMarketPrice" not in info:
                logger.error(f"No market price data found for ticker {yahoo_ticker}")
                return {"error": f"No data found for ticker {yahoo_ticker}"}

            # Get historical data
            hist = stock.history(period="1y", interval="1d")
            if hist.empty:
                logger.error(f"No historical data for ticker {yahoo_ticker}")
                return {"error": f"No historical data for ticker {yahoo_ticker}"}

            # Extract current data
            current_price = info.get("regularMarketPrice", 0)
            market_cap = info.get("marketCap", 0)
            volume = info.get("regularMarketVolume", 0)
            pe_ratio = info.get("trailingPE", 0)
            revenue = info.get("totalRevenue", 0)
            eps = info.get("trailingEps", 0)

            logger.info(
                f"Extracted key metrics for {yahoo_ticker}: price={current_price}, cap={market_cap}, volume={volume}"
            )

            # Prepare historical prices
            historical_prices = []
            for date, row in hist.iterrows():
                historical_prices.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                })

            # Get financial statements data
            financial_data = {
                "profitability": {
                    "roe": info.get("returnOnEquity", 0),
                    "roa": info.get("returnOnAssets", 0),
                    "gross_margin": info.get("grossMargins", 0),
                    "net_margin": info.get("profitMargins", 0),
                },
                "growth": {
                    "revenue_growth": info.get("revenueGrowth", 0),
                    "earnings_growth": info.get("earningsGrowth", 0),
                    "eps_growth": info.get("earningsQuarterlyGrowth", 0),
                },
                "financial_health": {
                    "debt_to_equity": info.get("debtToEquity", 0),
                    "current_ratio": info.get("currentRatio", 0),
                    "quick_ratio": info.get("quickRatio", 0),
                },
                "valuation": {
                    "pe_ratio": pe_ratio,
                    "pb_ratio": info.get("priceToBook", 0),
                    "ps_ratio": info.get("priceToSales", 0),
                    "ev_ebitda": info.get("enterpriseToEbitda", 0),
                },
            }

            return {
                "name": info.get("shortName", yahoo_ticker),
                "current_price": current_price,
                "market_cap": market_cap,
                "volume": volume,
                "pe_ratio": pe_ratio,
                "revenue": revenue,
                "eps": eps,
                "historical_prices": historical_prices,
                "financial_data": financial_data,
                "currency": info.get("currency", "CNY"),
                "exchange": info.get("exchange", "Unknown"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
            }

        except Exception as e:
            logger.exception(f"Yahoo Finance API failed for {yahoo_ticker}")
            return {"error": str(e)}

    async def _calculate_technical_indicators(self, market_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate technical indicators from historical price data."""
        try:
            historical_prices = market_data.get("historical_prices", [])
            if len(historical_prices) < 50:
                return {"error": "Insufficient data for technical indicators"}

            closes = [p["close"] for p in historical_prices]

            # Calculate RSI (14-day)
            rsi = self._calculate_rsi(closes, 14)

            # Calculate MACD
            macd = self._calculate_macd(closes)

            # Calculate Bollinger Bands
            bollinger = self._calculate_bollinger_bands(closes)

            # Calculate moving averages
            sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1]
            sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else closes[-1]
            ema_12 = self._calculate_ema(closes, 12)[-1] if len(closes) >= 12 else closes[-1]

            return {
                "rsi": rsi,
                "macd": macd,
                "bollinger_bands": bollinger,
                "moving_averages": {
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "ema_12": ema_12,
                },
                "current_price": closes[-1],
                "price_change": closes[-1] - closes[-2] if len(closes) >= 2 else 0,
                "price_change_pct": ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else 0,
            }

        except Exception as e:
            logger.exception("Failed to calculate technical indicators")
            return {"error": str(e)}

    def _calculate_rsi(self, prices: list[float], period: int = 14) -> dict[str, Any]:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return {"rsi": 50, "overbought": False, "oversold": False}

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [d for d in deltas if d > 0]
        losses = [abs(d) for d in deltas if d < 0]

        avg_gain = sum(gains[-period:]) / period if len(gains) >= period else 0
        avg_loss = sum(losses[-period:]) / period if len(losses) >= period else 0

        if avg_loss == 0:
            return {"rsi": 100, "overbought": True, "oversold": False}

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return {
            "rsi": rsi,
            "overbought": rsi > 70,
            "oversold": rsi < 30,
            "signal": "buy" if rsi < 30 else "sell" if rsi > 70 else "hold",
        }

    def _calculate_macd(self, prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict[str, Any]:
        """Calculate MACD indicator."""
        if len(prices) < slow:
            return {"macd": 0, "signal_line": 0, "histogram": 0, "signal": "hold"}

        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        ema_signal = self._calculate_ema(ema_fast, signal)

        macd_line = ema_fast[-1] - ema_slow[-1] if len(ema_fast) > 0 and len(ema_slow) > 0 else 0
        signal_line = ema_signal[-1] if len(ema_signal) > 0 else 0
        histogram = macd_line - signal_line

        return {
            "macd": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
            "signal": "buy"
            if histogram > 0 and macd_line > 0
            else "sell"
            if histogram < 0 and macd_line < 0
            else "hold",
        }

    def _calculate_bollinger_bands(self, prices: list[float], period: int = 20, std_dev: float = 2) -> dict[str, Any]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return {"upper_band": 0, "middle_band": 0, "lower_band": 0, "signal": "hold"}

        recent_prices = prices[-period:]
        sma = sum(recent_prices) / period
        variance = sum((price - sma) ** 2 for price in recent_prices) / period
        std = variance**0.5

        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        current_price = prices[-1]

        return {
            "upper_band": upper_band,
            "middle_band": sma,
            "lower_band": lower_band,
            "signal": "buy" if current_price < lower_band else "sell" if current_price > upper_band else "hold",
        }

    def _calculate_ema(self, prices: list[float], period: int) -> list[float]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices

        multiplier = 2 / (period + 1)
        ema = [sum(prices[:period]) / period]  # Start with SMA

        multiplier = 2 / (period + 1)
        ema = [sum(prices[:period]) / period]  # Start with SMA

        for price in prices[period:]:
            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))

        return ema

    def _get_market_context(self) -> dict[str, Any]:
        """Get current market context."""
        return {
            "market_sentiment": "neutral",
            "volatility_index": "moderate",
            "sector_performance": "+0.5%",
            "economic_outlook": "stable",
            "timestamp": datetime.now().isoformat(),
        }


# ... (rest of the code remains the same)
