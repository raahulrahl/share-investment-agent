"""Financial analysis tools for investment analysis."""

from datetime import datetime
from typing import Any

import pandas as pd

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)


class FinancialAnalyzer:
    """Handles financial analysis calculations and metrics."""

    def __init__(self):
        """Initialize financial analyzer."""
        self.risk_free_rate = 0.02  # 2% risk-free rate

    async def calculate_technical_indicators(self, price_history: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate comprehensive technical indicators."""
        try:
            if not price_history or len(price_history) < 20:
                return {"error": "Insufficient price history for technical analysis"}

            # Extract price data
            df = pd.DataFrame(price_history)

            if "close" not in df.columns:
                return {"error": "Price data missing 'close' column"}

            closes = df["close"].astype(float)
            highs = df["high"].astype(float) if "high" in df.columns else closes
            lows = df["low"].astype(float) if "low" in df.columns else closes
            volumes = df["volume"].astype(float) if "volume" in df.columns else pd.Series([100000] * len(closes))

            indicators = {}

            # Trend Indicators
            indicators.update(self._calculate_trend_indicators(closes))

            # Momentum Indicators
            indicators.update(self._calculate_momentum_indicators(closes))

            # Volatility Indicators
            indicators.update(self._calculate_volatility_indicators(closes, highs, lows))

            # Volume Indicators
            indicators.update(self._calculate_volume_indicators(closes, volumes))

            # Pattern Recognition
            indicators.update(self._identify_patterns(closes, highs, lows))

            indicators["timestamp"] = datetime.now().isoformat()

            logger.info("Technical indicators calculated successfully")
            return indicators
        except Exception as e:
            logger.exception("Failed to calculate technical indicators")
            return {"error": str(e)}

    def _calculate_trend_indicators(self, closes: pd.Series) -> dict[str, Any]:
        """Calculate trend-based indicators."""
        try:
            indicators = {}

            # Moving Averages
            indicators["sma_20"] = closes.rolling(window=20).mean().iloc[-1]
            indicators["sma_50"] = closes.rolling(window=50).mean().iloc[-1] if len(closes) >= 50 else None
            indicators["ema_12"] = closes.ewm(span=12).mean().iloc[-1]
            indicators["ema_26"] = closes.ewm(span=26).mean().iloc[-1]

            # MACD
            ema_12 = closes.ewm(span=12).mean()
            ema_26 = closes.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line

            indicators["macd"] = macd_line.iloc[-1]
            indicators["macd_signal"] = signal_line.iloc[-1]
            indicators["macd_histogram"] = histogram.iloc[-1]

            # Trend Direction
            current_price = closes.iloc[-1]
            sma_20 = indicators["sma_20"]

            if current_price > sma_20:
                indicators["trend"] = "bullish"
            elif current_price < sma_20:
                indicators["trend"] = "bearish"
            else:
                indicators["trend"] = "neutral"

            return indicators
        except Exception as e:
            logger.exception("Failed to calculate trend indicators")
            return {"error": str(e)}

    def _calculate_momentum_indicators(self, closes: pd.Series) -> dict[str, Any]:
        """Calculate momentum-based indicators."""
        try:
            indicators = {}

            # RSI
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            indicators["rsi"] = rsi.iloc[-1]
            indicators["rsi_overbought"] = rsi.iloc[-1] > 70
            indicators["rsi_oversold"] = rsi.iloc[-1] < 30

            # Stochastic Oscillator
            low_14 = closes.rolling(window=14).min()
            high_14 = closes.rolling(window=14).max()
            k_percent = 100 * ((closes - low_14) / (high_14 - low_14))
            d_percent = k_percent.rolling(window=3).mean()

            indicators["stochastic_k"] = k_percent.iloc[-1]
            indicators["stochastic_d"] = d_percent.iloc[-1]

            # Rate of Change (ROC)
            roc_10 = ((closes - closes.shift(10)) / closes.shift(10)) * 100
            indicators["roc_10"] = roc_10.iloc[-1]

            # Momentum
            momentum_10 = closes - closes.shift(10)
            indicators["momentum_10"] = momentum_10.iloc[-1]

            return indicators
        except Exception as e:
            logger.exception("Failed to calculate momentum indicators")
            return {"error": str(e)}

    def _calculate_volatility_indicators(self, closes: pd.Series, highs: pd.Series, lows: pd.Series) -> dict[str, Any]:
        """Calculate volatility-based indicators."""
        try:
            indicators = {}

            # Bollinger Bands
            sma_20 = closes.rolling(window=20).mean()
            std_20 = closes.rolling(window=20).std()

            indicators["bb_upper"] = sma_20 + (2 * std_20)
            indicators["bb_middle"] = sma_20
            indicators["bb_lower"] = sma_20 - (2 * std_20)
            indicators["bb_width"] = (indicators["bb_upper"] - indicators["bb_lower"]) / indicators["bb_middle"]

            # Position relative to Bollinger Bands
            current_price = closes.iloc[-1]
            bb_upper = indicators["bb_upper"].iloc[-1]
            bb_lower = indicators["bb_lower"].iloc[-1]

            if current_price > bb_upper:
                indicators["bb_position"] = "above_upper"
            elif current_price < bb_lower:
                indicators["bb_position"] = "below_lower"
            else:
                indicators["bb_position"] = "between"

            # Average True Range (ATR)
            high_low = highs - lows
            high_close = abs(highs - closes.shift())
            low_close = abs(lows - closes.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()

            indicators["atr"] = atr.iloc[-1]
            indicators["atr_percent"] = (atr.iloc[-1] / closes.iloc[-1]) * 100

            # Historical Volatility
            returns = closes.pct_change().dropna()
            volatility = returns.rolling(window=20).std() * (252**0.5)  # Annualized

            indicators["volatility"] = volatility.iloc[-1]

            return indicators
        except Exception as e:
            logger.exception("Failed to calculate volatility indicators")
            return {"error": str(e)}

    def _calculate_volume_indicators(self, closes: pd.Series, volumes: pd.Series) -> dict[str, Any]:
        """Calculate volume-based indicators."""
        try:
            indicators = {}

            # On-Balance Volume (OBV)
            obv = []
            obv_value = 0

            for i in range(len(closes)):
                if i == 0:
                    obv_value = volumes.iloc[i]
                else:
                    if closes.iloc[i] > closes.iloc[i - 1]:
                        obv_value += volumes.iloc[i]
                    elif closes.iloc[i] < closes.iloc[i - 1]:
                        obv_value -= volumes.iloc[i]
                    # No change, OBV stays the same

                obv.append(obv_value)

            indicators["obv"] = obv[-1] if obv else 0
            indicators["obv_trend"] = "up" if len(obv) > 1 and obv[-1] > obv[-2] else "down"

            # Volume Moving Average
            volume_sma = volumes.rolling(window=20).mean()
            indicators["volume_sma"] = volume_sma.iloc[-1]

            # Volume Ratio
            current_volume = volumes.iloc[-1]
            avg_volume = volume_sma.iloc[-1]
            indicators["volume_ratio"] = current_volume / avg_volume if avg_volume > 0 else 1

            # Price-Volume Trend
            price_change = closes.pct_change()
            # volume_change is available but not used in this calculation
            pvt = (price_change * volumes).cumsum()

            indicators["pvt"] = pvt.iloc[-1]

            return indicators
        except Exception as e:
            logger.exception("Failed to calculate volume indicators")
            return {"error": str(e)}

    def _identify_patterns(self, closes: pd.Series, highs: pd.Series, lows: pd.Series) -> dict[str, Any]:
        """Identify common chart patterns."""
        try:
            indicators = {}

            # Support and Resistance Levels
            recent_highs = highs.rolling(window=20).max()
            recent_lows = lows.rolling(window=20).min()

            indicators["resistance"] = recent_highs.iloc[-1]
            indicators["support"] = recent_lows.iloc[-1]

            current_price = closes.iloc[-1]
            indicators["distance_to_resistance"] = ((indicators["resistance"] - current_price) / current_price) * 100
            indicators["distance_to_support"] = ((current_price - indicators["support"]) / current_price) * 100

            # Simple Pattern Detection
            if len(closes) >= 5:
                # Recent trend
                recent_closes = closes.tail(5)
                if all(recent_closes.iloc[i] > recent_closes.iloc[i - 1] for i in range(1, len(recent_closes))):
                    indicators["pattern"] = "ascending"
                elif all(recent_closes.iloc[i] < recent_closes.iloc[i - 1] for i in range(1, len(recent_closes))):
                    indicators["pattern"] = "descending"
                else:
                    indicators["pattern"] = "sideways"
            else:
                indicators["pattern"] = "insufficient_data"

            return indicators
        except Exception as e:
            logger.exception("Failed to identify patterns")
            return {"error": str(e)}

    async def analyze_financial_metrics(self, financial_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze financial metrics and ratios."""
        try:
            if "error" in financial_data:
                return financial_data

            analysis = {
                "profitability_analysis": self._analyze_profitability(financial_data),
                "growth_analysis": self._analyze_growth(financial_data),
                "financial_health_analysis": self._analyze_financial_health(financial_data),
                "valuation_analysis": self._analyze_valuation_metrics(financial_data),
                "timestamp": datetime.now().isoformat(),
            }

            # Overall financial score
            scores = [
                analysis["profitability_analysis"].get("score", 0),
                analysis["growth_analysis"].get("score", 0),
                analysis["financial_health_analysis"].get("score", 0),
            ]
            analysis["overall_financial_score"] = sum(scores) / len(scores)

            logger.info("Financial metrics analysis completed")
            return analysis
        except Exception as e:
            logger.exception("Failed to analyze financial metrics")
            return {"error": str(e)}

    def _analyze_profitability(self, financial_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze profitability metrics."""
        try:
            profitability = financial_data.get("profitability", {})

            metrics = {
                "roe": profitability.get("roe", 0),
                "roa": profitability.get("roa", 0),
                "gross_margin": profitability.get("gross_margin", 0),
                "net_margin": profitability.get("net_margin", 0),
            }

            # Scoring based on industry benchmarks
            scores = {
                "roe": min(100, max(0, (metrics["roe"] - 0.05) * 1000)),  # 15% = 100 points
                "roa": min(100, max(0, (metrics["roa"] - 0.02) * 2000)),  # 7% = 100 points
                "gross_margin": min(100, max(0, (metrics["gross_margin"] - 0.2) * 500)),  # 40% = 100 points
                "net_margin": min(100, max(0, (metrics["net_margin"] - 0.05) * 400)),  # 30% = 100 points
            }

            avg_score = sum(scores.values()) / len(scores)

            # Rating
            if avg_score >= 80:
                rating = "excellent"
            elif avg_score >= 60:
                rating = "good"
            elif avg_score >= 40:
                rating = "average"
            else:
                rating = "poor"

            return {
                "metrics": metrics,
                "scores": scores,
                "score": avg_score,
                "rating": rating,
                "assessment": f"Profitability is {rating} with ROE at {metrics['roe']:.1%} and net margin at {metrics['net_margin']:.1%}",
            }

        except Exception as e:
            logger.exception("Failed to analyze profitability")
            return {"error": str(e), "score": 0}

    def _analyze_growth(self, financial_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze growth metrics."""
        try:
            growth = financial_data.get("growth", {})

            metrics = {
                "revenue_growth": growth.get("revenue_growth", 0),
                "earnings_growth": growth.get("earnings_growth", 0),
                "eps_growth": growth.get("eps_growth", 0),
            }

            # Scoring based on growth rates
            scores = {
                "revenue_growth": min(100, max(0, (metrics["revenue_growth"] + 0.1) * 500)),  # 10% = 100 points
                "earnings_growth": min(100, max(0, (metrics["earnings_growth"] + 0.1) * 500)),  # 10% = 100 points
                "eps_growth": min(100, max(0, (metrics["eps_growth"] + 0.1) * 500)),  # 10% = 100 points
            }

            avg_score = sum(scores.values()) / len(scores)

            # Rating
            if avg_score >= 80:
                rating = "high_growth"
            elif avg_score >= 60:
                rating = "moderate_growth"
            elif avg_score >= 40:
                rating = "low_growth"
            else:
                rating = "no_growth"

            return {
                "metrics": metrics,
                "scores": scores,
                "score": avg_score,
                "rating": rating,
                "assessment": f"Growth is {rating} with revenue growth at {metrics['revenue_growth']:.1%} and earnings growth at {metrics['earnings_growth']:.1%}",
            }

        except Exception as e:
            logger.exception("Failed to analyze growth")
            return {"error": str(e), "score": 0}

    def _analyze_financial_health(self, financial_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze financial health metrics."""
        try:
            health = financial_data.get("financial_health", {})

            metrics = {
                "current_ratio": health.get("current_ratio", 0),
                "debt_to_equity": health.get("debt_to_equity", 0),
                "interest_coverage": health.get("interest_coverage", 0),
                "free_cash_flow": health.get("free_cash_flow", 0),
            }

            # Scoring based on financial health ratios
            scores = {
                "current_ratio": min(100, max(0, (metrics["current_ratio"] - 1) * 100)),  # 2.0 = 100 points
                "debt_to_equity": min(100, max(0, (1 - metrics["debt_to_equity"]) * 200)),  # 0.5 = 100 points
                "interest_coverage": min(100, max(0, (metrics["interest_coverage"] - 3) * 25)),  # 7x = 100 points
                "free_cash_flow": 100 if metrics["free_cash_flow"] > 0 else 0,  # Positive FCF = 100 points
            }

            avg_score = sum(scores.values()) / len(scores)

            # Rating
            if avg_score >= 80:
                rating = "excellent_health"
            elif avg_score >= 60:
                rating = "good_health"
            elif avg_score >= 40:
                rating = "average_health"
            else:
                rating = "poor_health"

            return {
                "metrics": metrics,
                "scores": scores,
                "score": avg_score,
                "rating": rating,
                "assessment": f"Financial health is {rating} with current ratio at {metrics['current_ratio']:.1f} and debt-to-equity at {metrics['debt_to_equity']:.1f}",
            }

        except Exception as e:
            logger.exception("Failed to analyze financial health")
            return {"error": str(e), "score": 0}

    def _analyze_valuation_metrics(self, financial_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze valuation metrics."""
        try:
            basic_info = financial_data.get("basic_info", {})

            metrics = {
                "pe_ratio": basic_info.get("pe_ratio", 0),
                "pb_ratio": basic_info.get("pb_ratio", 0),
                "dividend_yield": basic_info.get("dividend_yield", 0),
            }

            # Valuation assessment
            pe_assessment = (
                "undervalued"
                if 0 < metrics["pe_ratio"] < 15
                else "overvalued"
                if metrics["pe_ratio"] > 25
                else "fairly_valued"
            )
            pb_assessment = (
                "undervalued"
                if 0 < metrics["pb_ratio"] < 1.5
                else "overvalued"
                if metrics["pb_ratio"] > 3
                else "fairly_valued"
            )

            return {
                "metrics": metrics,
                "pe_assessment": pe_assessment,
                "pb_assessment": pb_assessment,
                "assessment": f"Valuation appears {pe_assessment} based on P/E ratio of {metrics['pe_ratio']:.1f} and {pb_assessment} based on P/B ratio of {metrics['pb_ratio']:.1f}",
            }

        except Exception as e:
            logger.exception("Failed to analyze valuation metrics")
            return {"error": str(e)}

    async def calculate_valuation_metrics(
        self, financial_data: dict[str, Any], market_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate comprehensive valuation metrics."""
        try:
            if "error" in financial_data or "error" in market_data:
                return {"error": "Invalid financial or market data"}

            valuation = {
                "dcf_analysis": await self._calculate_dcf(financial_data),
                "relative_valuation": self._calculate_relative_valuation(financial_data, market_data),
                "asset_based_valuation": self._calculate_asset_based_valuation(financial_data),
                "timestamp": datetime.now().isoformat(),
            }

            # Calculate fair value range
            dcf_value = valuation["dcf_analysis"].get("intrinsic_value", 0)
            relative_value = valuation["relative_valuation"].get("fair_value", 0)
            asset_value = valuation["asset_based_valuation"].get("asset_value", 0)

            values = [v for v in [dcf_value, relative_value, asset_value] if v > 0]
            if values:
                valuation["fair_value_range"] = {
                    "min": min(values),
                    "max": max(values),
                    "average": sum(values) / len(values),
                }
            else:
                valuation["fair_value_range"] = {"error": "Unable to calculate fair values"}

            logger.info("Valuation metrics calculated successfully")
            return valuation
        except Exception as e:
            logger.exception("Failed to calculate valuation metrics")
            return {"error": str(e)}

    async def _calculate_dcf(self, financial_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate Discounted Cash Flow valuation."""
        try:
            # Simplified DCF calculation
            profitability = financial_data.get("profitability", {})

            # Extract metrics
            # revenue is available but not used in this calculation
            net_income = profitability.get("net_income", 0)
            fcf = financial_data.get("financial_health", {}).get("free_cash_flow", 0)

            if fcf <= 0:
                fcf = net_income * 0.8  # Estimate FCF as 80% of net income

            # Growth assumptions
            growth = financial_data.get("growth", {})
            revenue_growth = growth.get("revenue_growth", 0.05)

            # DCF parameters
            terminal_growth = 0.03  # 3% terminal growth
            discount_rate = 0.10  # 10% discount rate
            projection_years = 5

            # Project future cash flows
            fcf_projections = []
            current_fcf = fcf

            for year in range(1, projection_years + 1):
                # Declining growth rate
                growth_rate = revenue_growth * (0.8 ** (year - 1))
                current_fcf = current_fcf * (1 + growth_rate)
                fcf_projections.append(current_fcf)

            # Calculate present values
            present_values = []
            for i, fcf_proj in enumerate(fcf_projections):
                pv = fcf_proj / ((1 + discount_rate) ** (i + 1))
                present_values.append(pv)

            # Terminal value
            terminal_fcf = fcf_projections[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            terminal_pv = terminal_value / ((1 + discount_rate) ** projection_years)

            # Enterprise value
            enterprise_value = sum(present_values) + terminal_pv

            return {
                "method": "Discounted Cash Flow",
                "fcf_projections": fcf_projections,
                "present_values": present_values,
                "terminal_value": terminal_value,
                "enterprise_value": enterprise_value,
                "intrinsic_value": enterprise_value,  # Simplified - should adjust for debt/cash
                "assumptions": {
                    "terminal_growth": terminal_growth,
                    "discount_rate": discount_rate,
                    "projection_years": projection_years,
                },
            }
        except Exception as e:
            logger.exception("Failed to calculate DCF")
            return {"error": str(e)}

    def _calculate_relative_valuation(
        self, financial_data: dict[str, Any], market_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate relative valuation using multiples."""
        try:
            basic_info = financial_data.get("basic_info", {})

            # Current multiples
            current_pe = basic_info.get("pe_ratio", 0)
            current_pb = basic_info.get("pb_ratio", 0)
            current_ps = basic_info.get("price_to_sales", 0)

            # Industry averages (simplified - should use actual industry data)
            industry_pe = 18.0
            industry_pb = 2.5
            industry_ps = 3.0

            # Calculate fair values based on industry multiples
            earnings = financial_data.get("profitability", {}).get("net_income", 0)
            book_value = market_data.get("market_data", {}).get("market_cap", 0) / current_pb if current_pb > 0 else 0
            revenue = financial_data.get("profitability", {}).get("revenue", 0)

            fair_values = {}
            if earnings > 0 and industry_pe > 0:
                fair_values["pe_based"] = earnings * industry_pe

            if book_value > 0 and industry_pb > 0:
                fair_values["pb_based"] = book_value * industry_pb

            if revenue > 0 and industry_ps > 0:
                fair_values["ps_based"] = revenue * industry_ps

            # Average fair value
            valid_values = [v for v in fair_values.values() if v > 0]
            fair_value = sum(valid_values) / len(valid_values) if valid_values else 0

            return {
                "method": "Relative Valuation",
                "current_multiples": {"pe": current_pe, "pb": current_pb, "ps": current_ps},
                "industry_multiples": {"pe": industry_pe, "pb": industry_pb, "ps": industry_ps},
                "fair_values": fair_values,
                "fair_value": fair_value,
            }
        except Exception as e:
            logger.exception("Failed to calculate relative valuation")
            return {"error": str(e)}

    def _calculate_asset_based_valuation(self, financial_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate asset-based valuation."""
        try:
            # Simplified asset-based valuation
            # In practice, this would use balance sheet data

            basic_info = financial_data.get("basic_info", {})
            market_cap = basic_info.get("market_cap", 0)

            # Book value estimation (simplified)
            book_value_multiple = 0.8  # Conservative estimate
            asset_value = market_cap * book_value_multiple

            return {
                "method": "Asset-Based Valuation",
                "market_cap": market_cap,
                "book_value_multiple": book_value_multiple,
                "asset_value": asset_value,
                "note": "Simplified calculation - should use actual book value from balance sheet",
            }
        except Exception as e:
            logger.exception("Failed to calculate asset-based valuation")
            return {"error": str(e)}
