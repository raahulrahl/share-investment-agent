# Portfolio Management Specialist

## Overview
The Portfolio Management Specialist serves as the final decision-maker in the investment analysis process. This specialist synthesizes inputs from all other specialists, applies risk management constraints, and makes final trading decisions while maintaining portfolio objectives and risk parameters.

## Core Responsibilities

### Decision Synthesis
- **Signal Aggregation**: Combine signals from all analysis specialists
- **Weight Assignment**: Allocate appropriate weights to different signals
- **Conflict Resolution**: Handle conflicting signals from different specialists
- **Final Decision**: Make ultimate buy/sell/hold recommendations

### Risk Management
- **Position Sizing**: Determine optimal trade quantities
- **Portfolio Risk**: Manage overall portfolio risk exposure
- **Stop Loss Planning**: Set appropriate risk limits
- **Diversification**: Maintain portfolio diversification standards

### Performance Monitoring
- **Trade Execution**: Implement trading decisions
- **Performance Tracking**: Monitor portfolio performance
- **Risk Metrics**: Track portfolio risk indicators
- **Rebalancing**: Periodic portfolio adjustments

## Decision Framework

### Signal Integration System
```python
# Signal Aggregation Framework
def aggregate_specialist_signals(specialist_recommendations):
    """
    Aggregate signals from all specialists with weighted scoring
    """
    signal_weights = {
        "technical_analysis": 0.25,
        "fundamental_analysis": 0.30,
        "sentiment_analysis": 0.10,
        "valuation_analysis": 0.25,
        "market_data_analysis": 0.10
    }

    weighted_score = 0
    total_weight = 0

    for specialist, recommendation in specialist_recommendations.items():
        weight = signal_weights.get(specialist, 0)
        signal_strength = convert_signal_to_strength(recommendation["signal"])
        confidence = recommendation.get("confidence", 0.5)

        # Weighted signal calculation
        weighted_signal = weight * signal_strength * confidence
        weighted_score += weighted_signal
        total_weight += weight * confidence

    # Normalize score
    final_score = weighted_score / total_weight if total_weight > 0 else 0

    return {
        "final_score": final_score,
        "recommendation": score_to_recommendation(final_score),
        "confidence": calculate_decision_confidence(specialist_recommendations)
    }
```

### Risk Constraint Application
```python
# Risk Management Integration
def apply_risk_constraints(decision, portfolio_state, risk_parameters):
    """
    Apply risk management constraints to trading decisions
    """
    # Position size limits
    max_position_size = risk_parameters["max_position_size"]
    current_exposure = portfolio_state["current_exposure"]

    # Calculate allowable position
    available_capacity = max_position_size - current_exposure
    proposed_size = decision["proposed_quantity"]

    # Apply position size constraint
    if proposed_size > available_capacity:
        adjusted_size = available_capacity
        decision["adjusted"] = True
        decision["adjustment_reason"] = "Position size limit"
    else:
        adjusted_size = proposed_size
        decision["adjusted"] = False

    # Apply portfolio-level risk constraints
    portfolio_risk = calculate_portfolio_risk(portfolio_state, decision)
    if portfolio_risk > risk_parameters["max_portfolio_risk"]:
        adjusted_size *= 0.5  # Reduce position by half
        decision["adjusted"] = True
        decision["adjustment_reason"] = "Portfolio risk limit"

    decision["final_quantity"] = adjusted_size
    return decision
```

## Signal Processing

### Signal Conversion
```python
# Convert specialist signals to standardized format
def convert_signal_to_strength(signal):
    signal_mapping = {
        "strong_buy": 1.0,
        "buy": 0.75,
        "bullish": 0.6,
        "moderate_buy": 0.4,
        "neutral": 0.0,
        "moderate_sell": -0.4,
        "bearish": -0.6,
        "sell": -0.75,
        "strong_sell": -1.0
    }
    return signal_mapping.get(signal.lower(), 0.0)

# Convert final score to recommendation
def score_to_recommendation(score):
    if score >= 0.7:
        return "strong_buy"
    elif score >= 0.4:
        return "buy"
    elif score >= 0.1:
        return "moderate_buy"
    elif score > -0.1:
        return "hold"
    elif score > -0.4:
        return "moderate_sell"
    elif score > -0.7:
        return "sell"
    else:
        return "strong_sell"
```

### Confidence Calculation
```python
# Calculate decision confidence based on specialist agreement
def calculate_decision_confidence(specialist_recommendations):
    signals = [rec["signal"] for rec in specialist_recommendations.values()]
    confidences = [rec.get("confidence", 0.5) for rec in specialist_recommendations.values()]

    # Signal agreement (how similar are the signals)
    signal_strengths = [convert_signal_to_strength(s) for s in signals]
    signal_variance = calculate_variance(signal_strengths)
    agreement_score = 1.0 - min(signal_variance, 1.0)

    # Average confidence
    avg_confidence = sum(confidences) / len(confidences)

    # Combined confidence
    final_confidence = (agreement_score + avg_confidence) / 2

    return final_confidence
```

## Position Management

### Position Sizing Models
```python
# Kelly Criterion Position Sizing
def kelly_position_size(win_rate, avg_win, avg_loss):
    """
    Calculate optimal position size using Kelly Criterion
    """
    if avg_loss == 0:
        return 0

    win_loss_ratio = avg_win / abs(avg_loss)
    kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

    # Apply safety factor (typically 0.25 to 0.5)
    safety_factor = 0.25
    kelly_position = max(0, kelly_fraction * safety_factor)

    return min(kelly_position, 0.25)  # Cap at 25% maximum

# Volatility-Based Position Sizing
def volatility_position_size(price_volatility, portfolio_volatility_target):
    """
    Adjust position size based on volatility
    """
    volatility_ratio = price_volatility / portfolio_volatility_target

    # Inverse relationship - higher volatility = smaller position
    base_position = 0.1  # 10% base position
    adjusted_position = base_position / volatility_ratio

    return min(adjusted_position, 0.2)  # Cap at 20% maximum
```

### Portfolio Construction
```python
# Optimize portfolio allocation
def optimize_portfolio(expected_returns, covariance_matrix, risk_aversion):
    """
    Mean-variance optimization for portfolio allocation
    """
    num_assets = len(expected_returns)

    # Calculate portfolio weights using mean-variance optimization
    # Simplified implementation for demonstration
    inverse_covariance = np.linalg.inv(covariance_matrix)
    ones_vector = np.ones((num_assets, 1))

    # Calculate optimal weights
    numerator = np.dot(inverse_covariance, expected_returns)
    denominator = np.dot(ones_vector.T, np.dot(inverse_covariance, ones_vector))

    optimal_weights = numerator / denominator

    # Apply risk aversion adjustment
    risk_adjusted_weights = optimal_weights * (1 / (1 + risk_aversion))

    # Normalize weights
    normalized_weights = risk_adjusted_weights / sum(risk_adjusted_weights)

    return normalized_weights
```

## Risk Management Integration

### Risk Metrics Calculation
```python
# Portfolio risk metrics
def calculate_portfolio_risk(portfolio_state, new_decision):
    """
    Calculate comprehensive portfolio risk metrics
    """
    # Current positions
    current_positions = portfolio_state["positions"]

    # Add new position
    updated_positions = current_positions.copy()
    if new_decision["action"] in ["buy", "strong_buy"]:
        ticker = new_decision["ticker"]
        quantity = new_decision["final_quantity"]
        updated_positions[ticker] = updated_positions.get(ticker, 0) + quantity

    # Calculate portfolio metrics
    total_value = sum(pos["quantity"] * pos["current_price"] for pos in updated_positions.values())

    # Concentration risk
    position_sizes = [pos["quantity"] * pos["current_price"] / total_value for pos in updated_positions.values()]
    max_concentration = max(position_sizes) if position_sizes else 0
    herfindahl_index = sum(size ** 2 for size in position_sizes)

    # Sector concentration
    sector_exposures = calculate_sector_exposure(updated_positions)
    max_sector_exposure = max(sector_exposures.values()) if sector_exposures else 0

    # Overall risk score
    risk_score = (
        max_concentration * 0.4 +
        herfindahl_index * 0.3 +
        max_sector_exposure * 0.3
    )

    return {
        "total_value": total_value,
        "max_concentration": max_concentration,
        "herfindahl_index": herfindahl_index,
        "max_sector_exposure": max_sector_exposure,
        "overall_risk_score": risk_score
    }
```

### Stop Loss Planning
```python
# Dynamic stop loss calculation
def calculate_stop_loss(current_price, volatility, technical_levels, risk_tolerance):
    """
    Calculate optimal stop loss level
    """
    # Volatility-based stop (2x ATR)
    volatility_stop = current_price - (2 * volatility * current_price)

    # Technical support level
    technical_stop = technical_levels.get("nearest_support", current_price * 0.9)

    # Maximum acceptable loss based on risk tolerance
    max_loss_stop = current_price * (1 - risk_tolerance)

    # Choose the highest (most conservative) stop level
    optimal_stop = max(volatility_stop, technical_stop, max_loss_stop)

    return {
        "stop_loss_price": optimal_stop,
        "stop_loss_percentage": (current_price - optimal_stop) / current_price,
        "method": "volatility" if optimal_stop == volatility_stop else
                "technical" if optimal_stop == technical_stop else
                "risk_limit"
    }
```

## Execution Strategy

### Trade Implementation
```python
# Trade execution planning
def plan_trade_execution(decision, market_conditions):
    """
    Plan optimal trade execution strategy
    """
    trade_plan = {
        "action": decision["action"],
        "quantity": decision["final_quantity"],
        "ticker": decision["ticker"],
        "urgency": "normal"
    }

    # Market condition adjustments
    if market_conditions["volatility"] > 0.3:  # High volatility
        trade_plan["execution_method"] = "scaled"
        trade_plan["time_horizon"] = "intraday"
        trade_plan["urgency"] = "low"
    elif market_conditions["volume"] < market_conditions["avg_volume"] * 0.5:  # Low volume
        trade_plan["execution_method"] = "patient"
        trade_plan["time_horizon"] = "multi_day"
        trade_plan["urgency"] = "low"
    else:  # Normal conditions
        trade_plan["execution_method"] = "immediate"
        trade_plan["time_horizon"] = "immediate"
        trade_plan["urgency"] = "normal"

    return trade_plan
```

### Performance Tracking
```python
# Trade performance analysis
def analyze_trade_performance(trade, market_data):
    """
    Analyze performance of executed trades
    """
    entry_price = trade["entry_price"]
    exit_price = trade.get("exit_price", market_data["current_price"])
    quantity = trade["quantity"]

    # Calculate returns
    if trade["action"] in ["buy", "strong_buy"]:
        profit_loss = (exit_price - entry_price) * quantity
        return_percentage = (exit_price - entry_price) / entry_price
    else:  # Short position
        profit_loss = (entry_price - exit_price) * quantity
        return_percentage = (entry_price - exit_price) / entry_price

    # Risk metrics
    max_adverse_excursion = calculate_max_adverse_excursion(trade, market_data)
    favorable_excursion = calculate_favorable_excursion(trade, market_data)

    return {
        "profit_loss": profit_loss,
        "return_percentage": return_percentage,
        "max_adverse_excursion": max_adverse_excursion,
        "favorable_excursion": favorable_excursion,
        "holding_period": calculate_holding_period(trade),
        "sharpe_ratio": calculate_sharpe_ratio(trade, market_data)
    }
```

## Integration with Specialists

### Specialist Input Processing
```python
# Process inputs from all specialists
def process_specialist_inputs(specialist_data):
    """
    Standardize and validate inputs from all analysis specialists
    """
    processed_inputs = {}

    for specialist_name, data in specialist_data.items():
        # Validate input format
        if not validate_specialist_input(data):
            continue

        # Extract key information
        processed_input = {
            "signal": data.get("signal", "hold"),
            "confidence": data.get("confidence", 0.5),
            "reasoning": data.get("reasoning", ""),
            "key_metrics": data.get("key_metrics", {}),
            "risk_factors": data.get("risk_factors", []),
            "timestamp": data.get("timestamp")
        }

        processed_inputs[specialist_name] = processed_input

    return processed_inputs
```

### Conflict Resolution
```python
# Handle conflicting specialist signals
def resolve_conflicting_signals(processed_inputs):
    """
    Resolve conflicts between different specialist recommendations
    """
    signals = [inp["signal"] for inp in processed_inputs.values()]
    confidences = [inp["confidence"] for inp in processed_inputs.values()]

    # Check for major conflicts
    bullish_signals = sum(1 for s in signals if s in ["buy", "strong_buy", "bullish"])
    bearish_signals = sum(1 for s in signals if s in ["sell", "strong_sell", "bearish"])

    conflict_resolution = {
        "has_conflict": min(bullish_signals, bearish_signals) > 0,
        "conflict_level": min(bullish_signals, bearish_signals) / len(signals),
        "resolution_strategy": "weighted_average"
    }

    # High conflict resolution strategies
    if conflict_resolution["conflict_level"] > 0.4:
        # Use higher confidence weighting
        conflict_resolution["resolution_strategy"] = "confidence_weighted"
    elif conflict_resolution["conflict_level"] > 0.6:
        # Use fundamental analysis as tie-breaker
        conflict_resolution["resolution_strategy"] = "fundamental_priority"

    return conflict_resolution
```

## Reporting and Documentation

### Decision Documentation
```python
# Document portfolio management decisions
def document_decision(decision, specialist_inputs, market_context):
    """
    Create comprehensive decision documentation
    """
    documentation = {
        "decision_timestamp": datetime.now().isoformat(),
        "ticker": decision["ticker"],
        "action": decision["action"],
        "quantity": decision["final_quantity"],
        "rationale": decision["reasoning"],

        # Specialist inputs summary
        "specialist_consensus": {
            "technical_analysis": specialist_inputs.get("technical_analysis", {}),
            "fundamental_analysis": specialist_inputs.get("fundamental_analysis", {}),
            "sentiment_analysis": specialist_inputs.get("sentiment_analysis", {}),
            "valuation_analysis": specialist_inputs.get("valuation_analysis", {})
        },

        # Risk assessment
        "risk_assessment": {
            "position_size_risk": decision.get("position_size_risk"),
            "portfolio_impact": decision.get("portfolio_impact"),
            "stop_loss_plan": decision.get("stop_loss_plan")
        },

        # Market context
        "market_context": {
            "market_regime": market_context.get("regime"),
            "volatility_level": market_context.get("volatility"),
            "liquidity_conditions": market_context.get("liquidity")
        },

        # Performance expectations
        "performance_expectations": {
            "target_return": decision.get("target_return"),
            "time_horizon": decision.get("time_horizon"),
            "risk_tolerance": decision.get("risk_tolerance")
        }
    }

    return documentation
```

## Performance Monitoring

### Portfolio Metrics
```python
# Calculate portfolio performance metrics
def calculate_portfolio_metrics(portfolio_state, benchmark_data):
    """
    Comprehensive portfolio performance analysis
    """
    # Return calculations
    portfolio_return = calculate_portfolio_return(portfolio_state)
    benchmark_return = calculate_benchmark_return(benchmark_data)
    alpha = portfolio_return - benchmark_return

    # Risk metrics
    portfolio_volatility = calculate_portfolio_volatility(portfolio_state)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    max_drawdown = calculate_max_drawdown(portfolio_state)

    # Risk-adjusted returns
    information_ratio = alpha / tracking_error
    treynor_ratio = (portfolio_return - risk_free_rate) / portfolio_beta

    return {
        "total_return": portfolio_return,
        "alpha": alpha,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "information_ratio": information_ratio,
        "treynor_ratio": treynor_ratio,
        "volatility": portfolio_volatility
    }
```

## Best Practices

### Decision Quality
- **Systematic Approach**: Consistent decision-making framework
- **Documentation**: Complete decision audit trail
- **Review Process**: Regular decision quality assessment
- **Continuous Improvement**: Learning from outcomes

### Risk Management
- **Conservative Approach**: Prioritize capital preservation
- **Diversification**: Maintain appropriate diversification
- **Risk Monitoring**: Continuous risk assessment
- **Stress Testing**: Portfolio stress scenarios

### Professional Standards
- **Fiduciary Duty**: Act in client's best interest
- **Transparency**: Clear decision rationale
- **Compliance**: Regulatory requirement adherence
- **Ethical Conduct**: High ethical standards
