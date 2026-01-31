---
name: crypto-trading
description: Cryptocurrency trading expert specializing in automated trading systems, strategy implementation, risk management, and exchange API integration.
---

# Cryptocurrency Trading Systems

Expert guidance for building automated cryptocurrency trading systems, implementing trading strategies, managing risk, and integrating with exchange APIs.

## When to Activate

- Building automated trading bots
- Implementing trading strategies (momentum, mean reversion, arbitrage, etc.)
- Integrating with cryptocurrency exchange APIs
- Designing risk management systems
- Backtesting and optimizing trading strategies
- Real-time market data processing
- Portfolio management and rebalancing

## Core Trading Concepts

### Order Types and Execution

```go
type OrderType string

const (
    OrderTypeMarket     OrderType = "MARKET"
    OrderTypeLimit      OrderType = "LIMIT"
    OrderTypeStopLoss   OrderType = "STOP_LOSS"
    OrderTypeStopLimit  OrderType = "STOP_LIMIT"
    OrderTypeTakeProfit OrderType = "TAKE_PROFIT"
)

type OrderSide string

const (
    OrderSideBuy  OrderSide = "BUY"
    OrderSideSell OrderSide = "SELL"
)

type Order struct {
    ID            string
    Symbol        string
    Side          OrderSide
    Type          OrderType
    Quantity      decimal.Decimal
    Price         decimal.Decimal  // For limit orders
    StopPrice     decimal.Decimal  // For stop orders
    TimeInForce   string           // GTC, IOC, FOK
    Status        OrderStatus
    FilledQty     decimal.Decimal
    AvgFillPrice  decimal.Decimal
    CreatedAt     time.Time
    UpdatedAt     time.Time
}
```

### OHLCV Candlestick Data

```go
type Candle struct {
    Symbol    string
    Interval  string          // 1m, 5m, 15m, 1h, 4h, 1d
    OpenTime  time.Time
    CloseTime time.Time
    Open      decimal.Decimal
    High      decimal.Decimal
    Low       decimal.Decimal
    Close     decimal.Decimal
    Volume    decimal.Decimal
    Trades    int64
}

type CandleStore interface {
    GetCandles(ctx context.Context, symbol, interval string, start, end time.Time) ([]Candle, error)
    StreamCandles(ctx context.Context, symbol, interval string) (<-chan Candle, error)
    SaveCandle(ctx context.Context, candle Candle) error
}
```

## Exchange Integration Patterns

### Exchange Client Interface

```go
type Exchange interface {
    // Account
    GetBalance(ctx context.Context) (map[string]Balance, error)
    GetPositions(ctx context.Context) ([]Position, error)

    // Market Data
    GetTicker(ctx context.Context, symbol string) (*Ticker, error)
    GetOrderBook(ctx context.Context, symbol string, depth int) (*OrderBook, error)
    GetCandles(ctx context.Context, symbol, interval string, limit int) ([]Candle, error)

    // Trading
    PlaceOrder(ctx context.Context, req OrderRequest) (*Order, error)
    CancelOrder(ctx context.Context, symbol, orderID string) error
    GetOrder(ctx context.Context, symbol, orderID string) (*Order, error)
    GetOpenOrders(ctx context.Context, symbol string) ([]Order, error)

    // WebSocket Streams
    SubscribeTrades(ctx context.Context, symbol string) (<-chan Trade, error)
    SubscribeOrderBook(ctx context.Context, symbol string) (<-chan OrderBookUpdate, error)
    SubscribeUserData(ctx context.Context) (<-chan UserDataEvent, error)
}

type Balance struct {
    Asset     string
    Free      decimal.Decimal
    Locked    decimal.Decimal
    UpdatedAt time.Time
}
```

### Rate Limiting and Retry Logic

```go
type RateLimiter struct {
    requestsPerSecond int
    requestsPerMinute int
    orderWeight       int
    tokens            chan struct{}
    mu                sync.Mutex
}

func NewRateLimiter(rps, rpm int) *RateLimiter {
    rl := &RateLimiter{
        requestsPerSecond: rps,
        requestsPerMinute: rpm,
        tokens:            make(chan struct{}, rps),
    }
    go rl.refill()
    return rl
}

func (rl *RateLimiter) Wait(ctx context.Context) error {
    select {
    case <-rl.tokens:
        return nil
    case <-ctx.Done():
        return ctx.Err()
    }
}

// Retry with exponential backoff for transient errors
func RetryWithBackoff(ctx context.Context, fn func() error, maxRetries int) error {
    var lastErr error
    for i := 0; i < maxRetries; i++ {
        if err := fn(); err != nil {
            if !isRetryable(err) {
                return err
            }
            lastErr = err
            backoff := time.Duration(1<<uint(i)) * 100 * time.Millisecond
            select {
            case <-time.After(backoff):
            case <-ctx.Done():
                return ctx.Err()
            }
            continue
        }
        return nil
    }
    return fmt.Errorf("max retries exceeded: %w", lastErr)
}

func isRetryable(err error) bool {
    // Rate limits, temporary network issues, etc.
    var apiErr *APIError
    if errors.As(err, &apiErr) {
        return apiErr.Code == 429 || apiErr.Code >= 500
    }
    return false
}
```

### WebSocket Connection Management

```go
type WSClient struct {
    url           string
    conn          *websocket.Conn
    mu            sync.RWMutex
    subscriptions map[string]chan json.RawMessage
    done          chan struct{}
    reconnectWait time.Duration
}

func (ws *WSClient) Connect(ctx context.Context) error {
    ws.mu.Lock()
    defer ws.mu.Unlock()

    conn, _, err := websocket.DefaultDialer.DialContext(ctx, ws.url, nil)
    if err != nil {
        return fmt.Errorf("websocket dial: %w", err)
    }
    ws.conn = conn
    ws.done = make(chan struct{})

    go ws.readLoop()
    go ws.pingLoop()

    return nil
}

func (ws *WSClient) readLoop() {
    defer close(ws.done)
    for {
        _, msg, err := ws.conn.ReadMessage()
        if err != nil {
            log.Printf("websocket read error: %v", err)
            return
        }
        ws.dispatch(msg)
    }
}

func (ws *WSClient) Subscribe(stream string) (<-chan json.RawMessage, error) {
    ws.mu.Lock()
    defer ws.mu.Unlock()

    ch := make(chan json.RawMessage, 100)
    ws.subscriptions[stream] = ch

    // Send subscription message
    sub := map[string]interface{}{
        "method": "SUBSCRIBE",
        "params": []string{stream},
        "id":     time.Now().UnixNano(),
    }
    return ch, ws.conn.WriteJSON(sub)
}
```

## Trading Strategy Framework

### Strategy Interface

```go
type Signal int

const (
    SignalNone Signal = iota
    SignalBuy
    SignalSell
    SignalCloseLong
    SignalCloseShort
)

type Strategy interface {
    Name() string
    Init(ctx context.Context) error
    OnCandle(candle Candle) Signal
    OnTrade(trade Trade) Signal
    OnOrderUpdate(order Order)
    GetParameters() map[string]interface{}
    SetParameters(params map[string]interface{}) error
}

type StrategyEngine struct {
    exchange   Exchange
    strategies []Strategy
    positions  map[string]*Position
    risk       RiskManager
    logger     *slog.Logger
}

func (e *StrategyEngine) Run(ctx context.Context, symbol string) error {
    candles, err := e.exchange.SubscribeCandles(ctx, symbol, "1m")
    if err != nil {
        return fmt.Errorf("subscribe candles: %w", err)
    }

    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case candle := <-candles:
            for _, strategy := range e.strategies {
                signal := strategy.OnCandle(candle)
                if err := e.processSignal(ctx, strategy, symbol, signal); err != nil {
                    e.logger.Error("process signal failed",
                        "strategy", strategy.Name(),
                        "error", err)
                }
            }
        }
    }
}
```

### Moving Average Crossover Strategy

```go
type MACrossover struct {
    fastPeriod int
    slowPeriod int
    fastMA     *EMA
    slowMA     *EMA
    prevFast   decimal.Decimal
    prevSlow   decimal.Decimal
}

func NewMACrossover(fastPeriod, slowPeriod int) *MACrossover {
    return &MACrossover{
        fastPeriod: fastPeriod,
        slowPeriod: slowPeriod,
        fastMA:     NewEMA(fastPeriod),
        slowMA:     NewEMA(slowPeriod),
    }
}

func (s *MACrossover) OnCandle(candle Candle) Signal {
    fastVal := s.fastMA.Update(candle.Close)
    slowVal := s.slowMA.Update(candle.Close)

    defer func() {
        s.prevFast = fastVal
        s.prevSlow = slowVal
    }()

    // Need history before generating signals
    if s.prevFast.IsZero() || s.prevSlow.IsZero() {
        return SignalNone
    }

    // Golden cross: fast crosses above slow
    if s.prevFast.LessThan(s.prevSlow) && fastVal.GreaterThan(slowVal) {
        return SignalBuy
    }

    // Death cross: fast crosses below slow
    if s.prevFast.GreaterThan(s.prevSlow) && fastVal.LessThan(slowVal) {
        return SignalSell
    }

    return SignalNone
}
```

### RSI Strategy

```go
type RSIStrategy struct {
    period        int
    overbought    decimal.Decimal
    oversold      decimal.Decimal
    rsi           *RSI
    prevRSI       decimal.Decimal
    inPosition    bool
}

func NewRSIStrategy(period int, overbought, oversold float64) *RSIStrategy {
    return &RSIStrategy{
        period:     period,
        overbought: decimal.NewFromFloat(overbought),
        oversold:   decimal.NewFromFloat(oversold),
        rsi:        NewRSI(period),
    }
}

func (s *RSIStrategy) OnCandle(candle Candle) Signal {
    rsiVal := s.rsi.Update(candle.Close)
    defer func() { s.prevRSI = rsiVal }()

    if s.prevRSI.IsZero() {
        return SignalNone
    }

    // Oversold -> crossing above = buy signal
    if s.prevRSI.LessThan(s.oversold) && rsiVal.GreaterThanOrEqual(s.oversold) {
        return SignalBuy
    }

    // Overbought -> crossing below = sell signal
    if s.prevRSI.GreaterThan(s.overbought) && rsiVal.LessThanOrEqual(s.overbought) {
        return SignalSell
    }

    return SignalNone
}
```

### Grid Trading Strategy

```go
type GridStrategy struct {
    symbol      string
    upperPrice  decimal.Decimal
    lowerPrice  decimal.Decimal
    gridCount   int
    gridLevels  []GridLevel
    totalAmount decimal.Decimal
}

type GridLevel struct {
    Price    decimal.Decimal
    BuyOrder *Order
    SellOrder *Order
    Filled   bool
}

func (s *GridStrategy) Init(ctx context.Context) error {
    step := s.upperPrice.Sub(s.lowerPrice).Div(decimal.NewFromInt(int64(s.gridCount)))
    amountPerGrid := s.totalAmount.Div(decimal.NewFromInt(int64(s.gridCount)))

    s.gridLevels = make([]GridLevel, s.gridCount)
    for i := 0; i < s.gridCount; i++ {
        price := s.lowerPrice.Add(step.Mul(decimal.NewFromInt(int64(i))))
        s.gridLevels[i] = GridLevel{
            Price: price,
        }
    }
    return nil
}

func (s *GridStrategy) PlaceGridOrders(ctx context.Context, exchange Exchange, currentPrice decimal.Decimal) error {
    for i := range s.gridLevels {
        level := &s.gridLevels[i]
        if level.Price.LessThan(currentPrice) && level.BuyOrder == nil {
            // Place buy order below current price
            order, err := exchange.PlaceOrder(ctx, OrderRequest{
                Symbol:   s.symbol,
                Side:     OrderSideBuy,
                Type:     OrderTypeLimit,
                Price:    level.Price,
                Quantity: s.totalAmount.Div(decimal.NewFromInt(int64(s.gridCount))),
            })
            if err != nil {
                return err
            }
            level.BuyOrder = order
        }
    }
    return nil
}
```

## Technical Indicators

### Exponential Moving Average (EMA)

```go
type EMA struct {
    period     int
    multiplier decimal.Decimal
    value      decimal.Decimal
    count      int
}

func NewEMA(period int) *EMA {
    mult := decimal.NewFromFloat(2.0 / float64(period+1))
    return &EMA{
        period:     period,
        multiplier: mult,
    }
}

func (e *EMA) Update(price decimal.Decimal) decimal.Decimal {
    e.count++
    if e.count == 1 {
        e.value = price
        return e.value
    }

    // EMA = Price * multiplier + EMA(prev) * (1 - multiplier)
    e.value = price.Mul(e.multiplier).Add(
        e.value.Mul(decimal.NewFromInt(1).Sub(e.multiplier)),
    )
    return e.value
}
```

### Relative Strength Index (RSI)

```go
type RSI struct {
    period   int
    avgGain  decimal.Decimal
    avgLoss  decimal.Decimal
    prevClose decimal.Decimal
    count    int
}

func NewRSI(period int) *RSI {
    return &RSI{period: period}
}

func (r *RSI) Update(close decimal.Decimal) decimal.Decimal {
    r.count++

    if r.count == 1 {
        r.prevClose = close
        return decimal.Zero
    }

    change := close.Sub(r.prevClose)
    r.prevClose = close

    gain := decimal.Zero
    loss := decimal.Zero
    if change.IsPositive() {
        gain = change
    } else {
        loss = change.Abs()
    }

    if r.count <= r.period+1 {
        // Initial SMA calculation
        r.avgGain = r.avgGain.Add(gain)
        r.avgLoss = r.avgLoss.Add(loss)
        if r.count == r.period+1 {
            r.avgGain = r.avgGain.Div(decimal.NewFromInt(int64(r.period)))
            r.avgLoss = r.avgLoss.Div(decimal.NewFromInt(int64(r.period)))
        }
        return decimal.Zero
    }

    // Smoothed moving average
    period := decimal.NewFromInt(int64(r.period))
    r.avgGain = r.avgGain.Mul(period.Sub(decimal.NewFromInt(1))).Add(gain).Div(period)
    r.avgLoss = r.avgLoss.Mul(period.Sub(decimal.NewFromInt(1))).Add(loss).Div(period)

    if r.avgLoss.IsZero() {
        return decimal.NewFromInt(100)
    }

    rs := r.avgGain.Div(r.avgLoss)
    rsi := decimal.NewFromInt(100).Sub(
        decimal.NewFromInt(100).Div(decimal.NewFromInt(1).Add(rs)),
    )
    return rsi
}
```

### Bollinger Bands

```go
type BollingerBands struct {
    period    int
    stdDevMul decimal.Decimal
    prices    []decimal.Decimal
    sma       decimal.Decimal
}

type BBResult struct {
    Upper  decimal.Decimal
    Middle decimal.Decimal
    Lower  decimal.Decimal
}

func NewBollingerBands(period int, stdDevMul float64) *BollingerBands {
    return &BollingerBands{
        period:    period,
        stdDevMul: decimal.NewFromFloat(stdDevMul),
        prices:    make([]decimal.Decimal, 0, period),
    }
}

func (bb *BollingerBands) Update(price decimal.Decimal) BBResult {
    bb.prices = append(bb.prices, price)
    if len(bb.prices) > bb.period {
        bb.prices = bb.prices[1:]
    }

    if len(bb.prices) < bb.period {
        return BBResult{}
    }

    // Calculate SMA
    sum := decimal.Zero
    for _, p := range bb.prices {
        sum = sum.Add(p)
    }
    bb.sma = sum.Div(decimal.NewFromInt(int64(bb.period)))

    // Calculate standard deviation
    variance := decimal.Zero
    for _, p := range bb.prices {
        diff := p.Sub(bb.sma)
        variance = variance.Add(diff.Mul(diff))
    }
    variance = variance.Div(decimal.NewFromInt(int64(bb.period)))
    stdDev := decimal.NewFromFloat(math.Sqrt(variance.InexactFloat64()))

    band := stdDev.Mul(bb.stdDevMul)
    return BBResult{
        Upper:  bb.sma.Add(band),
        Middle: bb.sma,
        Lower:  bb.sma.Sub(band),
    }
}
```

## Risk Management

### Position Sizing

```go
type RiskManager struct {
    maxPositionSize   decimal.Decimal // Max % of portfolio per position
    maxDrawdown       decimal.Decimal // Max allowed drawdown
    riskPerTrade      decimal.Decimal // Risk % per trade
    maxOpenPositions  int
    currentPositions  int
    portfolioValue    decimal.Decimal
    peakValue         decimal.Decimal
}

func (rm *RiskManager) CalculatePositionSize(
    entryPrice, stopLoss, portfolioValue decimal.Decimal,
) decimal.Decimal {
    // Risk amount = portfolio * risk per trade
    riskAmount := portfolioValue.Mul(rm.riskPerTrade)

    // Distance to stop loss
    stopDistance := entryPrice.Sub(stopLoss).Abs()
    if stopDistance.IsZero() {
        return decimal.Zero
    }

    // Position size = risk amount / stop distance
    positionSize := riskAmount.Div(stopDistance)

    // Apply max position size limit
    maxSize := portfolioValue.Mul(rm.maxPositionSize).Div(entryPrice)
    if positionSize.GreaterThan(maxSize) {
        positionSize = maxSize
    }

    return positionSize
}

func (rm *RiskManager) CheckDrawdown(currentValue decimal.Decimal) bool {
    if currentValue.GreaterThan(rm.peakValue) {
        rm.peakValue = currentValue
    }

    drawdown := rm.peakValue.Sub(currentValue).Div(rm.peakValue)
    return drawdown.LessThan(rm.maxDrawdown)
}
```

### Stop Loss and Take Profit

```go
type PositionManager struct {
    exchange Exchange
    logger   *slog.Logger
}

type StopConfig struct {
    StopLossPercent   decimal.Decimal
    TakeProfitPercent decimal.Decimal
    TrailingStop      bool
    TrailingPercent   decimal.Decimal
}

func (pm *PositionManager) SetStopLoss(
    ctx context.Context,
    position *Position,
    config StopConfig,
) error {
    stopPrice := position.EntryPrice.Mul(
        decimal.NewFromInt(1).Sub(config.StopLossPercent),
    )

    _, err := pm.exchange.PlaceOrder(ctx, OrderRequest{
        Symbol:    position.Symbol,
        Side:      OrderSideSell,
        Type:      OrderTypeStopLoss,
        Quantity:  position.Quantity,
        StopPrice: stopPrice,
    })
    return err
}

func (pm *PositionManager) ManageTrailingStop(
    ctx context.Context,
    position *Position,
    currentPrice decimal.Decimal,
    trailingPercent decimal.Decimal,
) error {
    newStopPrice := currentPrice.Mul(
        decimal.NewFromInt(1).Sub(trailingPercent),
    )

    // Only update if new stop is higher than current
    if newStopPrice.GreaterThan(position.StopLoss) {
        return pm.updateStopOrder(ctx, position, newStopPrice)
    }
    return nil
}
```

### Portfolio Risk Metrics

```go
type PortfolioMetrics struct {
    TotalValue        decimal.Decimal
    DailyPnL          decimal.Decimal
    TotalPnL          decimal.Decimal
    WinRate           decimal.Decimal
    SharpeRatio       decimal.Decimal
    MaxDrawdown       decimal.Decimal
    CurrentDrawdown   decimal.Decimal
    OpenPositions     int
    TotalTrades       int
}

func CalculateSharpeRatio(returns []decimal.Decimal, riskFreeRate decimal.Decimal) decimal.Decimal {
    if len(returns) < 2 {
        return decimal.Zero
    }

    // Calculate mean return
    sum := decimal.Zero
    for _, r := range returns {
        sum = sum.Add(r)
    }
    mean := sum.Div(decimal.NewFromInt(int64(len(returns))))

    // Calculate standard deviation
    variance := decimal.Zero
    for _, r := range returns {
        diff := r.Sub(mean)
        variance = variance.Add(diff.Mul(diff))
    }
    variance = variance.Div(decimal.NewFromInt(int64(len(returns) - 1)))
    stdDev := decimal.NewFromFloat(math.Sqrt(variance.InexactFloat64()))

    if stdDev.IsZero() {
        return decimal.Zero
    }

    // Sharpe = (mean - risk free) / std dev
    // Annualized: multiply by sqrt(252) for daily returns
    sharpe := mean.Sub(riskFreeRate).Div(stdDev)
    return sharpe.Mul(decimal.NewFromFloat(math.Sqrt(252)))
}
```

## Backtesting Framework

### Backtester Engine

```go
type Backtester struct {
    strategy      Strategy
    initialCap    decimal.Decimal
    commission    decimal.Decimal // Per trade commission rate
    slippage      decimal.Decimal // Simulated slippage
    trades        []Trade
    equity        []EquityPoint
}

type BacktestResult struct {
    InitialCapital   decimal.Decimal
    FinalCapital     decimal.Decimal
    TotalReturn      decimal.Decimal
    MaxDrawdown      decimal.Decimal
    WinRate          decimal.Decimal
    ProfitFactor     decimal.Decimal
    SharpeRatio      decimal.Decimal
    TotalTrades      int
    WinningTrades    int
    LosingTrades     int
    AvgWin           decimal.Decimal
    AvgLoss          decimal.Decimal
    Trades           []Trade
    EquityCurve      []EquityPoint
}

func (bt *Backtester) Run(candles []Candle) BacktestResult {
    capital := bt.initialCap
    position := decimal.Zero
    entryPrice := decimal.Zero

    for _, candle := range candles {
        signal := bt.strategy.OnCandle(candle)

        switch signal {
        case SignalBuy:
            if position.IsZero() {
                // Apply slippage to entry
                price := candle.Close.Mul(decimal.NewFromInt(1).Add(bt.slippage))
                position = capital.Div(price)
                commission := capital.Mul(bt.commission)
                capital = capital.Sub(commission)
                entryPrice = price
            }

        case SignalSell:
            if position.GreaterThan(decimal.Zero) {
                // Apply slippage to exit
                price := candle.Close.Mul(decimal.NewFromInt(1).Sub(bt.slippage))
                proceeds := position.Mul(price)
                commission := proceeds.Mul(bt.commission)
                capital = proceeds.Sub(commission)

                bt.trades = append(bt.trades, Trade{
                    EntryPrice: entryPrice,
                    ExitPrice:  price,
                    Quantity:   position,
                    PnL:        proceeds.Sub(position.Mul(entryPrice)),
                    Timestamp:  candle.CloseTime,
                })
                position = decimal.Zero
            }
        }

        bt.equity = append(bt.equity, EquityPoint{
            Time:  candle.CloseTime,
            Value: capital.Add(position.Mul(candle.Close)),
        })
    }

    return bt.calculateResults()
}
```

### Walk-Forward Optimization

```go
type WalkForwardOptimizer struct {
    strategy        Strategy
    parameterRanges map[string][]interface{}
    inSampleRatio   float64 // e.g., 0.7 for 70% in-sample
    windows         int
}

func (wfo *WalkForwardOptimizer) Optimize(candles []Candle) []OptimizationResult {
    windowSize := len(candles) / wfo.windows
    results := make([]OptimizationResult, 0, wfo.windows)

    for i := 0; i < wfo.windows; i++ {
        start := i * windowSize
        end := start + windowSize
        if end > len(candles) {
            end = len(candles)
        }

        window := candles[start:end]
        inSampleEnd := int(float64(len(window)) * wfo.inSampleRatio)

        // Optimize on in-sample data
        inSample := window[:inSampleEnd]
        bestParams := wfo.findBestParameters(inSample)

        // Test on out-of-sample data
        outOfSample := window[inSampleEnd:]
        wfo.strategy.SetParameters(bestParams)
        bt := &Backtester{strategy: wfo.strategy}
        result := bt.Run(outOfSample)

        results = append(results, OptimizationResult{
            Window:     i,
            Parameters: bestParams,
            InSample:   wfo.evaluateParameters(inSample, bestParams),
            OutOfSample: result,
        })
    }

    return results
}
```

## Market Data Management

### Historical Data Fetcher

```go
type DataFetcher struct {
    exchange Exchange
    store    CandleStore
    logger   *slog.Logger
}

func (df *DataFetcher) FetchHistorical(
    ctx context.Context,
    symbol, interval string,
    start, end time.Time,
) error {
    current := start
    batchSize := 1000 // Most exchanges limit to 1000 candles per request

    for current.Before(end) {
        candles, err := df.exchange.GetCandles(ctx, symbol, interval, batchSize)
        if err != nil {
            return fmt.Errorf("fetch candles: %w", err)
        }

        if len(candles) == 0 {
            break
        }

        for _, c := range candles {
            if err := df.store.SaveCandle(ctx, c); err != nil {
                return fmt.Errorf("save candle: %w", err)
            }
        }

        current = candles[len(candles)-1].CloseTime
        df.logger.Info("fetched candles",
            "symbol", symbol,
            "count", len(candles),
            "until", current)

        // Respect rate limits
        time.Sleep(100 * time.Millisecond)
    }

    return nil
}
```

### Real-time Data Aggregator

```go
type CandleAggregator struct {
    interval time.Duration
    current  *Candle
    output   chan Candle
    mu       sync.Mutex
}

func (ca *CandleAggregator) ProcessTrade(trade Trade) {
    ca.mu.Lock()
    defer ca.mu.Unlock()

    candleStart := trade.Timestamp.Truncate(ca.interval)

    if ca.current == nil || candleStart.After(ca.current.OpenTime) {
        // Emit completed candle
        if ca.current != nil {
            ca.output <- *ca.current
        }

        // Start new candle
        ca.current = &Candle{
            Symbol:    trade.Symbol,
            OpenTime:  candleStart,
            CloseTime: candleStart.Add(ca.interval),
            Open:      trade.Price,
            High:      trade.Price,
            Low:       trade.Price,
            Close:     trade.Price,
            Volume:    trade.Quantity,
            Trades:    1,
        }
        return
    }

    // Update current candle
    if trade.Price.GreaterThan(ca.current.High) {
        ca.current.High = trade.Price
    }
    if trade.Price.LessThan(ca.current.Low) {
        ca.current.Low = trade.Price
    }
    ca.current.Close = trade.Price
    ca.current.Volume = ca.current.Volume.Add(trade.Quantity)
    ca.current.Trades++
}
```

## Best Practices

### Decimal Precision

Always use `decimal.Decimal` (shopspring/decimal) for financial calculations:

```go
// Bad: Float precision issues
price := 0.1 + 0.2 // = 0.30000000000000004

// Good: Exact decimal arithmetic
price := decimal.NewFromFloat(0.1).Add(decimal.NewFromFloat(0.2))
```

### Order Execution Safety

```go
// Always verify order status before assuming execution
func (e *Executor) PlaceAndVerify(ctx context.Context, req OrderRequest) (*Order, error) {
    order, err := e.exchange.PlaceOrder(ctx, req)
    if err != nil {
        return nil, err
    }

    // Poll for confirmation
    ticker := time.NewTicker(500 * time.Millisecond)
    defer ticker.Stop()

    timeout := time.After(30 * time.Second)
    for {
        select {
        case <-ticker.C:
            current, err := e.exchange.GetOrder(ctx, order.Symbol, order.ID)
            if err != nil {
                continue
            }
            if current.Status == OrderStatusFilled ||
               current.Status == OrderStatusCanceled ||
               current.Status == OrderStatusRejected {
                return current, nil
            }
        case <-timeout:
            return nil, fmt.Errorf("order confirmation timeout")
        case <-ctx.Done():
            return nil, ctx.Err()
        }
    }
}
```

### Logging and Monitoring

```go
type TradeLogger struct {
    logger *slog.Logger
}

func (tl *TradeLogger) LogOrder(order *Order) {
    tl.logger.Info("order executed",
        slog.String("id", order.ID),
        slog.String("symbol", order.Symbol),
        slog.String("side", string(order.Side)),
        slog.String("type", string(order.Type)),
        slog.String("quantity", order.Quantity.String()),
        slog.String("price", order.Price.String()),
        slog.String("status", string(order.Status)),
        slog.Time("timestamp", order.UpdatedAt),
    )
}

func (tl *TradeLogger) LogPnL(position *Position, exitPrice decimal.Decimal) {
    pnl := exitPrice.Sub(position.EntryPrice).Mul(position.Quantity)
    pnlPercent := exitPrice.Sub(position.EntryPrice).Div(position.EntryPrice).Mul(decimal.NewFromInt(100))

    tl.logger.Info("position closed",
        slog.String("symbol", position.Symbol),
        slog.String("pnl", pnl.String()),
        slog.String("pnl_percent", pnlPercent.StringFixed(2)+"%"),
        slog.String("entry", position.EntryPrice.String()),
        slog.String("exit", exitPrice.String()),
    )
}
```

## Quick Reference: Trading Terminology

| Term | Description |
|------|-------------|
| OHLCV | Open, High, Low, Close, Volume - candlestick data |
| Slippage | Difference between expected and actual execution price |
| Drawdown | Peak-to-trough decline in portfolio value |
| Sharpe Ratio | Risk-adjusted return metric (higher is better) |
| Win Rate | Percentage of profitable trades |
| Profit Factor | Gross profit / gross loss ratio |
| Maker/Taker | Order book liquidity roles (limit vs market orders) |
| Leverage | Borrowed funds for amplified positions |
| Liquidation | Forced position closure due to insufficient margin |

## Security Considerations

- Never store API keys in code; use environment variables or secret managers
- Use IP whitelisting on exchange API keys when available
- Enable withdrawal address whitelisting
- Implement circuit breakers for abnormal market conditions
- Monitor for exchange API rate limit changes
- Always use HTTPS and verify SSL certificates
- Implement proper error handling to avoid order duplication
