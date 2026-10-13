# Coinbase Spot Scanner — Mobile V2

A responsive, read-only Streamlit app that can be hosted as a private/unlisted URL and opened from an iPhone. It **is not hosted yet**.

## Deploy for iPhone access
1. On a computer, create a private GitHub repository and upload `app.py` and `requirements.txt` (Streamlit Community Cloud availability for private repos depends on your account/plan).
2. In Streamlit Community Cloud, create a new app pointing at `app.py`. Check current hosting terms and visibility settings; if private access is required, choose hosting that supports authentication.
3. Open the generated HTTPS URL in Safari on your iPhone and choose Share → Add to Home Screen.
4. Open the settings panel and replace **placeholder** maker/taker fees with your actual Coinbase Advanced fee tier.

Local testing: `pip install -r requirements.txt && streamlit run app.py`

## What it does
Scans up to 20 public Coinbase Exchange spot markets, uses completed candles, ranks heuristic breakout/sweep/trend signals, shows mobile cards, and computes a fee-adjusted risk/reward estimate. No API keys, account access or execution.

## Important limitations
- Refresh is **manual**; no persistent streaming or notifications.
- Public Coinbase Exchange markets can differ from Coinbase Advanced; verify pairs and prices on your trading ticket.
- Indicators and rankings are **not backtested**; the VWAP is history-window anchored.
- No guarantee of order fills or stop prices. Spread snapshots, slippage, conversion costs and tax may affect profitability.
- GBP-equivalent risk calculation needs actual GBP/quote conversion for USD pairs.
- No authentication is built into the app; use hosting access controls if you need a private dashboard.
