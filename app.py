from flask import Flask, render_template, request
from iron_condor_app import IronCondorFinder
import pandas as pd

app = Flask(__name__)

# Available stocks (same as in iron_condor_app.py)
AVAILABLE_STOCKS = ['SPY', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'GOOGL', 'NFLX', 'PLTR', 'MSFT', 'C', 'GS', 'META']

@app.route('/', methods=['GET', 'POST'])
def index():
    # Default parameters
    config = {
        'portfolio_value': 180000,
        'min_pop': 80,
        'target_premium': 7500,
        'min_days_to_expiry': 5,
        'max_days_to_expiry': 200,
        'stocks': AVAILABLE_STOCKS,  # Default to all stocks
        'output_file': None
    }
    
    recommendations = []
    if request.method == 'POST':
        # Update config with user inputs
        config['portfolio_value'] = float(request.form['portfolio_value'])
        config['min_pop'] = float(request.form['min_pop'])
        config['target_premium'] = float(request.form['target_premium'])
        config['min_days_to_expiry'] = int(request.form['min_days'])
        config['max_days_to_expiry'] = int(request.form['max_days'])
        
        # Get selected stocks
        selected_stocks = request.form.getlist('stocks')
        
        # Get custom tickers and append to selected stocks
        custom_tickers = request.form.get('custom_tickers', '').strip()
        if custom_tickers:
            custom_ticker_list = [ticker.strip().upper() for ticker in custom_tickers.split(',') if ticker.strip()]
            selected_stocks.extend(custom_ticker_list)
            # Remove duplicates while preserving order
            selected_stocks = list(dict.fromkeys(selected_stocks))
        
        config['stocks'] = selected_stocks if selected_stocks else AVAILABLE_STOCKS
        
        # Run the IronCondorFinder
        finder = IronCondorFinder(config)
        recommendations = finder.recommend_strategy()
    
    # Prepare results for display
    if recommendations:
        df = pd.DataFrame(recommendations)
        display_df = df[['ticker', 'expiration_date', 'current_price', 'call_sell_strike', 'call_buy_strike',
                         'put_sell_strike', 'put_buy_strike', 'contracts', 'total_premium_all',
                         'max_loss', 'pop', 'weekly_premium']]
        display_df = display_df.round({
            'current_price': 2, 'call_sell_strike': 2, 'call_buy_strike': 2,
            'put_sell_strike': 2, 'put_buy_strike': 2, 'total_premium_all': 2,
            'max_loss': 2, 'pop': 2, 'weekly_premium': 2
        })
        # Convert DataFrame to HTML table with custom classes for conditional formatting
        table_html = display_df.to_html(index=False, classes='table table-striped table-bordered', table_id='results-table')
    else:
        table_html = None
    
    return render_template('index.html', table_html=table_html, config=config, available_stocks=AVAILABLE_STOCKS)

if __name__ == '__main__':
    app.run(debug=True)