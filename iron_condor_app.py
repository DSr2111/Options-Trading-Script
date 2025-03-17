#!/usr/bin/env python3
"""
Iron Condor Recommendation App

This program identifies and recommends iron condor options trades for a given portfolio.
It fetches options data using yfinance, calculates trade metrics (e.g., premium, PoP),
and outputs a sorted table of recommendations. Trades are filtered by total premium
and risk constraints, and sorted by weekly premium.

Usage:
    python iron_condor_app.py [--portfolio_value VALUE] [--min_pop POP] [--target_premium PREMIUM]
                              [--min_days DAYS] [--max_days DAYS] [--output FILE]

Example:
    python iron_condor_app.py --portfolio_value 200000 --min_pop 75 --target_premium 8000
                              --min_days 5 --max_days 200 --output trades.csv
"""
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
from scipy.stats import norm
import logging
import argparse
import concurrent.futures
from functools import lru_cache
import sys
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IronCondorFinder:
    """Class to find and recommend iron condor options trades."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration parameters."""
        self.portfolio_value = config['portfolio_value']
        self.buying_power = self.portfolio_value / 0.3
        self.max_risk_per_trade = self.buying_power * 0.25
        self.target_premium = config['target_premium']
        self.min_pop = config['min_pop']
        self.risk_free_rate = 0.02
        self.stocks = config['stocks']
        self.min_days_to_expiry = config['min_days_to_expiry']
        self.max_days_to_expiry = config['max_days_to_expiry']
        self.output_file = config.get('output_file', None)
        self.today = datetime.today()

    @staticmethod
    @lru_cache(maxsize=1000)
    def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Tuple[float, float, float, float]:
        """Calculate Black-Scholes Greeks for an option."""
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == 'call':
                delta = norm.cdf(d1)
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = (-(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
                vega = S * np.sqrt(T) * norm.pdf(d1) / 100
            else:
                delta = norm.cdf(d1) - 1
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                theta = (-(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
                vega = S * np.sqrt(T) * norm.pdf(d1) / 100
            return delta, gamma, theta, vega
        except Exception as e:
            logger.error(f"Black-Scholes error: S={S}, K={K}, T={T}, r={r}, sigma={sigma}, type={option_type}, error={e}")
            return 0, 0, 0, 0

    @lru_cache(maxsize=100)
    def get_options_data(self, ticker: str, expiration_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, yf.Ticker]:
        """Fetch options data for a ticker and expiration date."""
        try:
            stock = yf.Ticker(ticker)
            opt_chain = stock.option_chain(expiration_date)
            return opt_chain.calls, opt_chain.puts, stock
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame(), pd.DataFrame(), None

    def find_iron_condor(self, ticker: str, calls: pd.DataFrame, puts: pd.DataFrame, current_price: float, 
                        expiration_date: str, T: float) -> Optional[Dict]:
        """Find an iron condor trade for a given ticker and expiration."""
        try:
            if calls.empty or puts.empty:
                logger.warning(f"{ticker}: No valid options data.")
                return None

            calls_otm = calls[calls['strike'] > current_price].sort_values('strike')
            puts_otm = puts[puts['strike'] < current_price].sort_values('strike', ascending=False)
            
            if len(calls_otm) < 2 or len(puts_otm) < 2:
                logger.warning(f"{ticker}: Insufficient OTM options.")
                return None
            
            call_sell = calls_otm.iloc[min(4, len(calls_otm) - 1)]
            put_sell = puts_otm.iloc[min(4, len(puts_otm) - 1)]
            
            max_put_spread = current_price * 0.10
            target_put_buy_strike = max(put_sell['strike'] - max_put_spread, puts['strike'].min())
            put_buy_candidates = puts[puts['strike'] <= target_put_buy_strike].sort_values('strike', ascending=False)
            put_buy = put_buy_candidates.iloc[0] if not put_buy_candidates.empty else puts[puts['strike'] < put_sell['strike']].iloc[0]
            
            call_buy = calls[calls['strike'] > call_sell['strike']].iloc[0]
            
            logger.debug(f"{ticker}: Strikes - Call Sell: {call_sell['strike']}, Call Buy: {call_buy['strike']}, "
                        f"Put Sell: {put_sell['strike']}, Put Buy: {put_buy['strike']}")

            call_sell_premium = call_sell['bid'] if pd.notna(call_sell['bid']) and call_sell['bid'] > 0 else call_sell.get('lastPrice', 0)
            call_buy_premium = call_buy['ask'] if pd.notna(call_buy['ask']) and call_buy['ask'] > 0 else call_buy.get('lastPrice', 0)
            put_sell_premium = put_sell['bid'] if pd.notna(put_sell['bid']) and put_sell['bid'] > 0 else put_sell.get('lastPrice', 0)
            put_buy_premium = put_buy['ask'] if pd.notna(put_buy['ask']) and put_buy['ask'] > 0 else put_buy.get('lastPrice', 0)
            
            total_premium = (call_sell_premium - call_buy_premium + put_sell_premium - put_buy_premium) * 100
            max_loss = max(call_buy['strike'] - call_sell['strike'], put_sell['strike'] - put_buy['strike']) * 100 - total_premium
            logger.debug(f"{ticker}: Calculated Max Loss per Contract: {max_loss}, Total with Contracts: {max_loss * (self.target_premium // total_premium + 1)}")

            delta_sell_call, gamma_sell_call, theta_sell_call, vega_sell_call = self.black_scholes_greeks(
                current_price, call_sell['strike'], T, self.risk_free_rate, call_sell['impliedVolatility'], 'call')
            delta_buy_call, gamma_buy_call, theta_buy_call, vega_buy_call = self.black_scholes_greeks(
                current_price, call_buy['strike'], T, self.risk_free_rate, call_buy['impliedVolatility'], 'call')
            delta_sell_put, gamma_sell_put, theta_sell_put, vega_sell_put = self.black_scholes_greeks(
                current_price, put_sell['strike'], T, self.risk_free_rate, put_sell['impliedVolatility'], 'put')
            delta_buy_put, gamma_buy_put, theta_buy_put, vega_buy_put = self.black_scholes_greeks(
                current_price, put_buy['strike'], T, self.risk_free_rate, put_buy['impliedVolatility'], 'put')
            
            net_gamma = -gamma_sell_call + gamma_buy_call - gamma_sell_put + gamma_buy_put
            net_theta = -theta_sell_call + theta_buy_call - theta_sell_put + theta_buy_put
            net_vega = -vega_sell_call + vega_buy_call - vega_sell_put + vega_buy_put
            
            pop = (1 - max(delta_sell_call, abs(delta_sell_put))) * 100
            pop = min(max(pop, 0), 100)
            
            if total_premium <= 0:
                logger.warning(f"{ticker}: Rejected - Premium <= 0 ({total_premium})")
                return None
            if max_loss <= 0:
                logger.warning(f"{ticker}: Rejected - Max Loss <= 0 ({max_loss})")
                return None
            if pop < self.min_pop:
                logger.warning(f"{ticker}: Rejected - PoP {pop:.2f}% < {self.min_pop}%")
                return None
            if max_loss > self.max_risk_per_trade:
                logger.warning(f"{ticker}: Rejected - Max Loss {max_loss} > {self.max_risk_per_trade}")
                return None
            
            return {
                'ticker': ticker,
                'call_sell_strike': call_sell['strike'],
                'call_buy_strike': call_buy['strike'],
                'put_sell_strike': put_sell['strike'],
                'put_buy_strike': put_buy['strike'],
                'total_premium': total_premium,
                'max_loss': max_loss,
                'pop': pop,
                'net_gamma': net_gamma,
                'net_theta': net_theta,
                'net_vega': net_vega,
                'expiration_date': expiration_date,
                'current_price': current_price
            }
        except Exception as e:
            logger.error(f"Error in {ticker} iron condor: {e}")
            return None

    def process_ticker(self, ticker: str) -> List[Dict]:
        """Process a single ticker and return a list of iron condor trades."""
        stock = yf.Ticker(ticker)
        recommendations = []
        for exp in stock.options:
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            days_to_exp = (exp_date - self.today).days
            logger.info(f"{ticker}: Checking expiration {exp} ({days_to_exp} days)")
            if self.min_days_to_expiry <= days_to_exp <= self.max_days_to_expiry:
                logger.info(f"{ticker}: Using expiration {exp}")
                calls, puts, stock_data = self.get_options_data(ticker, exp)
                current_price = stock_data.info.get('regularMarketPrice', stock_data.history(period='1d')['Close'].iloc[-1])
                T = days_to_exp / 365
                condor = self.find_iron_condor(ticker, calls, puts, current_price, exp, T)
                if condor:
                    target_contracts = int(self.target_premium / condor['total_premium']) + 1
                    max_contracts = int(self.max_risk_per_trade / condor['max_loss'])
                    condor['contracts'] = min(target_contracts, max_contracts)
                    condor['total_premium_all'] = condor['total_premium'] * condor['contracts']
                    if condor['total_premium_all'] < 0.5 * self.target_premium:
                        logger.warning(f"{ticker}: Rejected - Total premium {condor['total_premium_all']} < 50% of target ({0.5 * self.target_premium})")
                        continue
                    weeks_to_exp = days_to_exp / 7
                    condor['weekly_premium'] = condor['total_premium_all'] / weeks_to_exp if weeks_to_exp > 0 else 0
                    if condor['max_loss'] * condor['contracts'] <= self.max_risk_per_trade:
                        recommendations.append(condor)
                    else:
                        logger.warning(f"{ticker}: Rejected - Total risk {condor['max_loss'] * condor['contracts']} exceeds {self.max_risk_per_trade}")
        return recommendations

    def recommend_strategy(self) -> List[Dict]:
        """Generate iron condor recommendations for all tickers."""
        recommendations = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_ticker = {executor.submit(self.process_ticker, ticker): ticker for ticker in self.stocks}
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    ticker_recs = future.result()
                    recommendations.extend(ticker_recs)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
        recommendations.sort(key=lambda x: x['weekly_premium'], reverse=True)
        return recommendations

    def display_results(self, recommendations: List[Dict]) -> None:
        """Display or save the recommended trades."""
        if not recommendations:
            logger.info("No suitable iron condors found.")
            return
        
        df = pd.DataFrame(recommendations)
        display_df = df[['ticker', 'expiration_date', 'current_price', 'call_sell_strike', 'call_buy_strike',
                         'put_sell_strike', 'put_buy_strike', 'contracts', 'total_premium_all',
                         'max_loss', 'pop', 'weekly_premium']]
        # Round numerical columns for better readability
        display_df = display_df.round({
            'current_price': 2, 'call_sell_strike': 2, 'call_buy_strike': 2,
            'put_sell_strike': 2, 'put_buy_strike': 2, 'total_premium_all': 2,
            'max_loss': 2, 'pop': 2, 'weekly_premium': 2
        })
        logger.info("\nRecommended Iron Condors (Sorted by Weekly Premium, Highest to Lowest):")
        print(display_df.to_string(index=False))
        
        if self.output_file:
            try:
                display_df.to_csv(self.output_file, index=False)
                logger.info(f"Results saved to {self.output_file}")
            except Exception as e:
                logger.error(f"Error saving to {self.output_file}: {e}")

def parse_arguments() -> Dict:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Iron Condor Recommendation App")
    parser.add_argument('--portfolio_value', type=float, default=180000, help="Portfolio value in USD")
    parser.add_argument('--min_pop', type=float, default=80, help="Minimum probability of profit (%)")
    parser.add_argument('--target_premium', type=float, default=7500, help="Target premium for trades")
    parser.add_argument('--min_days', type=int, default=5, help="Minimum days to expiration")
    parser.add_argument('--max_days', type=int, default=200, help="Maximum days to expiration")
    parser.add_argument('--output', type=str, help="Output CSV file path")
    args = parser.parse_args()
    
    return {
        'portfolio_value': args.portfolio_value,
        'min_pop': args.min_pop,
        'target_premium': args.target_premium,
        'min_days_to_expiry': args.min_days,
        'max_days_to_expiry': args.max_days,
        'stocks': ['SPY', 'AAPL', 'TSLA', 'NVDA', 'AMD', 'GOOGL', 'NFLX', 'PLTR', 'MSFT', 'C', 'GS', 'META'],
        'output_file': args.output
    }

def main():
    """Main function to run the app."""
    config = parse_arguments()
    
    logger.info(f"Portfolio Value: ${config['portfolio_value']}")
    logger.info(f"Buying Power (30% Margin): ${config['portfolio_value'] / 0.3}")
    logger.info(f"Maximum Risk per Trade: ${(config['portfolio_value'] / 0.3) * 0.25}")
    logger.info(f"Target Premium: ${config['target_premium']}")
    logger.info(f"Minimum Probability of Profit: {config['min_pop']}%")
    
    finder = IronCondorFinder(config)
    recommendations = finder.recommend_strategy()
    finder.display_results(recommendations)

if __name__ == "__main__":
    main()