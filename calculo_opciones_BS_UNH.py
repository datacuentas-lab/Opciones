import yfinance as yf
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import pandas as pd
import requests
import warnings
warnings.filterwarnings('ignore')

def get_risk_free_rate_maturity(maturity_days):
    """
    Obtiene la tasa libre de riesgo según el vencimiento usando la curva de tesoro US
    """
    try:
        # Mapeo de vencimientos a tickers de bonos del Tesoro
        treasury_tickers = {
            (0, 30): "^IRX",      # 13-week Treasury Bill
            (31, 90): "^IRX",     # 13-week Treasury Bill  
            (91, 180): "^FVX",    # 5-year Treasury Note (como proxy)
            (181, 365): "^FVX",   # 5-year Treasury Note
            (366, 730): "^TNX",   # 10-year Treasury Note
            (731, 9999): "^TNX"   # 10-year Treasury Note
        }
        
        # Encontrar el ticker apropiado según los días al vencimiento
        ticker = "^IRX"  # Default
        for (min_days, max_days), tkr in treasury_tickers.items():
            if min_days <= maturity_days <= max_days:
                ticker = tkr
                break
        
        treasury = yf.Ticker(ticker)
        hist = treasury.history(period="1d")
        
        if not hist.empty:
            rate = hist['Close'].iloc[-1] / 100  # Convertir de porcentaje a decimal
            print(f"   → Usando {ticker} ({maturity_days} días): {rate*100:.2f}%")
            return rate
            
    except Exception as e:
        print(f"   → Error obteniendo tasa para {maturity_days} días: {e}")
    
    # Fallback: curva de tasas aproximada basada en días
    if maturity_days <= 30:
        default_rate = 0.045  # 4.5% para corto plazo
    elif maturity_days <= 90:
        default_rate = 0.047  # 4.7% 
    elif maturity_days <= 180:
        default_rate = 0.048  # 4.8%
    elif maturity_days <= 365:
        default_rate = 0.049  # 4.9%
    else:
        default_rate = 0.050  # 5.0% para largo plazo
        
    print(f"   → Usando tasa default para {maturity_days} días: {default_rate*100:.2f}%")
    return default_rate

def get_dividend_yield(ticker_symbol):
    """
    Obtiene el dividend yield del activo
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        
        # Intentar obtener dividend yield de diferentes campos
        dividend_yield = info.get('dividendYield') or info.get('trailingAnnualDividendYield') or info.get('forwardDividendYield')
        
        if dividend_yield:
            print(f"   → Dividend Yield encontrado: {dividend_yield*100:.2f}%")
            return dividend_yield
        else:
            # Calcular basado en dividendos recientes
            dividends = stock.dividends
            if len(dividends) > 0:
                last_dividend = dividends.iloc[-1]
                current_price = stock.history(period="1d")['Close'].iloc[-1]
                calculated_yield = (last_dividend * 4) / current_price  # Asumiendo dividendos trimestrales
                print(f"   → Dividend Yield calculado: {calculated_yield*100:.2f}%")
                return calculated_yield
                
    except Exception as e:
        print(f"   → Error obteniendo dividend yield: {e}")
    
    print("   → No se pudo obtener dividend yield, usando 0%")
    return 0.0

def validate_implied_volatility(iv):
    """
    Valida que la volatilidad implícita sea razonable
    """
    if pd.isna(iv) or iv <= 0:
        return 0.30  # Volatilidad default del 30%
    elif iv > 5:  # Más del 500% - probable error
        return 0.50  # Limitar al 50%
    else:
        return iv

def black_scholes_with_dividends(S, K, T, r, sigma, q=0, option_type='call'):
    """
    Calcula el precio de una opción usando Black-Scholes-Merton (con dividendos)
    
    Parámetros:
    S - Precio spot del activo
    K - Precio de ejercicio
    T - Tiempo al vencimiento en años
    r - Tasa libre de riesgo
    sigma - Volatilidad
    q - Dividend yield (tasa continua de dividendos)
    option_type - 'call' o 'put'
    """
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    # Validar inputs
    sigma = validate_implied_volatility(sigma)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
    return price

def get_option_contract_name(ticker, option_type, strike, expiration_date):
    """
    Genera el nombre del contrato de opción en formato estándar
    """
    date_str = expiration_date.strftime("%y%m%d")
    type_code = 'C' if option_type.lower() == 'call' else 'P'
    strike_cents = int(strike * 1000)
    strike_str = f"{strike_cents:08d}"
    
    contract_name = f"{ticker}{date_str}{type_code}{strike_str}"
    return contract_name

def get_option_data(ticker_symbol="UNH"):
    """
    Obtiene todos los datos necesarios para Black-Scholes
    """
    print(f"🔍 Obteniendo datos de {ticker_symbol}...")
    
    stock = yf.Ticker(ticker_symbol)
    
    # Obtener datos del spot price
    stock_info = stock.history(period="1d")
    if stock_info.empty:
        print("Error: No se pudo obtener datos del activo subyacente")
        return None
    
    S = stock_info['Close'].iloc[-1]
    print(f"📊 Precio spot de {ticker_symbol}: ${S:.2f}")
    
    # Obtener dividend yield una vez para todo el análisis
    dividend_yield = get_dividend_yield(ticker_symbol)
    
    try:
        expiration_dates = stock.options
        print(f"\n📅 Fechas de vencimiento disponibles: {len(expiration_dates)}")
        
        all_calls = []
        all_puts = []
        
        for exp_date in expiration_dates:
            print(f"  - {exp_date}")
            try:
                options_chain = stock.option_chain(exp_date)
                expiration_dt = datetime.strptime(exp_date, "%Y-%m-%d")
                
                # Procesar calls
                calls = options_chain.calls.copy()
                calls['expirationDate'] = expiration_dt
                calls['contractName'] = calls.apply(
                    lambda row: get_option_contract_name(ticker_symbol, 'call', row['strike'], expiration_dt), 
                    axis=1
                )
                calls['impliedVolatility'] = calls['impliedVolatility'].apply(validate_implied_volatility)
                all_calls.append(calls)
                
                # Procesar puts
                puts = options_chain.puts.copy()
                puts['expirationDate'] = expiration_dt
                puts['contractName'] = puts.apply(
                    lambda row: get_option_contract_name(ticker_symbol, 'put', row['strike'], expiration_dt), 
                    axis=1
                )
                puts['impliedVolatility'] = puts['impliedVolatility'].apply(validate_implied_volatility)
                all_puts.append(puts)
                
            except Exception as e:
                print(f"    Error procesando {exp_date}: {e}")
                continue
        
        # Combinar todos los datos
        all_calls_df = pd.concat(all_calls, ignore_index=True) if all_calls else pd.DataFrame()
        all_puts_df = pd.concat(all_puts, ignore_index=True) if all_puts else pd.DataFrame()
        
        print(f"\n📈 OPCIONES CALL totales: {len(all_calls_df)} contratos")
        print(f"📉 OPCIONES PUT totales: {len(all_puts_df)} contratos")
        
        return {
            'stock_price': S,
            'calls': all_calls_df,
            'puts': all_puts_df,
            'ticker': ticker_symbol,
            'expiration_dates': expiration_dates,
            'dividend_yield': dividend_yield
        }
        
    except Exception as e:
        print(f"Error obteniendo opciones: {e}")
        return None

def calculate_black_scholes_for_options(option_data):
    """
    Calcula precios Black-Scholes para todas las opciones con tasas dinámicas
    """
    if not option_data:
        return
    
    S = option_data['stock_price']
    q = option_data['dividend_yield']
    today = datetime.now()
    
    print(f"\n💡 DATOS BASE PARA CÁLCULOS:")
    print(f"   Precio Spot (S): ${S:.2f}")
    print(f"   Dividend Yield (q): {q:.4f} ({q*100:.2f}%)")
    print(f"   Fecha de análisis: {today.strftime('%Y-%m-%d')}")
    
    # Agrupar por fecha de vencimiento
    calls_by_expiration = option_data['calls'].groupby('expirationDate')
    puts_by_expiration = option_data['puts'].groupby('expirationDate')
    
    # Procesar CALLS por fecha de vencimiento
    for exp_date, calls_group in calls_by_expiration:
        if calls_group.empty:
            continue
            
        T_days_common = (exp_date - today).days
        expiration_str = exp_date.strftime("%Y-%m-%d")
        
        # Obtener tasa libre de riesgo específica para este vencimiento
        r = get_risk_free_rate_maturity(T_days_common)
        
        print(f"\n{'='*120}")
        print(f"📈 OPCIONES CALL - Vencimiento: {expiration_str} (en {T_days_common} días)")
        print(f"   Tasa libre de riesgo: {r*100:.2f}%, Dividend Yield: {q*100:.2f}%")
        print(f"{'='*120}")
        print(f"{'Contrato':<20} {'Strike':<10} {'Last Price':<12} {'IV':<8} {'BS Price':<12} {'Diferencia':<12} {'% Diff':<10} {'Moneyness':<10}")
        print(f"{'-'*120}")
        
        for _, option in calls_group.iterrows():
            try:
                T_days = (option['expirationDate'] - today).days
                T = T_days / 365.0
                
                if T <= 0:
                    continue
                
                K = option['strike']
                last_price = option['lastPrice'] if option['lastPrice'] > 0 else (option['bid'] + option['ask']) / 2
                iv = option['impliedVolatility']
                contract_name = option['contractName']
                
                # Calcular moneyness
                moneyness = "ITM" if S > K else "OTM" if S < K else "ATM"
                
                # Calcular Black-Scholes con dividendos
                bs_price = black_scholes_with_dividends(S, K, T, r, iv, q, 'call')
                
                # Calcular diferencias
                diff = bs_price - last_price
                percent_diff = (diff / last_price) * 100 if last_price > 0 else 0
                
                print(f"{contract_name:<20} ${K:<9.1f} ${last_price:<11.2f} {iv:<7.3f} ${bs_price:<11.2f} ${diff:<11.2f} {percent_diff:<9.1f}% {moneyness:<10}")
                
            except Exception as e:
                continue
    
    # Procesar PUTS por fecha de vencimiento
    for exp_date, puts_group in puts_by_expiration:
        if puts_group.empty:
            continue
            
        T_days_common = (exp_date - today).days
        expiration_str = exp_date.strftime("%Y-%m-%d")
        
        # Obtener tasa libre de riesgo específica para este vencimiento
        r = get_risk_free_rate_maturity(T_days_common)
        
        print(f"\n{'='*120}")
        print(f"📉 OPCIONES PUT - Vencimiento: {expiration_str} (en {T_days_common} días)")
        print(f"   Tasa libre de riesgo: {r*100:.2f}%, Dividend Yield: {q*100:.2f}%")
        print(f"{'='*120}")
        print(f"{'Contrato':<20} {'Strike':<10} {'Last Price':<12} {'IV':<8} {'BS Price':<12} {'Diferencia':<12} {'% Diff':<10} {'Moneyness':<10}")
        print(f"{'-'*120}")
        
        for _, option in puts_group.iterrows():
            try:
                T_days = (option['expirationDate'] - today).days
                T = T_days / 365.0
                
                if T <= 0:
                    continue
                
                K = option['strike']
                last_price = option['lastPrice'] if option['lastPrice'] > 0 else (option['bid'] + option['ask']) / 2
                iv = option['impliedVolatility']
                contract_name = option['contractName']
                
                # Calcular moneyness
                moneyness = "ITM" if S < K else "OTM" if S > K else "ATM"
                
                # Calcular Black-Scholes con dividendos
                bs_price = black_scholes_with_dividends(S, K, T, r, iv, q, 'put')
                
                # Calcular diferencias
                diff = bs_price - last_price
                percent_diff = (diff / last_price) * 100 if last_price > 0 else 0
                
                print(f"{contract_name:<20} ${K:<9.1f} ${last_price:<11.2f} {iv:<7.3f} ${bs_price:<11.2f} ${diff:<11.2f} {percent_diff:<9.1f}% {moneyness:<10}")
                
            except Exception as e:
                continue

def show_expiration_summary(option_data):
    """
    Muestra un resumen de las fechas de vencimiento disponibles
    """
    if not option_data:
        return
    
    print(f"\n📋 RESUMEN DE FECHAS DE VENCIMIENTO")
    print(f"{'='*60}")
    
    today = datetime.now()
    
    for i, exp_date_str in enumerate(option_data['expiration_dates']):
        exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d")
        days_to_exp = (exp_date - today).days
        
        calls_count = len(option_data['calls'][option_data['calls']['expirationDate'] == exp_date])
        puts_count = len(option_data['puts'][option_data['puts']['expirationDate'] == exp_date])
        
        # Obtener tasa para mostrar en resumen
        rate = get_risk_free_rate_maturity(days_to_exp)
        
        print(f"{i+1:2d}. {exp_date_str} (en {days_to_exp:3d} días) - {calls_count:3d}C/{puts_count:3d}P - Tasa: {rate*100:5.2f}%")

def main():
    """
    Función principal
    """
    print("🎯 CALCULADORA BLACK-SCHOLES-MERTON MEJORADA")
    print("UnitedHealth (UNH) - Con tasas dinámicas y dividendos\n")
    
    option_data = get_option_data("UNH")
    
    if option_data is not None:
        show_expiration_summary(option_data)
        calculate_black_scholes_for_options(option_data)
        
        print(f"\n🎯 RESUMEN EJECUTIVO:")
        print(f"• Ticker analizado: {option_data['ticker']}")
        print(f"• Precio actual: ${option_data['stock_price']:.2f}")
        print(f"• Dividend Yield: {option_data['dividend_yield']*100:.2f}%")
        print(f"• Fechas de vencimiento: {len(option_data['expiration_dates'])}")
        print(f"• Total calls analizadas: {len(option_data['calls'])}")
        print(f"• Total puts analizadas: {len(option_data['puts'])}")
        print(f"• ITM = In the Money, OTM = Out of the Money, ATM = At the Money")
    else:
        print("❌ No se pudieron obtener los datos de opciones")

if _name_ == "_main_":
    main()