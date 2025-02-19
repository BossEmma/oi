import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
import requests
import math


def fetch_data(pair, timeframe=mt5.TIMEFRAME_M5, count=200):
    rates = mt5.copy_rates_from_pos(pair, timeframe, 0, count)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df



def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_indicators(df):
    df['200_SMA'] = df['close'].rolling(window=200).mean()
    df['21_SMA'] = df['close'].rolling(window=21).mean()
    df['50_SMA'] = df['close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df, 14)
    return df


def generate_signal(df, currency_pair):
    df = calculate_indicators(df)
    latest = df.iloc[-1]
    stop_loss = None
    current_market_price = latest['close']

    if latest['close'] > latest['200_SMA'] and abs(latest['21_SMA'] - latest['50_SMA']) > 0.0001:
        df['direction'] = np.sign(df['close'] - df['open'])
        consecutive_sells = (df['direction'].shift(2) == -1) & (df['direction'].shift(3) == -1) & (df['direction'].shift(4) == -1)

        if consecutive_sells.iloc[-8]:
            last_candle = df.iloc[-4]
            previous_candle = df.iloc[-3]

            if last_candle['open'] < last_candle['close'] and \
                    last_candle['open'] < previous_candle['close'] < previous_candle['open'] < last_candle['close'] and \
                    df['RSI'].iloc[-4] > 50:                
                current_market_price = latest['close']
                stop_loss_distance = abs(latest['21_SMA'] - current_market_price)
                stop_loss = current_market_price - stop_loss_distance
                last_candle_type = "buy"

                if last_candle['open'] > last_candle['close']:
                    last_candle_type = "sell"

                return 1, stop_loss, current_market_price

    if latest['close'] < latest['200_SMA'] and abs(latest['21_SMA'] - latest['50_SMA']) > 0.0001:        
        df['direction'] = np.sign(df['close'] - df['open'])
        consecutive_buys = (df['direction'].shift(2) == 1) & (df['direction'].shift(3) == 1) & (
                df['direction'].shift(4) == 1)

        if consecutive_buys.iloc[-8] and df['RSI'].iloc[-4] < 50:
            last_candle = df.iloc[-3]
            previous_candle = df.iloc[-4]

            if last_candle['open'] > last_candle['close'] and \
                    last_candle['open'] > previous_candle['close'] and \
                    previous_candle['close'] > previous_candle['open'] and \
                    previous_candle['open'] > last_candle['close']:

                current_market_price = latest['close']
                stop_loss_distance = abs(latest['21_SMA'] - current_market_price)
                stop_loss = current_market_price + stop_loss_distance
                last_candle_type = "sell"

                if last_candle['open'] < last_candle['close']:
                    last_candle_type = "buy"
                return -1, stop_loss, current_market_price

    return 0, 0, current_market_price


def get_balance():
    account_info = mt5.account_info()
    balance = account_info.balance if account_info else 0
    return balance


def open_trade(pair, volume, stop_loss, trade_type):
    current_price = mt5.symbol_info_tick(pair).ask if trade_type == 'buy' else mt5.symbol_info_tick(pair).bid

    leverage = 100  
    required_margin = (volume * current_price) / 100

    account_info = mt5.account_info()

    balance = account_info.balance
    margin_free = account_info.margin_free
    equity = account_info.equity


    order_type = mt5.ORDER_TYPE_BUY if trade_type == 'buy' else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(pair).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pair).bid
    sl_price = price - stop_loss if order_type == mt5.ORDER_TYPE_BUY else price + stop_loss

    order = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pair,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": 0.0,
        "tp": 0.0,
        "deviation": 10,
        "magic": 234000,
        "comment": "Trading Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK
    }

    result = mt5.order_send(order)
    return result


def get_contract_size(pair):
    symbol_info = mt5.symbol_info(pair)

    trade_contract_size = getattr(symbol_info, 'trade_contract_size', None)

    return trade_contract_size


def execute(forex_pair):
    FOREX_PAIR = forex_pair
    dfs = fetch_data(FOREX_PAIR)
    signal, stop_loss_value, current_market_price = generate_signal(dfs, forex_pair)
    balance = get_balance()

    lot_size = get_contract_size(FOREX_PAIR)

    symbol_info = mt5.symbol_info(FOREX_PAIR)

    if symbol_info is None:
        return 

    min_lot_size = symbol_info.volume_min 
    max_lot_size = symbol_info.volume_max 

    one_lot_price = current_market_price * lot_size

    risked_capital = (balance * 0.02)

    one_pip_movement = 0.01 if "JPY" in FOREX_PAIR else 0.0001

    stop_loss_in_pip = abs(current_market_price - stop_loss_value) * one_pip_movement

    pip_value = (one_pip_movement / current_market_price) / lot_size

    lot_quantity = (risked_capital) * (stop_loss_in_pip * pip_value)

    volume = lot_quantity

    units = volume
    if signal == -1:
        trade_response = open_trade(FOREX_PAIR, units, stop_loss_value, 'sell')
        order_id = trade_response.order
        position = trade_response.request.position
        monitor_trade(order_id, position, units, "sell", stop_loss_value, current_market_price, FOREX_PAIR)

    elif signal == 1:
        trade_response = open_trade(FOREX_PAIR, units, stop_loss_value, 'buy')
        order_id = trade_response.order
        position = trade_response.request.position  
        monitor_trade(order_id, position, units, "buy", stop_loss_value, current_market_price, FOREX_PAIR)

    else:
        pass


def monitor_trade(order_id, position, volume, order_type, stop_loss, current_market_price, forex_pair):
    FOREX_PAIR = forex_pair
    para_para_stop_loss = stop_loss
    con_trade = {
        "v1":
            {
                "lot_size": 1,
                "trade_type": "",
                "status": False,
                "ticket_id": "",
            },
        "v2":
            {
                "lot_size": 1.33,
                "trade_type": "",
                "status": False,
                "ticket_id": "",
            },
        "v3":
            {
                "lot_size": 1,
                "trade_type": "",
                "status": False,
                "ticket_id": "",
            },
        "v4":
            {
                "lot_size": 1.33,
                "trade_type": "",
                "status": False,
                "ticket_id": "",
            },
        "v5":
            {
                "lot_size": 2.44,
                "trade_type": "",
                "status": False,
                "ticket_id": "",
            },
        "v6":
            {
                "lot_size": 3.99,
                "trade_type": "",
                "status": False,
                "ticket_id": "",
            },
        "v7":
            {
                "lot_size": 4.5,
                "trade_type": "",
                "status": False,
                "ticket_id": "",
            },
        "v8":
            {
                "lot_size": 6.7,
                "trade_type": "",
                "status": False,
                "ticket_id": "",
            },
        "v9":
            {
                "lot_size": 9.5,
                "trade_type": "",
                "status": False,
                "ticket_id": "",
            },
        "v10":
            {
                "lot_size": 11.33,
                "trade_type": "",
                "status": False,
                "ticket_id": "",
            },
        "v11":
            {
                "lot_size": 14.5,
                "trade_type": "",
                "status": False,
                "ticket_id": "",
            },
        "v12":
            {
                "lot_size": 17.53,
                "trade_type": "",
                "status": False,
                "ticket_id": "",
            },
        "v13":
            {
                "lot_size": 19.65,
                "trade_type": "",
                "status": False,
                "ticket_id": "",
            },
    }

    contingency_status = False

    while True:
        current_price = get_current_price(FOREX_PAIR)

        df = fetch_data(FOREX_PAIR)

        divergence_signal = detect_rsi_divergence(df)

        if divergence_signal == 1:  
            if order_type == "buy":
                response = mt5.Close(FOREX_PAIR, ticket=order_id)
                break
            elif order_type == "sell":
                response = mt5.Close(FOREX_PAIR, ticket=order_id)
                break

        if divergence_signal == -1: 
            if order_type == "buy":
                response = mt5.Close(FOREX_PAIR, ticket=order_id)
                break

            elif order_type == "sell":
                response = mt5.Close(FOREX_PAIR, ticket=order_id)
                break

        if check_stop_loss(current_price, stop_loss, order_type):
            contingency_status = True
            break

    if not contingency_status:
        return 0

    order_counter_type = "buy"
    order_counter_type_x = "sell"
    open_trade_bool = True
    v_counter_trade = 1

    lukki = "sell"
    if order_type == "sell":
        order_counter_type = "sell"
        order_counter_type_x = "buy"
        lukki = "buy"

    getta_trade = con_trade[f"v{v_counter_trade}"]
    getta_trade["status"] = True
    getta_trade["trade_type"] = lukki

    if order_type == "buy":
        leap = abs(current_market_price - stop_loss)
        counter_take_profit = stop_loss - leap
        normal_take_profit = current_market_price + leap

    elif order_type == "sell":
        leap = abs(stop_loss - current_market_price)
        counter_take_profit = stop_loss + leap
        normal_take_profit = current_market_price - leap

    else:
        return 0

    counter_take_profit = counter_take_profit
    normal_take_profit = normal_take_profit

    while True:
        if not contingency_status:
            break

        current_price = get_current_price(FOREX_PAIR)

        df = fetch_data(FOREX_PAIR)

        if not contingency_status:
            break
            
        if v_counter_trade == 14:
            break

        if open_trade_bool:
            get_trade = con_trade[f"v{v_counter_trade}"]
            if get_trade["status"]:
                new_volume = volume * get_trade["lot_size"]
                new_volume = math.ceil(new_volume * 100) / 100
 
                if order_counter_type == "sell":
                    order_counter_type = "buy"
                    get_trade["trade_type"] = "sell"
                    v_counter_trade += 1
                    trade_response = open_trade(FOREX_PAIR, new_volume, stop_loss, 'sell')
                    get_trade["ticket_id"] = trade_response.order

                elif order_counter_type == "buy":
                    order_counter_type = "sell"
                    get_trade["trade_type"] = "buy"
                    v_counter_trade += 1
                    trade_response = open_trade(FOREX_PAIR, new_volume, stop_loss, 'buy')
                    get_trade["ticket_id"] = trade_response.order

        if check_stop_loss(current_price, counter_take_profit, order_counter_type_x):
            response = mt5.Close(FOREX_PAIR, ticket=order_id)
            for key in con_trade.keys():
                if con_trade[key]["status"]:
                    response = mt5.Close(FOREX_PAIR, ticket=con_trade[key]["ticket_id"])
            break

        if check_take_profit(current_price, normal_take_profit, order_type):
            response = mt5.Close(FOREX_PAIR, ticket=order_id)
            for key in con_trade.keys():
                if con_trade[key]["status"]:
                    response = mt5.Close(FOREX_PAIR, ticket=con_trade[key]["ticket_id"])
            break


        if order_type == "buy":
            if order_counter_type == "sell":
                if current_price < current_market_price:
                    get_trade = con_trade[f"v{v_counter_trade}"]
                    get_trade["status"] = False
                    get_trade["trade_type"] = "buy"
                    pass

            elif order_counter_type == "buy":
                # current price must be above stop loss price
                if current_price < stop_loss:
                    get_trade = con_trade[f"v{v_counter_trade}"]
                    get_trade["status"] = False
                    get_trade["trade_type"] = "sell"
                    pass

        elif order_type == "sell":
            if order_counter_type == "buy":
                if current_price > current_market_price:
                    get_trade = con_trade[f"v{v_counter_trade}"]
                    get_trade["status"] = False
                    get_trade["trade_type"] = "sell"
                    pass

            elif order_counter_type == "sell":
                if current_price < stop_loss:
                    open_trade_bool = True
                    get_trade = con_trade[f"v{v_counter_trade}"]
                    get_trade["status"] = False
                    get_trade["trade_type"] = "buy"
                    pass
                pass



def detect_rsi_divergence(df, period=14):
    df = calculate_indicators(df)
    rsi = df['RSI']

    if (df['close'].iloc[-2] > df['close'].iloc[-period] and
            rsi.iloc[-3] < rsi.iloc[-period]):
        return 1

    if (df['close'].iloc[-2] < df['close'].iloc[-period] and
            rsi.iloc[-3] > rsi.iloc[-period]):
        return -1

    return 0


def get_current_price(symbol):
    price = mt5.symbol_info_tick(symbol).bid
    return price


def check_take_profit(current_price, take_profit, trade_type):
    if trade_type == 'buy':
        if current_price < take_profit:
            return True
    elif trade_type == 'sell':
        if current_price > take_profit:
            return True

    return False


def check_stop_loss(current_price, stop_loss, trade_type):
    if trade_type == 'buy':
        if current_price > stop_loss:
            return True

    elif trade_type == 'sell':
        if current_price < stop_loss:
            return True

    return False


def start_trading():
    mt5_login = 81338593
    mt5_password = "Mario123$$"
    mt5_server = "Exness-MT5Trial10"

    currency_pair = 'EURUSD'

    if not mt5.initialize(login=mt5_login, password=mt5_password, server=mt5_server):
        mt5.shutdown()
        exit()

    while True:
        execute(currency_pair)

    mt5.shutdown()



def login_trading():   
    mt5_login = 81338593
    mt5_password = "Mario123$$"
    mt5_server = "Exness-MT5Trial10"

    currency_pair = 'EURUSD'

    if not mt5.initialize(login=mt5_login, password=mt5_password, server=mt5_server):
        mt5.shutdown()
        exit()