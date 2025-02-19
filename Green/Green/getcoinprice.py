import requests

def get_litecoin_price():
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {'ids': 'litecoin', 'vs_currencies': 'usd'}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
        data = response.json()
        ltc_price = data['litecoin']['usd']
        return ltc_price
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Litecoin price: {e}")
        return None
    
def get_solana_price():
    # CoinGecko API endpoint for Solana
    api_url = "https://api.coingecko.com/api/v3/simple/price"
    
    # Parameters for the API request
    params = {
        'ids': 'solana',
        'vs_currencies': 'usd'
    }

    try:
        # Make the API request
        response = requests.get(api_url, params=params)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Get the current price of Solana in USD
            sol_price = data['solana']['usd']
            
            return sol_price
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

def get_usdt_tether_price():
    # CoinGecko API endpoint for Tether (TRC20) with vs_currencies parameter set to usd
    url = "https://api.coingecko.com/api/v3/simple/price?ids=tether&vs_currencies=usd"

    try:
        # Make a GET request to the CoinGecko API
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        # Parse the JSON response
        data = response.json()

        # Extract the price of USDT Tether (TRC20)
        usdt_price = data["tether"]["usd"]

        return usdt_price

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def get_shiba_price():
    # CoinGecko API endpoint for Shiba Inu
    url = "https://api.coingecko.com/api/v3/simple/price"
    
    # Parameters for the API request
    params = {
        'ids': 'shiba',
        'vs_currencies': 'usd'
    }

    try:
        # Making the API request
        response = requests.get(url, params=params)
        
        # Checking if the request was successful (status code 200)
        if response.status_code == 200:
            # Parsing the JSON response
            data = response.json()
            
            # Extracting the Shiba Inu price in USD
            shiba_price_usd = data['shiba']['usd']
            
            return shiba_price_usd
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code}")
            return None
    except Exception as e:
        # Print an error message if an exception occurs
        print(f"Error: {e}")
        return None
    

def get_eth_price():
    # CoinGecko API endpoint for Ethereum (ETH)
    api_url = 'https://api.coingecko.com/api/v3/simple/price'
    
    # Parameters for the API request
    params = {
        'ids': 'ethereum',
        'vs_currencies': 'usd'
    }

    try:
        # Make the API request
        response = requests.get(api_url, params=params)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            data = response.json()
            eth_price = data['ethereum']['usd']
            return eth_price
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_doge_price():
    # Define the CoinGecko API endpoint for Dogecoin
    url = "https://api.coingecko.com/api/v3/simple/price?ids=dogecoin&vs_currencies=usd"

    try:
        # Make a GET request to the CoinGecko API
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Extract the current price of Dogecoin in USD
            doge_price_usd = data['dogecoin']['usd']

            return doge_price_usd
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code}")
            return None
    except Exception as e:
        # Print an error message if an exception occurs
        print(f"An error occurred: {e}")
        return None
    
def get_bnb_price():
    # CoinGecko API endpoint for BNB on Binance Smart Chain
    url = "https://api.coingecko.com/api/v3/simple/price?ids=binancecoin&vs_currencies=usd"

    # Make the API request
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Get the current USD price of BNB
        bnb_price_usd = data['binancecoin']['usd']

        return bnb_price_usd
    else:
        # If the request was not successful, print the error code
        print(f"Error: {response.status_code}")
        return None
    
def send_smartsupp_notification(api_key, conversation_id, message):
    smartsupp_api_key = "e9eddf5bcef69feb01eb008588105d75c394a474"
    url = f'https://api.smartsupp.com/v2/conversations/{conversation_id}/messages'

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    payload = {
        'type': 'message',
        'text': message,
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        print('Message sent successfully')
    else:
        print(f'Error sending message: {response.text}')