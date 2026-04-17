#!/usr/bin/env python3
"""Swap native USDC to USDC.e via Uniswap V3 on Polygon."""
import os, json, time, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from eth_account import Account
from web3 import Web3

pk = os.getenv('PM_POLYMARKET_WALLET_PRIVATE_KEY')
acct = Account.from_key(pk)
wallet = acct.address
w3 = Web3(Web3.HTTPProvider('https://1rpc.io/matic'))

NATIVE = Web3.to_checksum_address('0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359')
BRIDGED = Web3.to_checksum_address('0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174')
ROUTER = Web3.to_checksum_address('0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45')

ABI = json.loads('[{"constant":true,"inputs":[{"name":"","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"},{"constant":false,"inputs":[{"name":"s","type":"address"},{"name":"a","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"},{"constant":true,"inputs":[{"name":"o","type":"address"},{"name":"s","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"type":"function"}]')
SWAP_ABI = json.loads('[{"inputs":[{"components":[{"name":"tokenIn","type":"address"},{"name":"tokenOut","type":"address"},{"name":"fee","type":"uint24"},{"name":"recipient","type":"address"},{"name":"amountIn","type":"uint256"},{"name":"amountOutMinimum","type":"uint256"},{"name":"sqrtPriceLimitX96","type":"uint160"}],"name":"params","type":"tuple"}],"name":"exactInputSingle","outputs":[{"name":"amountOut","type":"uint256"}],"stateMutability":"payable","type":"function"}]')

native = w3.eth.contract(address=NATIVE, abi=ABI)
bridged = w3.eth.contract(address=BRIDGED, abi=ABI)
router = w3.eth.contract(address=ROUTER, abi=SWAP_ABI)

bal = native.functions.balanceOf(wallet).call()
nonce = w3.eth.get_transaction_count(wallet)
gp = w3.eth.gas_price
print("Native USDC: %.2f | Nonce: %d" % (bal/1e6, nonce))

# Approve
allow = native.functions.allowance(wallet, ROUTER).call()
if allow < bal:
    print("Approving...")
    tx = native.functions.approve(ROUTER, 2**256-1).build_transaction(
        {"from": wallet, "nonce": nonce, "gas": 100000, "gasPrice": gp})
    s = w3.eth.account.sign_transaction(tx, pk)
    h = w3.eth.send_raw_transaction(s.raw_transaction)
    r = w3.eth.wait_for_transaction_receipt(h, timeout=60)
    print("Approve: status=%d" % r["status"])
    nonce += 1

# Swap — try fee tiers
for fee in [100, 500, 3000, 10000]:
    print("Trying fee=%d..." % fee)
    params = (NATIVE, BRIDGED, fee, wallet, bal, int(bal * 0.98), 0)
    try:
        tx = router.functions.exactInputSingle(params).build_transaction(
            {"from": wallet, "nonce": nonce, "gas": 350000, "gasPrice": gp, "value": 0})
        s = w3.eth.account.sign_transaction(tx, pk)
        h = w3.eth.send_raw_transaction(s.raw_transaction)
        print("  tx: %s" % h.hex())
        r = w3.eth.wait_for_transaction_receipt(h, timeout=120)
        if r["status"] == 1:
            new = bridged.functions.balanceOf(wallet).call()
            print("  SUCCESS! USDC.e: %.2f" % (new/1e6))
            sys.exit(0)
        else:
            print("  Failed")
            nonce += 1
    except Exception as e:
        msg = str(e)
        if "nonce" in msg.lower():
            nonce = w3.eth.get_transaction_count(wallet)
        print("  Error: %s" % msg[:80])

print("All fee tiers failed. Try swapping manually on quickswap.exchange")
