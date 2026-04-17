#!/usr/bin/env python3
"""Swap native USDC to USDC.e via QuickSwap V3 on Polygon."""

import os
import json
import sys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from eth_account import Account
from web3 import Web3

pk = os.getenv('PM_POLYMARKET_WALLET_PRIVATE_KEY')
acct = Account.from_key(pk)
wallet = acct.address

w3 = Web3(Web3.HTTPProvider('https://1rpc.io/matic'))
assert w3.is_connected(), "Not connected to Polygon"

NATIVE_USDC = Web3.to_checksum_address('0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359')
BRIDGED_USDC = Web3.to_checksum_address('0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174')

# QuickSwap V3 SwapRouter
QUICKSWAP_ROUTER = Web3.to_checksum_address('0xf5b509bB0909a69B1c207E495f687a596C168E12')

ERC20_ABI = json.loads(
    '[{"constant":true,"inputs":[{"name":"","type":"address"}],"name":"balanceOf",'
    '"outputs":[{"name":"","type":"uint256"}],"type":"function"},'
    '{"constant":false,"inputs":[{"name":"spender","type":"address"},{"name":"amount","type":"uint256"}],'
    '"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"},'
    '{"constant":true,"inputs":[{"name":"owner","type":"address"},{"name":"spender","type":"address"}],'
    '"name":"allowance","outputs":[{"name":"","type":"uint256"}],"type":"function"}]'
)

# SwapRouter exactInputSingle ABI
ROUTER_ABI = json.loads(
    '[{"inputs":[{"components":[{"name":"tokenIn","type":"address"},{"name":"tokenOut","type":"address"},'
    '{"name":"fee","type":"uint24"},{"name":"recipient","type":"address"},{"name":"deadline","type":"uint256"},'
    '{"name":"amountIn","type":"uint256"},{"name":"amountOutMinimum","type":"uint256"},'
    '{"name":"sqrtPriceLimitX96","type":"uint160"}],"name":"params","type":"tuple"}],'
    '"name":"exactInputSingle","outputs":[{"name":"amountOut","type":"uint256"}],'
    '"stateMutability":"payable","type":"function"}]'
)

native = w3.eth.contract(address=NATIVE_USDC, abi=ERC20_ABI)
bridged = w3.eth.contract(address=BRIDGED_USDC, abi=ERC20_ABI)
router = w3.eth.contract(address=QUICKSWAP_ROUTER, abi=ROUTER_ABI)

# Check balances
native_bal = native.functions.balanceOf(wallet).call()
bridged_bal = bridged.functions.balanceOf(wallet).call()
pol_bal = w3.eth.get_balance(wallet)

print("Wallet: %s" % wallet)
print("Native USDC: %.2f" % (native_bal / 1e6))
print("Bridged USDC.e: %.2f" % (bridged_bal / 1e6))
print("POL (gas): %.4f" % (pol_bal / 1e18))

if native_bal < 1_000_000:  # Less than $1
    print("Not enough native USDC to swap")
    sys.exit(1)

amount_in = native_bal  # Swap all
print("\nSwapping %.2f native USDC -> USDC.e..." % (amount_in / 1e6))

# Step 1: Approve router to spend native USDC
allowance = native.functions.allowance(wallet, QUICKSWAP_ROUTER).call()
if allowance < amount_in:
    print("Approving router...")
    tx = native.functions.approve(QUICKSWAP_ROUTER, 2**256 - 1).build_transaction({
        'from': wallet,
        'nonce': w3.eth.get_transaction_count(wallet),
        'gas': 100000,
        'gasPrice': w3.eth.gas_price,
    })
    signed = w3.eth.account.sign_transaction(tx, pk)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    print("Approved: %s (status=%d)" % (tx_hash.hex(), receipt['status']))

# Step 2: Swap via exactInputSingle
# Fee tier: 100 = 0.01% (stablecoin pool)
import time
deadline = int(time.time()) + 300

swap_params = (
    NATIVE_USDC,      # tokenIn
    BRIDGED_USDC,      # tokenOut
    100,               # fee (0.01%)
    wallet,            # recipient
    deadline,          # deadline
    amount_in,         # amountIn
    int(amount_in * 0.995),  # amountOutMinimum (0.5% slippage)
    0,                 # sqrtPriceLimitX96
)

print("Executing swap...")
try:
    tx = router.functions.exactInputSingle(swap_params).build_transaction({
        'from': wallet,
        'nonce': w3.eth.get_transaction_count(wallet),
        'gas': 300000,
        'gasPrice': w3.eth.gas_price,
        'value': 0,
    })
    signed = w3.eth.account.sign_transaction(tx, pk)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print("Swap tx: %s" % tx_hash.hex())
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
    print("Status: %s" % ("SUCCESS" if receipt['status'] == 1 else "FAILED"))

    # Check new balance
    new_bridged = bridged.functions.balanceOf(wallet).call()
    print("\nNew USDC.e balance: %.2f" % (new_bridged / 1e6))
    print("Gained: %.2f USDC.e" % ((new_bridged - bridged_bal) / 1e6))

except Exception as e:
    print("Swap failed: %s" % e)
    # Try with higher fee tier (500 = 0.05%)
    print("\nRetrying with 0.05% fee tier...")
    swap_params2 = (NATIVE_USDC, BRIDGED_USDC, 500, wallet, deadline, amount_in, int(amount_in * 0.99), 0)
    try:
        tx = router.functions.exactInputSingle(swap_params2).build_transaction({
            'from': wallet,
            'nonce': w3.eth.get_transaction_count(wallet),
            'gas': 300000,
            'gasPrice': w3.eth.gas_price,
            'value': 0,
        })
        signed = w3.eth.account.sign_transaction(tx, pk)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        print("Swap tx: %s" % tx_hash.hex())
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        print("Status: %s" % ("SUCCESS" if receipt['status'] == 1 else "FAILED"))
        new_bridged = bridged.functions.balanceOf(wallet).call()
        print("New USDC.e balance: %.2f" % (new_bridged / 1e6))
    except Exception as e2:
        print("Retry failed: %s" % e2)
