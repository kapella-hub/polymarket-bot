#!/usr/bin/env python3
"""Redeem resolved Polymarket positions for USDC on-chain."""
import os, json, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from eth_account import Account
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

pk = os.getenv('PM_POLYMARKET_WALLET_PRIVATE_KEY')
wallet = Account.from_key(pk).address
w3 = Web3(Web3.HTTPProvider('https://polygon-bor-rpc.publicnode.com'))
w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

CTF = Web3.to_checksum_address('0x4D97DCd97eC945f40cF65F87097ACe5EA0476045')
USDC_E = Web3.to_checksum_address('0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174')

ctf_abi = [
    {"inputs":[{"name":"_owner","type":"address"},{"name":"_id","type":"uint256"}],
     "name":"balanceOf","outputs":[{"name":"","type":"uint256"}],
     "stateMutability":"view","type":"function"},
    {"inputs":[{"name":"conditionId","type":"bytes32"}],
     "name":"payoutNumerators","outputs":[{"name":"","type":"uint256[]"}],
     "stateMutability":"view","type":"function"},
    {"inputs":[{"name":"conditionId","type":"bytes32"}],
     "name":"payoutDenominator","outputs":[{"name":"","type":"uint256"}],
     "stateMutability":"view","type":"function"},
    {"inputs":[
        {"name":"collateralToken","type":"address"},
        {"name":"parentCollectionId","type":"bytes32"},
        {"name":"conditionId","type":"bytes32"},
        {"name":"indexSets","type":"uint256[]"}
     ],
     "name":"redeemPositions","outputs":[],
     "stateMutability":"nonpayable","type":"function"},
]
ctf = w3.eth.contract(address=CTF, abi=ctf_abi)

# Get trade data
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds
creds = ApiCreds(
    api_key=os.getenv('PM_POLYMARKET_API_KEY'),
    api_secret=os.getenv('PM_POLYMARKET_API_SECRET'),
    api_passphrase=os.getenv('PM_POLYMARKET_API_PASSPHRASE'),
)
clob = ClobClient('https://clob.polymarket.com', key=pk, chain_id=137, signature_type=0)
clob.set_api_creds(creds)

trades = clob.get_trades()

# Map: market_condition_id -> {token_ids, balances}
markets = {}
for t in trades:
    if not isinstance(t, dict):
        continue
    market = t.get('market', '')
    asset = t.get('asset_id', '')
    if not market or not asset:
        continue
    if market not in markets:
        markets[market] = set()
    markets[market].add(asset)

print("Wallet: %s" % wallet)
print("Checking %d markets for redeemable positions...\n" % len(markets))

# Check USDC.e balance before
usdc_abi = [{"inputs":[{"name":"","type":"address"}],"name":"balanceOf",
             "outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"}]
usdc = w3.eth.contract(address=USDC_E, abi=usdc_abi)
usdc_before = usdc.functions.balanceOf(wallet).call() / 1e6
print("USDC.e before: $%.2f\n" % usdc_before)

total_redeemed = 0
# Fetch nonce once and increment manually to avoid pending-tx confusion
current_nonce = w3.eth.get_transaction_count(wallet, 'pending')

for market_id, token_ids in markets.items():
    # Check if any tokens have balance
    has_tokens = False
    for tid in token_ids:
        try:
            bal = ctf.functions.balanceOf(wallet, int(tid)).call()
        except Exception:
            continue
        if bal > 0:
            has_tokens = True
            break

    if not has_tokens:
        continue

    # Check if market has resolved (payoutDenominator > 0)
    condition_bytes = bytes.fromhex(market_id[2:]) if market_id.startswith('0x') else bytes.fromhex(market_id)

    try:
        denom = ctf.functions.payoutDenominator(condition_bytes).call()
    except Exception as e:
        print("SKIP %s... (payout check failed: %s)" % (market_id[:16], str(e)[:40]))
        continue

    if denom == 0:
        print("NOT RESOLVED: %s... (no payout set)" % market_id[:16])
        # Show token balances
        for tid in token_ids:
            try:
                bal = ctf.functions.balanceOf(wallet, int(tid)).call() / 1e6
            except Exception:
                continue
            if bal > 0:
                print("  token %s...: %.2f tokens (ACTIVE — hold)" % (tid[:16], bal))
        print()
        continue

    # Market is resolved! Check payouts
    try:
        numerators = ctf.functions.payoutNumerators(condition_bytes).call()
    except:
        numerators = []

    print("RESOLVED: %s..." % market_id[:16])
    print("  Payout: %s (denom=%d)" % (numerators, denom))

    # Show what we hold
    token_balances = []
    for tid in sorted(token_ids):
        try:
            bal = ctf.functions.balanceOf(wallet, int(tid)).call()
        except Exception:
            continue
        if bal > 0:
            token_balances.append((tid, bal))
            print("  Holding: %s... = %.2f tokens" % (tid[:16], bal / 1e6))

    if not token_balances:
        print("  No tokens to redeem")
        print()
        continue

    # Try to redeem
    # indexSets: [1, 2] for a 2-outcome binary market (bit 0 = YES, bit 1 = NO)
    # payoutNumerators may return [] even on resolved markets (UMA oracle path),
    # so always fall back to [1, 2] for standard Polymarket binary conditions.
    if numerators:
        index_sets = [1 << i for i in range(len(numerators))]
    else:
        index_sets = [1, 2]
    parent_collection = b'\x00' * 32

    print("  Redeeming with indexSets=%s..." % index_sets)
    try:
        gas_price = max(w3.eth.gas_price, w3.to_wei(50, 'gwei'))
        tx = ctf.functions.redeemPositions(
            USDC_E, parent_collection, condition_bytes, index_sets
        ).build_transaction({
            'from': wallet, 'nonce': current_nonce, 'gas': 300000,
            'gasPrice': gas_price,
        })
        signed = w3.eth.account.sign_transaction(tx, pk)
        h = w3.eth.send_raw_transaction(signed.raw_transaction)
        r = w3.eth.wait_for_transaction_receipt(h, timeout=60)
        current_nonce += 1

        usdc_after = usdc.functions.balanceOf(wallet).call() / 1e6
        gained = usdc_after - usdc_before
        usdc_before = usdc_after

        if r['status'] == 1:
            print("  REDEEMED! tx=%s gained=$%.2f" % (h.hex()[:20], gained))
            total_redeemed += gained
        else:
            print("  TX FAILED (status=0)")
    except Exception as e:
        print("  Redeem error: %s" % str(e)[:80])
    print()

# Final balance
usdc_final = usdc.functions.balanceOf(wallet).call() / 1e6
print("=" * 50)
print("Total redeemed: $%.2f" % total_redeemed)
print("USDC.e balance now: $%.2f" % usdc_final)
