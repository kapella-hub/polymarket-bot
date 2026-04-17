#!/usr/bin/env python3
"""Approve CTF Exchange contracts and try to sell all token positions."""
import os, json, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from eth_account import Account
from web3 import Web3
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.order_builder.constants import SELL

pk = os.getenv('PM_POLYMARKET_WALLET_PRIVATE_KEY')
wallet = Account.from_key(pk).address
w3 = Web3(Web3.HTTPProvider('https://1rpc.io/matic'))

creds = ApiCreds(
    api_key=os.getenv('PM_POLYMARKET_API_KEY'),
    api_secret=os.getenv('PM_POLYMARKET_API_SECRET'),
    api_passphrase=os.getenv('PM_POLYMARKET_API_PASSPHRASE'),
)
clob = ClobClient('https://clob.polymarket.com', key=pk, chain_id=137, signature_type=0)
clob.set_api_creds(creds)

CTF = Web3.to_checksum_address('0x4D97DCd97eC945f40cF65F87097ACe5EA0476045')
EXCHANGE = Web3.to_checksum_address('0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E')
NEG_EXCHANGE = Web3.to_checksum_address('0xC5d563A36AE78145C45a50134d48A1215220f80a')

approve_abi = [
    {"inputs":[{"name":"operator","type":"address"},{"name":"approved","type":"bool"}],
     "name":"setApprovalForAll","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"name":"owner","type":"address"},{"name":"operator","type":"address"}],
     "name":"isApprovedForAll","outputs":[{"name":"","type":"bool"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"name":"_owner","type":"address"},{"name":"_id","type":"uint256"}],
     "name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
]
ctf = w3.eth.contract(address=CTF, abi=approve_abi)

# Step 1: Approve Exchange contracts
for name, addr in [('Exchange', EXCHANGE), ('NegRisk', NEG_EXCHANGE)]:
    approved = ctf.functions.isApprovedForAll(wallet, addr).call()
    print("%s approved: %s" % (name, approved))
    if not approved:
        print("  Approving...")
        nonce = w3.eth.get_transaction_count(wallet)
        tx = ctf.functions.setApprovalForAll(addr, True).build_transaction({
            'from': wallet, 'nonce': nonce, 'gas': 100000, 'gasPrice': w3.eth.gas_price})
        signed = w3.eth.account.sign_transaction(tx, pk)
        h = w3.eth.send_raw_transaction(signed.raw_transaction)
        r = w3.eth.wait_for_transaction_receipt(h, timeout=60)
        print("  Done: status=%d" % r['status'])

# Step 2: Update CLOB allowance
print("\nUpdating CLOB allowance...")
try:
    clob.update_balance_allowance()
    print("Done")
except Exception as e:
    print("Error (may be OK): %s" % str(e)[:80])

# Step 3: Find all tokens with on-chain balance
print("\nScanning on-chain balances...")
trades = clob.get_trades()
full_ids = set()
for t in trades:
    if isinstance(t, dict):
        full_ids.add(t.get('asset_id', ''))

total_sold = 0
total_held_value = 0

for tid in sorted(full_ids):
    if not tid:
        continue
    try:
        bal = ctf.functions.balanceOf(wallet, int(tid)).call()
        tokens = bal / 1e6
        if tokens < 0.5:
            continue

        # Try to get order book
        try:
            book = clob.get_order_book(tid)
            bids = sorted([(float(b.price), float(b.size)) for b in book.bids], reverse=True)

            if bids and bids[0][0] > 0.01:
                best_bid = bids[0][0]
                bid_size = bids[0][1]
                sell_qty = min(tokens, bid_size * 0.8)
                proceeds = sell_qty * best_bid
                total_held_value += tokens * best_bid

                print("SELLING: %s..." % tid[:20])
                print("  %.1f tokens @ $%.3f bid = $%.2f" % (sell_qty, best_bid, proceeds))

                try:
                    order_args = OrderArgs(
                        token_id=tid, price=best_bid, size=round(sell_qty, 2), side=SELL)
                    signed_order = clob.create_order(order_args)
                    result = clob.post_order(signed_order, OrderType.GTC)
                    oid = result.get('orderID', '')
                    if oid:
                        print("  SOLD! order=%s" % oid[:20])
                        total_sold += proceeds
                    else:
                        errmsg = result.get('error', result.get('errorMsg', str(result)[:80]))
                        print("  Failed: %s" % errmsg)
                except Exception as e:
                    print("  Sell error: %s" % str(e)[:80])
            else:
                print("NO BIDS: %s... (%.1f tokens on-chain, no buyers)" % (tid[:16], tokens))
        except Exception as e:
            msg = str(e)
            if '404' in msg:
                print("EXPIRED: %s... (%.1f tokens — market closed, may be redeemable)" % (tid[:16], tokens))
            else:
                print("BOOK ERR: %s... %s" % (tid[:16], msg[:50]))
    except Exception as e:
        print("BALANCE ERR: %s" % str(e)[:60])

print("\n=== SUMMARY ===")
print("Proceeds from sales: $%.2f" % total_sold)
print("Estimated value of held tokens: $%.2f" % total_held_value)
