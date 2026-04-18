from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY
import os
import json
import httpx
import time
from dotenv import load_dotenv


def main() -> None:
    load_dotenv("/opt/polymarket-bot/.env")

    creds = ApiCreds(
        api_key=os.getenv("PM_POLYMARKET_API_KEY"),
        api_secret=os.getenv("PM_POLYMARKET_API_SECRET"),
        api_passphrase=os.getenv("PM_POLYMARKET_API_PASSPHRASE"),
    )
    clob = ClobClient(
        "https://clob.polymarket.com",
        key=os.getenv("PM_POLYMARKET_WALLET_PRIVATE_KEY"),
        chain_id=137,
        signature_type=0,
    )
    clob.set_api_creds(creds)

    ts = int(time.time() // 900) * 900
    resp = httpx.get(
        f"https://gamma-api.polymarket.com/markets?slug=btc-updown-15m-{ts}",
        timeout=10,
    )
    data = resp.json()
    if not data:
        print("No market")
        raise SystemExit(1)

    m = data[0]
    tokens = json.loads(m["clobTokenIds"]) if isinstance(m["clobTokenIds"], str) else m["clobTokenIds"]
    prices = json.loads(m["outcomePrices"]) if isinstance(m["outcomePrices"], str) else m["outcomePrices"]

    q = m.get("question", "?")
    print(f"Market: {q}")
    print(f"Up={prices[0]} Down={prices[1]}")

    print("Placing test order: buy Up @ $0.01 for 5 shares...")
    try:
        order_args = OrderArgs(token_id=tokens[0], price=0.01, size=5, side=BUY)
        signed = clob.create_order(order_args)
        result = clob.post_order(signed, OrderType.GTC)
        print(f"ORDER PLACED: {result}")
        oid = result.get("orderID", result.get("id", ""))
        if oid:
            time.sleep(1)
            clob.cancel(oid)
            print(f"Cancelled: {oid}")
        print()
        print("*** LIVE TRADING CONFIRMED WORKING ***")
    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    main()
