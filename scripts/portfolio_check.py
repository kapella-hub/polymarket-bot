#!/usr/bin/env python3
"""Check current portfolio state."""
import json

state = json.loads(open("data/power_trade_state.json").read())
print("=== POWER TRADE PORTFOLIO ===")
bankroll = state["bankroll"]
print("Bankroll (cash): $%.2f" % bankroll)
print("Open positions: %d" % len([t for t in state["trades"] if t["status"] == "open"]))
print()

total_tokens = 0
total_invested = 0
for t in state["trades"]:
    q = t["question"][:50]
    price = t["price"]
    size = t["size_usd"]
    tokens = t["tokens"]
    payout = tokens * 0.90
    profit = payout - size
    total_tokens += tokens
    total_invested += size
    print("%s | %s" % (t["status"], q))
    print("  %s @ $%.3f | $%.2f invested | %.1f tokens" % (t["side"], price, size, tokens))
    print("  If NO wins: $%.2f payout = $%+.2f profit (%+.0f%%)" % (payout, profit, profit/size*100))
    print()

print("Total invested: $%.2f" % total_invested)
print("Total tokens: %.1f" % total_tokens)
all_payout = total_tokens * 0.90
print("If ALL resolve NO: $%.2f payout" % all_payout)
print("Projected profit: $%+.2f" % (all_payout - total_invested))
print("Cash remaining: $%.2f" % bankroll)
print("Projected total value: $%.2f" % (bankroll + all_payout))
print("Starting bankroll was: $150.00")
print("Projected return: %+.0f%%" % ((bankroll + all_payout - 150) / 150 * 100))
