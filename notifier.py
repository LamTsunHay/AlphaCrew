"""Output delivery: terminal rendering and Telegram stub."""


def print_to_terminal(strategy_cards: list, log_entries: list) -> None:
    """Print strategy cards and gate summary to stdout."""
    print("\n" + "=" * 60)
    print("  STRATEGY ENGINE — RESULTS")
    print("=" * 60)

    # Gate summary from log entries
    gate_steps = [e for e in log_entries if "step" in e]
    if gate_steps:
        print("\n--- Gate Summary ---")
        for entry in gate_steps:
            print(f"  {entry['step']}: {entry.get('result', entry)}")

    # Strategy cards
    print(f"\n--- Strategy Cards ({len(strategy_cards)} qualified) ---\n")
    if not strategy_cards:
        print("  NO QUALIFIED CANDIDATES\n")
    else:
        for card in strategy_cards:
            print(card)
            print()

    print("=" * 60 + "\n")


def send_telegram(strategy_cards: list, log_entries: list) -> None:
    """FUTURE_STUB: Send strategy cards to a Telegram channel via bot.

    To activate:
      1. pip install python-telegram-bot
      2. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID to .env
      3. Implement this function using telegram.Bot.send_message()
    """
    pass
