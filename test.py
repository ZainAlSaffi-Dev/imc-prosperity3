def plot_relative_profiles(df: pd.DataFrame, prod: str, max_d: int = 10):
    """Aggregates LOB depth into mean relative volume profiles."""
    data = df[df["product"] == prod].copy()
    sides = {"bid": ("ask_price_1", "bid"), "ask": ("bid_price_1", "ask")}
    profiles = {}

    for name, (opp_ref, side) in sides.items():
        levels = []
        for i in range(1, 4):
            # Map absolute prices to tick distance from opposite best quote
            dist = (data[opp_ref] - data[f"{side}_price_{i}"]).abs()
            vol = data[f"{side}_volume_{i}"].fillna(0)
            levels.append(pd.DataFrame({"d": dist, "v": vol}))

        # Compute mean volume per tick distance via vectorized grouping
        profiles[name] = (
            pd.concat(levels).groupby("d")["v"].mean().reindex(range(1, max_d + 1))
        )

    plt.figure(figsize=(10, 4))
    plt.plot(
        profiles["bid"].index,
        profiles["bid"].values,
        "-.",
        color="blue",
        label="Bid Profile",
    )
    plt.plot(
        profiles["ask"].index,
        profiles["ask"].values,
        "-",
        color="red",
        label="Ask Profile",
    )

    plt.title(f"Mean Relative Volume Profile: {prod}")
    plt.xlabel("Tick Distance from Opposite Quote")
    plt.ylabel("Average Volume")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()


# Execution for high resolution regime identification
for asset in prices["product"].unique():
    plot_relative_profiles(prices, asset)
