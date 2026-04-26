import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style='whitegrid')


def plot_portfolio_performance(aggregated_performance: pd.DataFrame, filename='portfolio_performance.png') -> None:
    """
    Creates the graph of portfolio's absolute and percentage returns, including a bar for the total portfolio performance.

    Args:
        aggregated_performance (pd.DataFrame): The aggregated portfolio performance DataFrame.
        filename (str): The filename to save the plot.
    """
    # We assume that the last row is totals
    last_row = aggregated_performance.iloc[-1]
    aggr_performance_without_totals = aggregated_performance[:-1]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Absolute total return plot
    bars = axs[0].bar(aggr_performance_without_totals['ticker'], aggr_performance_without_totals['total return'], color='skyblue', label='Individual Stocks')
    total_bar = axs[0].bar(['total'], [last_row['total return']], color='steelblue', label='Portfolio Total')
    axs[0].set_title('Absolute Total Return (Including Dividends)')
    axs[0].set_ylabel('Total Return ($)')
    axs[0].legend()

    # Add numeric labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, yval, f'${round(yval, 2)}', ha='center', va='bottom')

    for bar in total_bar:
        yval = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, yval, f'${round(yval, 2)}', ha='center', va='bottom')

    # Percentage return plot
    bars = axs[1].bar(aggr_performance_without_totals['ticker'], aggr_performance_without_totals['percentage return'], color='lightgreen', label='Individual Stocks')
    total_bar_pct = axs[1].bar(['total'], [last_row['percentage return']], color='darkgreen', label='Portfolio Total')
    axs[1].set_title('Percentage Return (Including Dividends)')
    axs[1].set_ylabel('Return (%)')
    axs[1].legend()

    for bar in bars:
        yval = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2, yval, f'{round(yval, 2)}%', ha='center', va='bottom')

    for bar in total_bar_pct:
        yval = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2, yval, f'{round(yval, 2)}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def show_portfolio_as_image(portfolio_performance, filename='portfolio_table.png') -> None:
    """
    Generates an image representation of the portfolio DataFrame with adjustments for 'Total' row.
    """
    table_data = portfolio_performance.round(2).fillna("N/A")

    fig_w = len(table_data.columns)

    fig, ax = plt.subplots(figsize=(fig_w, 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     loc='center',
                     cellLoc='center',
                     colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(table_data.columns))))
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_portfolio_history_total(
    total_value: pd.Series,
    filename: str = 'portfolio_history_total.png',
    events_by_date: dict | None = None
) -> None:
    """
    Plot total portfolio value through time. Optionally draw vertical lines at purchase dates
    with ticker labels at the top.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(total_value.index, total_value.values, label='Portfolio Value')
    ax.set_title('Portfolio Value Over Time', pad=30)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value ($)')
    ax.legend()

    if events_by_date:
        # Merge events within 10 days of each other into a single label
        sorted_events = sorted(events_by_date.items())
        merged: dict = {}
        anchor_dt, anchor_label = sorted_events[0]
        for dt, label in sorted_events[1:]:
            if (dt - anchor_dt).days <= 10:
                existing = set(map(str.strip, anchor_label.split(',')))
                incoming = set(map(str.strip, label.split(',')))
                anchor_label = ", ".join(sorted(existing | incoming))
            else:
                merged[anchor_dt] = anchor_label
                anchor_dt, anchor_label = dt, label
        merged[anchor_dt] = anchor_label

        _, ymax = ax.get_ylim()
        y_text = ymax  # annotate at the top
        for dt, label in sorted(merged.items()):
            ax.axvline(dt, linestyle='--', linewidth=1, alpha=0.5)
            # place a small label at the top; slight vertical offset in axes coords
            ax.annotate(
                label,
                xy=(dt, y_text),
                xycoords=('data', 'data'),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center', va='bottom', fontsize=8, rotation=90
            )

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_portfolio_history_by_ticker(
    holdings_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    filename: str = 'portfolio_history_by_ticker.png',
    events_per_ticker: dict | None = None
) -> None:
    """
    Plot each ticker's contribution to portfolio value as lines.
    Optionally mark purchase dates with points and small labels.
    """
    per_ticker_values = holdings_df * prices_df

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in per_ticker_values.columns:
        series = per_ticker_values[col]
        ax.plot(series.index, series.values, label=col)

        if events_per_ticker and col in events_per_ticker:
            event_dates = events_per_ticker[col]
            # Find the y values at those dates (align on index)
            valid_dates = [d for d in event_dates if d in series.index]
            if valid_dates:
                yvals = series.loc[valid_dates].values
                ax.scatter(valid_dates, yvals, s=18, zorder=5)
                # Tiny labels just above each dot
                for x, y in zip(valid_dates, yvals):
                    ax.annotate(col, xy=(x, y), xytext=(0, 6),
                                textcoords='offset points', ha='center', va='bottom', fontsize=7, rotation=0)

    ax.set_title('Per-Ticker Position Value Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value ($)')
    ax.legend(ncols=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_prices_with_purchase_markers(
    prices_df: pd.DataFrame,
    events_per_ticker: dict[str, list[pd.Timestamp]] | None = None,
    filename: str = 'portfolio_prices_with_buys.png',
    normalize: bool = False,
    min_label_gap_days: int = 14,
    trim_pre_purchase: bool = True,   # hide days before first buy when normalize=True
) -> None:
    """
    Plot adjusted prices for all tickers (no quantity effect) and mark buy dates.
    If normalize=True, each ticker is rebased to 100 at its FIRST BUY date (per-ticker).
    """
    P = prices_df.copy()
    fig, ax = plt.subplots(figsize=(12, 6))

    for col in P.columns:
        s = P[col].dropna()
        if s.empty:
            continue

        # --- per-ticker normalization at first buy date ---
        if normalize:
            # pick the first buy date that exists in the price index; else fall back to first valid date
            anchor = None
            if events_per_ticker and col in events_per_ticker:
                aligned = [d for d in events_per_ticker[col] if d in s.index]
                if aligned:
                    anchor = min(aligned)
            if anchor is None:
                anchor = s.index[0]

            base = s.loc[anchor]
            if pd.isna(base) or base == 0:
                # Safety fallback
                base = s.dropna().iloc[0]

            s_norm = (s / float(base)) * 100.0

            if trim_pre_purchase and events_per_ticker and col in events_per_ticker:
                # hide dates before first buy so the line starts at the purchase
                s_norm = s_norm.where(s.index >= anchor)

            plot_series = s_norm
        else:
            plot_series = s

        ax.plot(plot_series.index, plot_series.values, label=col)

        if events_per_ticker and col in events_per_ticker:
            ev_dates = [d for d in events_per_ticker[col] if d in plot_series.index]
            if ev_dates:
                yvals = plot_series.loc[ev_dates].values
                ax.scatter(ev_dates, yvals, s=18, zorder=5)

                last_label_dt = None
                for x, y in zip(ev_dates, yvals):
                    if np.isnan(y):
                        continue
                    if last_label_dt is None or (x - last_label_dt).days >= min_label_gap_days:
                        ax.annotate(col, xy=(x, y), xytext=(0, 6),
                                    textcoords='offset points', ha='center', va='bottom', fontsize=7)
                        last_label_dt = x

    ax.set_title('Price History With Buy Markers' + (' — Rebased to First Buy (100)' if normalize else ''))
    ax.set_xlabel('Date')
    ax.set_ylabel('Index (100 at first buy)' if normalize else 'Price ($)')
    ax.legend(ncols=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)
