import pandas as pd
import numpy as np
from datetime import datetime

def parse_date(d):
    """Parse date-like object to pd.Timestamp (midnight)."""
    return pd.to_datetime(d).normalize()

def price_storage_contract(
    actions,
    price_lookup,
    max_capacity,
    inj_rate,
    wd_rate,
    storage_cost_per_unit_per_day=0.0,
    inj_cost_per_unit=0.0,
    wd_cost_per_unit=0.0,
    initial_inventory=0.0,
    allow_partial_fill=True,
    verbose=False
):
    """
    Prototype storage contract pricer.

    Parameters
    ----------
    actions : list of dict
        Each dict: {'date': date-like, 'type': 'inject'|'withdraw', 'volume': float}
        Volume is positive requested volume (MWh/MMBtu/etc).
        Actions need not be sorted — they will be processed in chronological order.
    price_lookup : callable or pandas.Series
        - If callable: price_lookup(date) -> price (float)
        - If pandas.Series indexed by pd.Timestamp: will use .loc[date] if exact match,
          else will interpolate linearly between index entries (time-based).
    max_capacity : float
        Maximum inventory capacity.
    inj_rate : float
        Maximum injection rate per action (applies per event).
    wd_rate : float
        Maximum withdrawal rate per action (applies per event).
    storage_cost_per_unit_per_day : float
        Storage cost charged on inventory per day (can be fractional).
    inj_cost_per_unit, wd_cost_per_unit : float
        Additional per-unit cost/fee for injecting/withdrawing (e.g. efficiencies, fees).
    initial_inventory : float
        Starting inventory (must be <= max_capacity).
    allow_partial_fill : bool
        If True, requested volumes exceeding constraints will be partially filled to feasible amount.
        If False, an infeasible action will raise an Exception.
    verbose : bool
        Print warnings about partial fills / constraints.

    Returns
    -------
    total_value : float
        NPV (interest rate = 0) of cash flows to the client (positive means client receives money).
    ledger : pandas.DataFrame
        Detailed chronological ledger with columns:
        ['date', 'action', 'requested_volume', 'executed_volume', 'price', 'cashflow', 'inventory_before', 'inventory_after',
         'days_since_prev', 'storage_cost_accrued']
    """

    # Prepare actions DF
    df_act = pd.DataFrame(actions).copy()
    if df_act.empty:
        raise ValueError("No actions provided.")
    if 'date' not in df_act.columns or 'type' not in df_act.columns or 'volume' not in df_act.columns:
        raise ValueError("Each action must contain 'date', 'type', and 'volume' fields.")
    # Normalize
    df_act['date'] = df_act['date'].apply(parse_date)
    df_act['type'] = df_act['type'].str.lower()
    if not set(df_act['type']).issubset({'inject', 'withdraw'}):
        raise ValueError("Action 'type' must be either 'inject' or 'withdraw'.")

    df_act = df_act.sort_values('date').reset_index(drop=True)

    # Validate initial inventory
    if initial_inventory < 0 or initial_inventory > max_capacity:
        raise ValueError("initial_inventory must be between 0 and max_capacity.")

    # Helper: price fetch
    def get_price(dt):
        if callable(price_lookup):
            p = price_lookup(dt)
            if p is None or (isinstance(p, float) and np.isnan(p)):
                raise ValueError(f"No price returned from price_lookup for date {dt}")
            return float(p)
        elif isinstance(price_lookup, pd.Series):
            s = price_lookup.copy()
            s.index = pd.to_datetime(s.index)
            dt = pd.to_datetime(dt)
            # if exact index match
            if dt in s.index:
                return float(s.loc[dt])
            # else linear time interpolation
            s = s.sort_index()
            if dt < s.index[0] or dt > s.index[-1]:
                raise ValueError(f"Price lookup date {dt.date()} outside price series range ({s.index[0].date()} - {s.index[-1].date()}).")
            return float(np.interp(pd.to_datetime(dt).value, s.index.astype(np.int64), s.values))
        else:
            raise ValueError("price_lookup must be callable or a pandas.Series")

    ledger_rows = []
    inventory = float(initial_inventory)
    prev_date = df_act.loc[0, 'date']  # storage cost accrual from first action date is zero
    cumulative_cash = 0.0

    for idx, row in df_act.iterrows():
        date = row['date']
        action_type = row['type']
        req_vol = float(row['volume'])
        if req_vol < 0:
            raise ValueError("Requested volumes must be non-negative.")

        # Accrue storage cost for days since prev action (inventory held during this interval)
        days = (date - prev_date).days if idx > 0 else 0
        storage_cost = inventory * storage_cost_per_unit_per_day * days

        # Determine feasible executed volume given rates & capacity & inventory
        if action_type == 'inject':
            # cannot inject more than inj_rate, or space remaining
            max_by_rate = inj_rate
            max_by_capacity = max_capacity - inventory
            feasible = min(max_by_rate, max_by_capacity)
            exec_vol = min(req_vol, feasible)
            if exec_vol < req_vol:
                if not allow_partial_fill:
                    raise ValueError(f"Injection of {req_vol} on {date.date()} infeasible (max feasible {feasible}).")
                if verbose:
                    print(f"WARNING: Injection on {date.date()} partial: requested {req_vol}, executed {exec_vol}")
            # cashflow: client pays price*executed + injection cost * executed (negative for client)
            price = get_price(date)
            cash = - (price + inj_cost_per_unit) * exec_vol
            inventory_after = inventory + exec_vol
        else:  # withdraw
            # cannot withdraw more than wd_rate or than inventory
            max_by_rate = wd_rate
            max_by_inventory = inventory
            feasible = min(max_by_rate, max_by_inventory)
            exec_vol = min(req_vol, feasible)
            if exec_vol < req_vol:
                if not allow_partial_fill:
                    raise ValueError(f"Withdrawal of {req_vol} on {date.date()} infeasible (max feasible {feasible}).")
                if verbose:
                    print(f"WARNING: Withdrawal on {date.date()} partial: requested {req_vol}, executed {exec_vol}")
            # cashflow: client receives price*executed - withdrawal cost (positive for client)
            price = get_price(date)
            cash = + (price - wd_cost_per_unit) * exec_vol
            inventory_after = inventory - exec_vol

        # Sum cashflow: include storage_cost for the preceding interval (negative for client)
        net_cash = cash - storage_cost
        cumulative_cash += net_cash

        # Ledger row
        ledger_rows.append({
            'date': date,
            'action': action_type,
            'requested_volume': req_vol,
            'executed_volume': exec_vol,
            'price': price,
            'cashflow_action': cash,
            'storage_cost_accrued': storage_cost,
            'net_cashflow_this_row': net_cash,
            'inventory_before': inventory,
            'inventory_after': inventory_after,
            'days_since_prev': days
        })

        # Update state
        inventory = inventory_after
        prev_date = date

    # Note: no storage cost accrual after last action unless user wants to include a terminal holding cost.
    total_value = cumulative_cash

    ledger = pd.DataFrame(ledger_rows)
    # sort and set types
    ledger = ledger.sort_values('date').reset_index(drop=True)
    # Add cumulative columns
    ledger['cumulative_cashflow'] = ledger['net_cashflow_this_row'].cumsum()

    return float(total_value), ledger

# ---------------------------
# Example tests
# ---------------------------
if __name__ == "__main__":
    # Simple synthetic price series (daily) for testing
    dates = pd.date_range("2024-01-01", "2024-04-30", freq='D')
    # make up a price curve: ramp up then down
    prices = 20 + 2 * np.sin(np.linspace(0, 3.14, len(dates)))  # simple oscillation
    price_series = pd.Series(prices, index=dates)

    # Example actions: client wants to inject on Jan 10, Feb 15; withdraw on Mar 20 and Apr 15
    actions = [
        {'date': '2024-01-10', 'type': 'inject', 'volume': 100.0},
        {'date': '2024-02-15', 'type': 'inject', 'volume': 50.0},
        {'date': '2024-03-20', 'type': 'withdraw', 'volume': 80.0},
        {'date': '2024-04-15', 'type': 'withdraw', 'volume': 100.0},
    ]

    # contract parameters
    max_capacity = 200.0
    inj_rate = 120.0  # can inject up to 120 in a single action
    wd_rate = 90.0    # can withdraw up to 90 in a single action
    storage_cost_per_unit_per_day = 0.001  # small daily storage fee
    initial_inventory = 0.0

    # Price lookup: we use the series; the function interpolates if a date not in index
    total_val, ledger = price_storage_contract(
        actions=actions,
        price_lookup=price_series,
        max_capacity=max_capacity,
        inj_rate=inj_rate,
        wd_rate=wd_rate,
        storage_cost_per_unit_per_day=storage_cost_per_unit_per_day,
        initial_inventory=initial_inventory,
        verbose=True
    )

    print("Total contract value to client (interest=0):", total_val)
    print("\nLedger:")
    print(ledger.to_string(index=False, 
                           columns=['date','action','requested_volume','executed_volume','price','cashflow_action',
                                    'storage_cost_accrued','net_cashflow_this_row','inventory_before','inventory_after',
                                    'cumulative_cashflow']))
