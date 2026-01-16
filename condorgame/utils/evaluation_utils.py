
def count_evaluations(history_price, horizon, interval):
    ts_values = [ts for ts, _ in history_price]
    count = 0
    prev_ts = ts_values[0]
    for ts in ts_values[1:]:
        if ts - prev_ts >= interval:
            if ts - ts_values[0] >= horizon:
                count += 1
            prev_ts = ts
    return count