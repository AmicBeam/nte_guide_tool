"""Gift costs from zero affinity to the configured full affinity cap."""
from __future__ import annotations

from copy import deepcopy


def with_gift_costs(affinity):
    result = deepcopy(affinity)
    total = sum(row['max_value'] for row in result['levels']) if result['levels'] else None
    result['total_affinity'] = total
    for gift in result['gifts']:
        gain = gift['affinity_gain']
        count = (total + gain - 1) // gain if total is not None and gain > 0 else None
        gift['required_gifts'] = count
        # Use Fons list prices only; conditional discounts are excluded.
        prices = {}
        for offer in gift['offers']:
            currency = offer['currency_id']
            if currency.casefold() != 'fons':
                continue
            if currency not in prices or offer['price'] < prices[currency]['unit_price']:
                prices[currency] = {
                    'currency_id': currency, 'currency_name': offer['currency_name'],
                    'unit_price': offer['price'],
                    'total_cost': count * offer['price'] if count is not None else None,
                }
        gift['costs'] = list(prices.values())
    result['gifts'].sort(key=lambda gift: (
        gift['costs'][0]['total_cost'] if gift['costs'] and
        gift['costs'][0]['total_cost'] is not None else float('inf')))
    return result
