"""
Runs the model without having to spin a server.
"""

import numpy as np
import timeline
import app

json_params = app.DEFAULT_PARAMS
params = app.make_json_params_callable(json_params)

timeline = app.generate_and_save_timeline_plots(params, output_dir='plots')

print('Probability of TAI by…')
for year, p in zip([2030, 2050, 2100], timeline['probabilities']):
    print(f'  {year}: {p}')

print()
print('Quantile')
for q, year in zip(['10%', 'Median', '90%'], timeline['quantiles']):
    print(f'  {q}: {year}')
