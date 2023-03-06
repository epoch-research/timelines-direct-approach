import inspect

import pandas as pd
import streamlit as st
from plots import plot_timeline, plot_tai_requirements, plot_tai_timeline
import timeline
import numpy as np
from typing import Callable

from common import DistributionCI, YEARS, Timeline


# TODO:
# - tooltips for each param, with description
# - Titles for each plot
# - proper number formats for each param
# - import and export param settings json

st.set_page_config(
    page_title='TAI Timelines Direct Approach',
    layout='wide',
    menu_items={
        'Report a bug': 'mailto:david@epochai.org',
    }
)


def ci_input(label: str, timeline_func: Callable[..., Timeline], param_name: str):
    distribution_ci = inspect.signature(timeline_func).parameters[param_name].default

    interval_width = distribution_ci.interval_width
    lower_default, upper_default = distribution_ci.interval_min, distribution_ci.interval_max
    assert lower_default < upper_default, 'Lower bound on CI must be less than or equal to upper bound'

    label_col, lower_bound, upper_bound = st.columns(3)
    with label_col:
        st.write(label + f' ({distribution_ci.distribution}, {interval_width}% CI)')
    with lower_bound:
        lower = st.number_input("lower", key=f'{label}_lower', value=lower_default, format='%.7g')
    with upper_bound:
        upper = st.number_input("upper", key=f'{label}_upper', value=upper_default, format='%.7g')

    return DistributionCI(distribution_ci.distribution, interval_width, lower, upper)

# Disable plus/minus buttons
st.markdown("""
<style>
    button.step-up {display: none;}
    button.step-down {display: none;}
    div[data-baseweb] {border-radius: 4px;}
</style>""", unsafe_allow_html=True)

with st.form("model_parameters"):
    st.subheader('Model parameters')
    samples_col, ci_interval_col = st.columns(2)
    with samples_col:
        samples = st.number_input("Samples", min_value=1, value=int(5e2), format='%g')
    with ci_interval_col:
        errorbar_interval = st.number_input("PI interval", min_value=1, max_value=100, value=95, format='%g')

    st.subheader('TAI requirements')
    tai_requirements_input, tai_requirements_output = st.columns(2)
    with tai_requirements_input:
        slowdown = ci_input('Slowdown', timeline.tai_requirements, 'slowdown')
        log_k_performance = ci_input('log(k-performance)', timeline.tai_requirements, 'log_k_performance')

        tai_requirements = timeline.tai_requirements(
            samples=samples,
            slowdown=slowdown,
            log_k_performance=log_k_performance,
        )
    with tai_requirements_output:
        st.pyplot(plot_tai_requirements(tai_requirements, 'FLOPs required'))

    st.subheader('Algorithmic progress')
    algorithmic_progress_input, algorithmic_progress_output = st.columns(2)
    with algorithmic_progress_input:
        growth_rate = ci_input('CV growth rate', timeline.algorithmic_improvements, 'growth_rate')
        transfer_multiplier = ci_input('Transfer multiplier', timeline.algorithmic_improvements, 'transfer_multiplier')
        limit = ci_input('Limit multiplier', timeline.algorithmic_improvements, 'limit')

        algorithmic_progress_timeline = timeline.algorithmic_improvements(
            samples=samples,
            growth_rate=growth_rate,
            transfer_multiplier=transfer_multiplier,
            limit=limit,
        )
    with algorithmic_progress_output:
        st.pyplot(plot_timeline(algorithmic_progress_timeline, 'log10(Algorithmic progress multiplier)',
                                errorbar_interval=errorbar_interval))

    st.subheader("Investment")
    investment_input, investment_output = st.columns(2)
    with investment_input:
        default_current_gwp = inspect.signature(timeline.spending).parameters['starting_gwp'].default
        current_gwp = st.number_input("Current GWP ($)", format='%e', value=default_current_gwp)
        default_current_max_spend = inspect.signature(timeline.spending).parameters['starting_max_spend'].default
        current_max_spend = st.number_input("Current maximum training run spend ($)", format='%e',
                                            value=default_current_max_spend)
        invest_growth_rate = ci_input('Investment growth rate', timeline.spending, 'invest_growth_rate')
        gwp_growth_rate = ci_input('GWP growth rate', timeline.spending, 'gwp_growth_rate')
        max_gwp_pct = ci_input('Highest share of of GWP that can be spent', timeline.spending, 'max_gwp_pct')

        investment_timeline = timeline.spending(
            samples=samples,
            invest_growth_rate=invest_growth_rate,
            gwp_growth_rate=gwp_growth_rate,
            max_gwp_pct=max_gwp_pct,
            starting_gwp=current_gwp,
            starting_max_spend=current_max_spend,
        )
    with investment_output:
        st.pyplot(plot_timeline(investment_timeline, 'log10(Largest Training Run ($))',
                                errorbar_interval=errorbar_interval))

    st.subheader("Compute")
    physical_compute_input, physical_compute_output = st.columns(2)
    with physical_compute_input:
        default_gpu_dollar_cost = inspect.signature(timeline.flops_per_dollar).parameters['gpu_dollar_cost'].default
        gpu_dollar_cost = st.number_input("GPU dollar cost ($)", format='%e', value=default_gpu_dollar_cost)
        transistors_per_core_limit = ci_input('Transistors per core limit', timeline.flops_per_dollar,
                                              'transistors_per_core_limit')
        process_size_limit = ci_input('Process size limit', timeline.flops_per_dollar, 'process_size_limit')
        process_efficiency = ci_input('Process efficiency rate', timeline.flops_per_dollar, 'process_efficiency')
        hardware_specialization = ci_input('Hardware specialization rate', timeline.flops_per_dollar,
                                           'hardware_specialization')

        flops_per_dollar_timeline = timeline.flops_per_dollar(
            samples=samples,
            gpu_dollar_cost=gpu_dollar_cost,
            transistors_per_core_limit=transistors_per_core_limit,
            process_size_limit=process_size_limit,
            process_efficiency=process_efficiency,
            hardware_specialization=hardware_specialization,
        )
        physical_flops_timeline = flops_per_dollar_timeline + investment_timeline
        effective_compute_timeline = physical_flops_timeline + algorithmic_progress_timeline

    with physical_compute_output:
        st.pyplot(plot_timeline(flops_per_dollar_timeline, 'log10(FLOPs) per dollar ($)',
                                errorbar_interval=errorbar_interval))
        st.pyplot(plot_timeline(physical_flops_timeline, 'log10(Physical FLOPs)',
                                errorbar_interval=errorbar_interval))
        st.pyplot(plot_timeline(effective_compute_timeline, 'log10(Effective FLOPs)',
                                errorbar_interval=errorbar_interval))

    st.subheader('Combined Timeline')
    arrivals = effective_compute_timeline.T > tai_requirements
    tai_probabilities = np.sum(arrivals, axis=1) / samples

    tai_plot, tai_table = st.columns(2)
    with tai_plot:
        st.pyplot(plot_tai_timeline(tai_probabilities))
    with tai_table:
        st.dataframe(pd.DataFrame({'Probability': tai_probabilities}, index=list(map(str, YEARS))))

    st.form_submit_button("Run")