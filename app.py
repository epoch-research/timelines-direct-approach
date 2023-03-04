import streamlit as st
from plots import plot
import timeline

from common import DistributionCI


# TODO:
# - force lower ci to be lower than higher
# - delay images being computed on first load

st.set_page_config(page_title='TAI Timelines: Direct Approach', layout="wide",
                   initial_sidebar_state="auto", menu_items={"Report a bug": 'mailto:david@epochai.org'})


def ci_input(label: str, distribution_type: str, interval: int, lower_default: float, upper_default: float):
    label_col, lower_bound, upper_bound = st.columns(3)
    with label_col:
        st.write(label + f' ({interval}% CI)')
    with lower_bound:
        lower = st.number_input("lower", key=f'{label}_lower', value=lower_default, format='%.7f')
    with upper_bound:
        upper = st.number_input("upper", key=f'{label}_upper', value=upper_default, format='%.7f')
    return DistributionCI(distribution_type, interval, lower, upper)


# Disable plus/minus buttons
st.markdown("""
<style>
    button.step-up {display: none;}
    button.step-down {display: none;}
    div[data-baseweb] {border-radius: 4px;}
</style>""", unsafe_allow_html=True)

with st.form("model_parameters"):
    st.subheader("Model Parameters")
    samples = st.number_input("Samples", min_value=1, value=int(1e2))

    st.subheader("Investment parameters")
    investment_input, investment_output = st.columns(2)
    with investment_input:
        current_gwp = st.number_input("Current GWP ($)", format='%e', value=1e14)
        current_max_spend = st.number_input("Current maximum training run spend ($)", format='%e', value=2e7)
        invest_growth_rate = ci_input('Investment growth rate', 'normal', 70, 1.1480341, 3.278781)
        gwp_growth_rate = ci_input('GWP growth rate', 'normal', 70, 0.0114741, 0.045136)
        max_gwp_pct = ci_input('Maximum GWP percentage', 'normal', 70, 0.0000083, 0.014047)

        investment_timeline = timeline.spending(
            samples=samples,
            growth_rate=invest_growth_rate,
            gwp_growth_rate=gwp_growth_rate,
            max_gwp_pct=max_gwp_pct,
            starting_gwp=current_gwp,
            starting_max_spend=current_max_spend,
        )
    with investment_output:
        investment_plot = plot(investment_timeline, 'log10(Largest Training Run ($))')
        st.pyplot(investment_plot)

    st.subheader("Compute parameters")
    physical_compute_input, physical_compute_output = st.columns(2)
    with physical_compute_input:
        gpu_dollar_cost = st.number_input("GPU dollar cost ($)", format='%e', value=500)
        transistors_per_core_limit = ci_input('Transistors per core limit', 'lognormal', 70, 0.896, 1.979)
        process_size_limit = ci_input('Process size limit', 'lognormal', 70, 1.396, 2.479)
        process_efficiency = ci_input('Process efficiency rate', 'lognormal', 95, 0.005, 0.01)
        hardware_specialization = ci_input('Hardware specialization rate', 'lognormal', 95, 0.005, 0.01)

        flops_per_dollar_timeline = timeline.flops_per_dollar(
            samples=samples,
            gpu_dollar_cost=gpu_dollar_cost,
            transistors_per_core_limit=transistors_per_core_limit,
            process_size_limit=process_size_limit,
            process_efficiency=process_efficiency,
            hardware_specialization=hardware_specialization,
        )

    with physical_compute_output:
        flops_per_dollar_plot = plot(flops_per_dollar_timeline, 'log10(FLOPs) per dollar ($)')
        print(flops_per_dollar_plot)
        st.pyplot(flops_per_dollar_plot)

    st.form_submit_button("Run")