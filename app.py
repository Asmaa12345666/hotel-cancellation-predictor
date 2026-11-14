import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨", layout="wide")

LOW_CUT, HIGH_CUT = 0.2, 0.5  # same risk-level cut-offs as the notebook (section 13)
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


@st.cache_resource
def load_artifact():
    return joblib.load("model.joblib")


@st.cache_data
def get_choices(_art):
    """Read the allowed categories from the fitted preprocessing step,
    so the dropdowns always match what the model was trained on."""
    pre = _art["model"].named_steps["pre"]
    cat_cols = pre.transformers_[1][2]
    enc = pre.named_transformers_["cat"].named_steps["oh"]
    choices = {}
    for i, col in enumerate(cat_cols):
        cats = list(enc.categories_[i])
        rare = set(enc.infrequent_categories_[i]) if enc.infrequent_categories_[i] is not None else set()
        common = [c for c in cats if c not in rare and pd.notna(c)]
        choices[col] = (sorted(common), bool(rare))
    return choices


try:
    art = load_artifact()
    choices = get_choices(art)
except Exception as e:  # noqa: BLE001
    st.error("The model file could not be loaded. Make sure `model.joblib` is next to `app.py` "
             "and that scikit-learn has the same version used for training.")
    st.exception(e)
    st.stop()


def pick(col, label, container, default=None):
    """Selectbox from the model's own categories (+ 'Other' if rare ones were grouped)."""
    options, has_rare = choices[col]
    if has_rare:
        options = options + ["Other (rare)"]
    idx = options.index(default) if default in options else 0
    value = container.selectbox(label, options, index=idx)
    return "OTHER_RARE_VALUE" if value == "Other (rare)" else value


st.title("🏨 Hotel Booking Cancellation Predictor")
st.caption("Estimates the risk that a booking will be cancelled and suggests a deposit policy. "
           "Trained on ~87K bookings from two hotels in Portugal (2015–2017).")

with st.form("booking"):
    c1, c2, c3 = st.columns(3)

    hotel = pick("hotel", "Hotel", c1)
    lead_time = c1.number_input("Lead time (days between booking and arrival)", 0, 750, 60)
    month_name = c1.selectbox("Arrival month", MONTHS, index=6)
    nights_we = c1.number_input("Weekend nights", 0, 20, 1)
    nights_wk = c1.number_input("Week nights", 0, 35, 2)
    adults = c1.number_input("Adults", 0, 15, 2)
    children = c1.number_input("Children", 0, 10, 0)
    babies = c1.number_input("Babies", 0, 10, 0)

    market = pick("market_segment", "Market segment", c2, "Online TA")
    channel = pick("distribution_channel", "Distribution channel", c2, "TA/TO")
    deposit = pick("deposit_type", "Deposit type", c2, "No Deposit")
    customer = pick("customer_type", "Customer type", c2, "Transient")
    meal = pick("meal", "Meal plan", c2, "BB")
    room = pick("reserved_room_type", "Reserved room type", c2, "A")
    adr = c2.number_input("Price per night (ADR, €)", 0.0, 1000.0, 100.0, step=5.0)

    country = pick("country", "Guest country (code)", c3, "PRT")
    agent_options, agent_rare = choices["agent"]
    agent_list = ["No agent"] + [a for a in agent_options if a != "0"]
    if agent_rare:
        agent_list.append("Other (rare)")
    agent_sel = c3.selectbox("Travel agent ID", agent_list)
    has_company = c3.checkbox("Booked by a company")
    repeated = c3.checkbox("Repeated guest")
    prev_cancel = c3.number_input("Previous cancellations", 0, 30, 0)
    prev_ok = c3.number_input("Previous bookings not cancelled", 0, 80, 0)
    waiting = c3.number_input("Days on waiting list", 0, 400, 0)
    requests = c3.number_input("Special requests", 0, 5, 0)

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    # The model was trained on stays of at least one night with at least one guest.
    total_nights = nights_we + nights_wk
    total_guests = adults + children + babies
    if total_nights < 1:
        st.error("Please enter at least one night (weekend or week nights).")
        st.stop()
    if total_guests < 1:
        st.error("Please enter at least one guest.")
        st.stop()

    agent_value = {"No agent": "0", "Other (rare)": "OTHER_RARE_VALUE"}.get(agent_sel, agent_sel)

    row = dict(art["defaults"])  # safe fallback for any column not asked in the form
    row.update({
        "hotel": hotel, "lead_time": lead_time, "arrival_month": MONTHS.index(month_name) + 1,
        "arrival_date_week_number": int(MONTHS.index(month_name) * 4.35 + 2),
        "stays_in_weekend_nights": nights_we, "stays_in_week_nights": nights_wk,
        "adults": adults, "children": float(children), "babies": babies,
        "meal": meal, "country": country, "market_segment": market,
        "distribution_channel": channel, "is_repeated_guest": int(repeated),
        "previous_cancellations": prev_cancel, "previous_bookings_not_canceled": prev_ok,
        "reserved_room_type": room, "deposit_type": deposit, "agent": agent_value,
        "company": int(has_company), "days_in_waiting_list": waiting,
        "customer_type": customer, "adr": adr, "total_of_special_requests": requests,
        "total_nights": total_nights, "total_guests": float(total_guests),
        "has_kids": int(children + babies > 0),
    })

    try:
        X_in = pd.DataFrame([row])[art["columns"]]
        p = float(art["model"].predict_proba(X_in)[0, 1])
    except Exception as e:  # noqa: BLE001
        st.error("Something went wrong while scoring this booking. Please check the inputs.")
        st.exception(e)
        st.stop()

    st.divider()
    left, right = st.columns([1, 2])
    left.metric("Cancellation probability", f"{p:.0%}")
    if p < LOW_CUT:
        level = "Low"
        right.success("**Low risk** → no deposit needed.")
    elif p < HIGH_CUT:
        level = "Medium"
        right.warning("**Medium risk** → request a partial deposit.")
    else:
        level = "High"
        right.error("**High risk** → require a full deposit or confirm the booking by phone.")
    right.progress(min(max(p, 0.0), 1.0))
    st.caption(f"Risk level: {level}. Cut-offs: low < {LOW_CUT:.0%}, high ≥ {HIGH_CUT:.0%}. "
               "This is a statistical estimate from historical data, not a guarantee.")

with st.expander("About this model"):
    st.markdown(
        "- **Goal:** help a revenue-management team decide on overbooking and deposits.\n"
        "- **Model:** HistGradientBoosting inside a scikit-learn pipeline, tuned with cross-validation.\n"
        "- **Data:** Hotel Booking Demand (Kaggle, CC BY 4.0), two hotels in Portugal, 2015–2017. "
        "Results may not transfer to hotels in other countries.\n"
        "- **Scope:** stays of at least one night with at least one guest.\n"
        "- Using nationality or agent to set deposits can be unfair; a real deployment should review this."
    )
