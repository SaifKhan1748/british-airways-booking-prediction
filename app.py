"""
British Airways – Task 2: Predicting Customer Buying Behaviour
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BA Booking Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Hide sidebar completely ───────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }
    .main-header {
        background: linear-gradient(135deg, #0D2137 0%, #065A82 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { color: white; font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #A0C4D8; margin: 0.5rem 0 0 0; font-size: 1rem; }
    .metric-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        text-align: center; border-left: 4px solid #065A82;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-card h2 { color: #065A82; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #64748B; margin: 0.2rem 0 0 0; font-size: 0.85rem; }
    .section-header {
        border-left: 4px solid #065A82; padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #F1F5F9; border-radius: 6px 6px 0 0;
        padding: 0.5rem 1.2rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>✈️ British Airways — Customer Booking Predictor</h1>
</div>
""", unsafe_allow_html=True)

# ── Fixed model constants ─────────────────────────────────────────────────────
N_TREES   = 200
MAX_DEPTH = 12
MIN_LEAF  = 10
N_FOLDS   = 5

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv("customer_booking.csv", encoding='latin1')
    except FileNotFoundError:
        st.error("⚠️ customer_booking.csv not found. Please place it in the same folder as app.py.")
        st.stop()

df_raw = load_data()

# ── Preprocessing ─────────────────────────────────────────────────────────────
@st.cache_data
def preprocess(df):
    df = df.copy()
    day_map = {'Mon':1,'Tue':2,'Wed':3,'Thu':4,'Fri':5,'Sat':6,'Sun':7}
    df['flight_day'] = df['flight_day'].map(day_map)
    le = LabelEncoder()
    df['sales_channel_enc']  = le.fit_transform(df['sales_channel'])
    df['trip_type_enc']      = le.fit_transform(df['trip_type'])
    df['booking_origin_enc'] = le.fit_transform(df['booking_origin'])
    df['route_enc']          = le.fit_transform(df['route'])
    df['total_extras']       = df['wants_extra_baggage'] + df['wants_preferred_seat'] + df['wants_in_flight_meals']
    df['is_weekend_flight']  = (df['flight_day'] >= 6).astype(int)
    df['is_long_haul']       = (df['flight_duration'] > 6).astype(int)
    df['booking_lead_bin']   = np.digitize(df['purchase_lead'], bins=[7, 30, 90])
    return df

df = preprocess(df_raw)

FEATURES = [
    'num_passengers','purchase_lead','length_of_stay','flight_hour','flight_day',
    'wants_extra_baggage','wants_preferred_seat','wants_in_flight_meals','flight_duration',
    'sales_channel_enc','trip_type_enc','booking_origin_enc','route_enc',
    'total_extras','is_weekend_flight','is_long_haul','booking_lead_bin'
]
FEATURE_LABELS = {
    'num_passengers':'No. Passengers','purchase_lead':'Purchase Lead (days)',
    'length_of_stay':'Length of Stay','flight_hour':'Flight Hour','flight_day':'Flight Day',
    'wants_extra_baggage':'Extra Baggage','wants_preferred_seat':'Preferred Seat',
    'wants_in_flight_meals':'In-Flight Meals','flight_duration':'Flight Duration',
    'sales_channel_enc':'Sales Channel','trip_type_enc':'Trip Type',
    'booking_origin_enc':'Booking Origin','route_enc':'Route',
    'total_extras':'★ Total Extras','is_weekend_flight':'★ Weekend Flight',
    'is_long_haul':'★ Long Haul','booking_lead_bin':'★ Booking Lead Bin'
}

X = df[FEATURES]
y = df['booking_complete']

# ── Train Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def train_model():
    rf = RandomForestClassifier(
        n_estimators=N_TREES, max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X, y)
    return rf

@st.cache_data
def run_cv():
    rf = RandomForestClassifier(
        n_estimators=N_TREES, max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    return cross_validate(rf, X, y, cv=cv,
        scoring=['accuracy','roc_auc','f1','precision','recall'],
        return_train_score=False, n_jobs=-1)

with st.spinner("🔄 Training model & running cross-validation..."):
    rf     = train_model()
    cv_res = run_cv()

y_pred      = rf.predict(X)
y_pred_prob = rf.predict_proba(X)[:, 1]
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
importances.index = [FEATURE_LABELS[f] for f in importances.index]

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🔍 EDA", "🤖 Model Performance", "📈 Feature Importance", "🎯 Predict"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header"><h3>Dataset Overview</h3></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h2>{len(df):,}</h2><p>Total Bookings</p></div>', unsafe_allow_html=True)
    with c2:
        rate = y.mean()*100
        st.markdown(f'<div class="metric-card"><h2>{rate:.1f}%</h2><p>Completion Rate</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h2>{len(FEATURES)}</h2><p>Features Used</p></div>', unsafe_allow_html=True)
    with c4:
        auc = cv_res['test_roc_auc'].mean()
        st.markdown(f'<div class="metric-card"><h2>{auc:.3f}</h2><p>ROC-AUC (5-Fold CV)</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Raw Data Preview")
        st.dataframe(df_raw.head(10), width='stretch')
    with col2:
        st.markdown("#### Column Info")
        info_df = pd.DataFrame({
            'Column': df_raw.columns.tolist(),
            'Type': df_raw.dtypes.astype(str).tolist(),
            'Nulls': df_raw.isnull().sum().tolist()
        })
        st.dataframe(info_df, width='stretch', height=380)

    st.markdown("---")
    st.markdown("#### Descriptive Statistics")
    st.dataframe(df_raw.describe().round(2), width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header"><h3>Exploratory Data Analysis</h3></div>', unsafe_allow_html=True)

    st.markdown("#### Target Variable: booking_complete")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        counts = y.value_counts()
        ax.bar(['Not Completed (0)', 'Completed (1)'], counts.values,
               color=['#EF4444','#10B981'], edgecolor='none', width=0.5)
        ax.set_ylabel('Count'); ax.set_title('Booking Completion Counts', fontweight='bold')
        for i, v in enumerate(counts.values):
            ax.text(i, v+200, f'{v:,}', ha='center', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.pie(counts.values, labels=['Not Completed','Completed'], autopct='%1.1f%%',
               colors=['#EF4444','#10B981'], startangle=90,
               wedgeprops={'edgecolor':'white','linewidth':2})
        ax.set_title('Class Split', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Numeric Feature Distributions")
    num_cols = ['purchase_lead','length_of_stay','flight_duration','flight_hour','num_passengers']
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    for ax, col in zip(axes, num_cols):
        ax.hist(df[col], bins=40, color='#065A82', alpha=0.85, edgecolor='none')
        ax.axvline(df[col].mean(), color='#F59E0B', linestyle='--', linewidth=1.5)
        ax.set_title(col.replace('_',' ').title(), fontweight='bold', fontsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Completion Rate by Sales Channel")
        fig, ax = plt.subplots(figsize=(5, 3))
        ct = df_raw.groupby('sales_channel')['booking_complete'].mean()*100
        ct.sort_values().plot(kind='barh', ax=ax, color='#065A82', edgecolor='none')
        ax.set_xlabel('Completion Rate (%)'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        st.markdown("#### Completion Rate by Trip Type")
        fig, ax = plt.subplots(figsize=(5, 3))
        ct2 = df_raw.groupby('trip_type')['booking_complete'].mean()*100
        ct2.sort_values().plot(kind='barh', ax=ax, color='#1C7293', edgecolor='none')
        ax.set_xlabel('Completion Rate (%)'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Add-On Selections vs Completion Rate")
    extras = ['wants_extra_baggage','wants_preferred_seat','wants_in_flight_meals']
    labels = ['Extra Baggage','Preferred Seat','In-Flight Meals']
    rates_yes = [df_raw[df_raw[e]==1]['booking_complete'].mean()*100 for e in extras]
    rates_no  = [df_raw[df_raw[e]==0]['booking_complete'].mean()*100 for e in extras]
    x_pos = np.arange(len(extras)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x_pos - w/2, rates_no,  w, label='No add-on',  color='#94A3B8', edgecolor='none')
    b2 = ax.bar(x_pos + w/2, rates_yes, w, label='With add-on', color='#065A82', edgecolor='none')
    ax.set_xticks(x_pos); ax.set_xticklabels(labels)
    ax.set_ylabel('Completion Rate (%)'); ax.legend()
    ax.set_title('Add-On Selections → Higher Completion Rate', fontweight='bold')
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                f'{bar.get_height():.1f}%', ha='center', fontsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header"><h3>Model Performance — 5-Fold Cross-Validation</h3></div>', unsafe_allow_html=True)

    metric_names  = ['accuracy','roc_auc','precision','recall','f1']
    metric_labels = ['Accuracy','ROC-AUC','Precision','Recall','F1 Score']
    means = [cv_res[f'test_{m}'].mean() for m in metric_names]
    stds  = [cv_res[f'test_{m}'].std()  for m in metric_names]
    colors_m = ['#065A82','#1C7293','#028090','#21295C','#10B981']

    cols = st.columns(5)
    for col, lbl, m, s in zip(cols, metric_labels, means, stds):
        with col:
            st.metric(lbl, f"{m:.4f}", f"±{s:.4f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### CV Scores per Fold")
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, col_c, lbl in zip(metric_names, colors_m, metric_labels):
            ax.plot(range(1, N_FOLDS+1), cv_res[f'test_{m}'], marker='o',
                    color=col_c, label=lbl, linewidth=2)
        ax.set_xticks(range(1, N_FOLDS+1))
        ax.set_xticklabels([f'Fold {i}' for i in range(1, N_FOLDS+1)])
        ax.set_ylabel('Score'); ax.legend(fontsize=8); ax.set_ylim(0, 1)
        ax.set_title('Score per CV Fold', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        st.markdown("#### Mean Scores ± Std Dev")
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(metric_labels, means, color=colors_m, edgecolor='none', width=0.6)
        ax.errorbar(metric_labels, means, yerr=stds, fmt='none', color='#374151', capsize=5, linewidth=2)
        ax.set_ylim(0, 1.15)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, m+0.03, f'{m:.3f}',
                    ha='center', fontweight='bold', fontsize=9)
        ax.set_title('Mean CV Metrics ± Std', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### ROC Curve")
        fpr, tpr, _ = roc_curve(y, y_pred_prob)
        auc_score = roc_auc_score(y, y_pred_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='#065A82', linewidth=2.5, label=f'AUC = {auc_score:.3f}')
        ax.plot([0,1],[0,1], color='gray', linestyle='--', linewidth=1)
        ax.fill_between(fpr, tpr, alpha=0.08, color='#065A82')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve', fontweight='bold'); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(cm, display_labels=['Not Complete','Complete'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title('Confusion Matrix', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Full Classification Report")
    report = classification_report(y, y_pred, target_names=['Not Complete','Complete'], output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3), width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header"><h3>Feature Importance Analysis</h3></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("#### All Feature Importances")
        fig, ax = plt.subplots(figsize=(7, 6))
        bar_colors = ['#F59E0B' if '★' in f else '#065A82' for f in importances.index[::-1]]
        ax.barh(importances.index[::-1], importances.values[::-1], color=bar_colors, edgecolor='none', height=0.7)
        ax.set_xlabel('Importance Score')
        ax.set_title('Random Forest Feature Importances', fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for i, (val, name) in enumerate(zip(importances.values[::-1], importances.index[::-1])):
            ax.text(val+0.002, i, f'{val:.3f}', va='center', fontsize=8)
        p1 = mpatches.Patch(color='#065A82', label='Original feature')
        p2 = mpatches.Patch(color='#F59E0B', label='Engineered ★')
        ax.legend(handles=[p1,p2], fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        st.markdown("#### Importance Table")
        imp_df = pd.DataFrame({'Feature': importances.index, 'Importance': importances.values.round(4),
                               'Cumulative %': (importances.values.cumsum()*100).round(1)})
        st.dataframe(imp_df, width='stretch', height=440)

    st.markdown("---")
    st.markdown("#### Cumulative Feature Importance")
    cumulative = importances.values.cumsum()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(range(1, len(cumulative)+1), cumulative*100, marker='o',
            color='#065A82', linewidth=2, markersize=5)
    ax.fill_between(range(1, len(cumulative)+1), cumulative*100, alpha=0.1, color='#065A82')
    ax.axhline(80, color='#EF4444', linestyle='--', linewidth=1.5, label='80% threshold')
    ax.axhline(95, color='#F59E0B', linestyle='--', linewidth=1.5, label='95% threshold')
    ax.set_xlabel('Number of Features'); ax.set_ylabel('Cumulative Importance (%)')
    ax.set_title('Cumulative Feature Importance', fontweight='bold'); ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Live Predictor
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header"><h3>🎯 Live Booking Completion Predictor</h3></div>', unsafe_allow_html=True)
    st.markdown("Fill in the passenger details below to predict whether they will complete a booking.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**✈️ Flight Details**")
        num_pax      = st.number_input("Number of Passengers", 1, 9, 2)
        flight_hour  = st.slider("Departure Hour", 0, 23, 10)
        flight_day   = st.selectbox("Flight Day", ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
        flight_dur   = st.number_input("Flight Duration (hours)", 0.5, 20.0, 4.0, step=0.5)
        trip_type    = st.selectbox("Trip Type", ['RoundTrip','OneWay','CircleTrip'])
    with col2:
        st.markdown("**📅 Booking Details**")
        purchase_lead = st.number_input("Days Before Travel (Purchase Lead)", 0, 365, 30)
        length_stay   = st.number_input("Length of Stay (nights)", 0, 60, 7)
        sales_channel = st.selectbox("Sales Channel", ['Internet','Mobile'])
    with col3:
        st.markdown("**🎒 Add-Ons**")
        extra_bag  = st.checkbox("Extra Baggage")
        pref_seat  = st.checkbox("Preferred Seat")
        meals      = st.checkbox("In-Flight Meals")

    st.markdown("---")
    if st.button("🔮 Predict Booking Completion", type="primary", use_container_width=True):
        day_map_pred = {'Mon':1,'Tue':2,'Wed':3,'Thu':4,'Fri':5,'Sat':6,'Sun':7}
        sc_map       = {'Internet':1,'Mobile':0}
        tt_map       = {'RoundTrip':1,'OneWay':0,'CircleTrip':2}

        total_extras = int(extra_bag) + int(pref_seat) + int(meals)
        is_weekend   = 1 if day_map_pred[flight_day] >= 6 else 0
        is_long      = 1 if flight_dur > 6 else 0
        lead_bin     = int(np.digitize(purchase_lead, bins=[7,30,90]))
        bo_enc       = df['booking_origin_enc'].median()
        rt_enc       = df['route_enc'].median()

        input_data = pd.DataFrame([{
            'num_passengers': num_pax,
            'purchase_lead': purchase_lead,
            'length_of_stay': length_stay,
            'flight_hour': flight_hour,
            'flight_day': day_map_pred[flight_day],
            'wants_extra_baggage': int(extra_bag),
            'wants_preferred_seat': int(pref_seat),
            'wants_in_flight_meals': int(meals),
            'flight_duration': flight_dur,
            'sales_channel_enc': sc_map.get(sales_channel, 1),
            'trip_type_enc': tt_map.get(trip_type, 1),
            'booking_origin_enc': bo_enc,
            'route_enc': rt_enc,
            'total_extras': total_extras,
            'is_weekend_flight': is_weekend,
            'is_long_haul': is_long,
            'booking_lead_bin': lead_bin,
        }])

        prob = rf.predict_proba(input_data[FEATURES])[0][1]
        pred = rf.predict(input_data[FEATURES])[0]

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            color   = "#10B981" if pred == 1 else "#EF4444"
            outcome = "✅ WILL COMPLETE" if pred == 1 else "❌ WON'T COMPLETE"
            st.markdown(f"""
            <div style="background:{color}20; border:2px solid {color}; border-radius:10px;
                        padding:1.5rem; text-align:center;">
                <h2 style="color:{color}; margin:0">{outcome}</h2>
                <p style="color:#374151; margin:0.5rem 0 0 0">Booking Prediction</p>
            </div>""", unsafe_allow_html=True)
        with col_r2:
            st.metric("Completion Probability", f"{prob:.1%}")
        with col_r3:
            st.metric("Non-Completion Probability", f"{1-prob:.1%}")

        st.markdown("---")
        st.markdown("**Key factors driving this prediction:**")
        fi_series = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False).head(5)
        for feat, imp in fi_series.items():
            lbl = FEATURE_LABELS[feat]
            val = input_data[feat].values[0]
            st.markdown(f"- **{lbl}** (importance: {imp:.3f}) → input value: `{val}`")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#94A3B8; font-size:0.8rem;'>"
    "British Airways Data Science · Forage Virtual Experience · Built with Streamlit & scikit-learn"
    "</p>",
    unsafe_allow_html=True
)
