# === Imports ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import ceil
from collections import defaultdict
import matplotlib.lines as mlines
import matplotlib as mpl
from scipy.stats import gaussian_kde
import re
from sklearn.cluster import KMeans
import seaborn as sns

# === Streamlit Config ===
st.set_page_config(page_title="Decision Tool for Near-Optimal Transition Pathways", layout="wide")

# === Custom Style (HTML) ===
st.markdown("""<style>
    body, .stApp { background-color: #f4f4f4; font-family: 'Segoe UI', sans-serif; }
    html, body, [class*="css"] { color: #2c2c2c; }
    .stContainer { background-color: #ffffff; border-radius: 16px; padding: 1.5rem;
                   box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05); }
    .stButton>button { background-color: #6e6e6e; color: white; border: none;
                       padding: 0.5em 1.2em; border-radius: 8px; font-weight: 500; }
    .stButton>button:hover { background-color: #5a5a5a; }
    .stSlider>div>div>div>div { background: #888888; }
    input[type=range]::-webkit-slider-thumb { background: #888888; }
    .stSlider [role="slider"] { background-color: #888888 !important; }
    input[type="text"] { background-color: #eaeaea; color: #2c2c2c;
                         border: 1px solid #cccccc; border-radius: 6px; padding: 0.4em; }
    input[type="text"]:focus { border-color: #888888; outline: none; }
</style>""", unsafe_allow_html=True)

# === Session State ===
def initialize_session_state():
    defaults = {
        'convex_combinations': pd.DataFrame(),
        'convex_additional': pd.DataFrame(),
        'show_convex': True,
        'show_convex_in_timeplot': True,
        'excel_loaded': False,
        'excel_path': '',
        'excel_error': None,
        'show_tech_info': False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

initialize_session_state()

# === Excel Laden ===
@st.cache_data
def load_excel_data(path):
    return pd.read_excel(path)

# === Konstanten ===
DEFAULT_FILENAME = "VERTEX_RESULTS.xlsx"
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_FILENAME)

st.title("🔬 Technology Decision Tool")

if not st.session_state["excel_loaded"]:
    st.subheader("📂 Upload Excel File")
    uploaded_file = st.file_uploader("Upload a .xlsx file", type=["xlsx"])

    if uploaded_file:
        st.subheader("🔀 Optional Clustering Before Analysis")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Read in all vertices from excel file"):
                try:
                    df = pd.read_excel(uploaded_file)
                    st.session_state["uploaded_excel"] = df.copy()
                    st.session_state["excel_loaded"] = True
                    st.rerun()
                except Exception as e:
                    st.session_state["excel_error"] = f"❌ Fehler beim Einlesen: {e}"
        with col2:
            k_value = st.number_input("Number of representative vertices to retain (KMeans)", 50, 5000, 1000, 50)
            if st.button("📊 Apply KMeans"):
                try:
                    df = pd.read_excel(uploaded_file)
                    coeff_cols = [c for c in df.columns if c.startswith("COEFF_")]
                    last_idx = df[df[coeff_cols[-1]] == -1].index.max()
                    df_head = df.loc[:last_idx]
                    df_tail = df.loc[last_idx + 1:].copy()

                    cluster_cols = [c for c in df.columns if c.startswith("VALUE_") or c.startswith("MAA_")]
                    df_tail_unique = df_tail.drop_duplicates(subset=cluster_cols)
                    k = k_value - len(df_head)
                    if len(df_tail_unique) > k:
                        X = df_tail_unique[cluster_cols].fillna(0).to_numpy()
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
                        df_tail_unique["cluster"] = kmeans.fit_predict(X)
                        idx = df_tail_unique.groupby("cluster").head(1).index
                        df_tail_clustered = df_tail.loc[idx]
                    else:
                        df_tail_clustered = df_tail_unique

                    df_final = pd.concat([df_head, df_tail_clustered], ignore_index=True)
                    st.session_state["uploaded_excel"] = df_final
                    st.session_state["excel_loaded"] = True
                    st.rerun()
                except Exception as e:
                    st.session_state["excel_error"] = f"❌ Fehler beim Clustern: {e}"

    if st.session_state["excel_error"]:
        st.error(st.session_state["excel_error"])
    st.stop()

vertex_df = st.session_state["uploaded_excel"]

# === Präfix-Erkennung ===
if any(col.startswith("VALUE_") for col in vertex_df.columns):
    MAA_PREFIX = "VALUE_"
    INSTALLED_CAPACITY_PREFIX = "INSTALLED_CAPACITY_"
    NEW_CAPACITY_PREFIX = "NEW_CAPACITY_"
elif any(col.startswith("MAA_") for col in vertex_df.columns):
    MAA_PREFIX = "MAA_"
    INSTALLED_CAPACITY_PREFIX = "MAA_INSTALLED_CAPACITY_"
    NEW_CAPACITY_PREFIX = "NEW_CAPACITY_"
else:
    st.error("❌ Could not detect expected prefixes ('VALUE_' or 'MAA_') in the Excel columns.")
    st.stop()

# === Technologien extrahieren ===
tech_cols = [col for col in vertex_df.columns if col.startswith(MAA_PREFIX)]
technologies = [col.replace(MAA_PREFIX, '') for col in tech_cols]
tech_data = vertex_df[tech_cols].drop_duplicates()
tech_indices = tech_data.index

# === Zeitreihen-Mapping ===
def extract_time_series_map(df):
    tech_time_map = defaultdict(list)
    seen = set()
    for col in df.columns:
        m1 = re.match(r"^(VALUE_[^[]+)\[\((\d{4}),?\)\]$", col)
        if m1:
            tech = m1.group(1).replace("VALUE_", "")
            year = int(m1.group(2))
            tech_time_map[tech].append((year, col))
            seen.add(col)
            continue
        m2 = re.match(r"^(.*)_(\d{4})$", col)
        if m2 and col not in seen:
            base, y = m2.groups()
            if "INSTALLED_CAPACITY_" in base:
                tech = base.split("INSTALLED_CAPACITY_")[-1]
                tech_time_map[tech].append((int(y), col))
    return tech_time_map

tech_time_map = extract_time_series_map(vertex_df)

# === Zusätzliche Metriken nach letzter NEW_CAPACITY_ Spalte ===
new_cap_indices = [i for i, c in enumerate(vertex_df.columns) if c.startswith(NEW_CAPACITY_PREFIX)]
last_idx = max(new_cap_indices) if new_cap_indices else -1
additional_cols = [
    c for c in vertex_df.columns[last_idx + 1:]
    if pd.api.types.is_numeric_dtype(vertex_df[c])
]

# === Sidebar Settings ===
st.sidebar.markdown("## ⚙️ Settings")

max_vertices = len(tech_data)
st.sidebar.number_input(
    "Max. number of vertices to show in plots:",
    min_value=1,
    max_value=max_vertices,
    value=min(5, max_vertices),
    key="max_plot_vertices"
)

st.sidebar.markdown("## 📊 Diagram Options")
st.sidebar.number_input(
    "Number of columns for installed capacity plots",
    min_value=1, max_value=5, step=1, key="n_cols_plots"
)
st.sidebar.checkbox(
    "Show convex combinations in plots", value=True, key="show_convex_in_timeplot"
)
st.sidebar.checkbox(
    "Show original flexibility ranges (red shaded)", value=False, key="show_original_ranges"
)

st.sidebar.markdown("### 📈 Density Plots")
st.sidebar.checkbox("Enable density plots", key="show_density")

st.sidebar.markdown("### 📌 Additional Metrics")
st.sidebar.radio(
    "Plot type for additional metrics",
    ["Violinplot", "Streudiagramm"],
    index=0,
    key="plot_type_selector"
)
st.sidebar.checkbox("Include convex combinations in metrics plot", value=True, key="include_convex_metrics")

# === Technologieauswahl und Sliderfilter ===
st.markdown("### 🧐 Select and Filter Technologies")

selected_techs = st.multiselect("Select technologies to constrain", technologies)
ordered_techs = selected_techs.copy()
filtered_indices = tech_data.index

for i, tech in enumerate(ordered_techs):
    key = f"slider_{tech}"
    col = f"{MAA_PREFIX}{tech}"

    # Nur zulässige Zeilen aus vorherigen Filtern
    subset = tech_data.loc[filtered_indices]
    for j in range(i):
        prev_tech = ordered_techs[j]
        prev_col = f"{MAA_PREFIX}{prev_tech}"
        prev_min, prev_max = st.session_state.get(f"slider_{prev_tech}", (subset[prev_col].min(), subset[prev_col].max()))
        subset = subset[(subset[prev_col] >= prev_min) & (subset[prev_col] <= prev_max)]

    valid_vals = subset[col].dropna()
    overall_min = tech_data[col].min()
    overall_max = tech_data[col].max()

    if valid_vals.empty:
        st.warning(f"⚠️ No valid values remaining for {tech}")
        st.slider(tech, float(overall_min), float(overall_max), (float(overall_min), float(overall_max)), disabled=True, key=key)
        continue

    min_val = valid_vals.min()
    max_val = valid_vals.max()
    if min_val == max_val:
        st.info(f"{tech}: constant value ({min_val:.2f})")
        st.session_state[key] = (min_val, max_val)
    else:
        value = st.session_state.get(key, (min_val, max_val))
        value = st.slider(tech, float(overall_min), float(overall_max), value, step=0.01, key=key)

        if value[0] < min_val or value[1] > max_val:
            st.warning(f"⚠️ Selection exceeds filtered range for {tech}. Resetting.")
            if key in st.session_state:
                del st.session_state[key]
            st.rerun()
        st.session_state[key] = value

    filtered_indices = subset[
        (subset[col] >= st.session_state[key][0]) & (subset[col] <= st.session_state[key][1])
    ].index

filtered_data = tech_data.loc[filtered_indices].reset_index(drop=True)

# === Konvexe Kombinationen Sidebar ===
st.sidebar.markdown("### ➕ Convex Combinations")
st.sidebar.number_input("Total number of convex combinations:", 10, 10000, 100, 10, key="n_samples")
n_vertices_available = len(filtered_indices)
st.sidebar.number_input("Vertices per combination:", 2, max(n_vertices_available, 2), min(3, n_vertices_available), 1, key="n_vertices_convex")
st.sidebar.selectbox("Dirichlet α (spread of weights):", [0.01, 0.1, 1.0], key="alpha_value")
st.sidebar.number_input("Combinations per batch:", 1, st.session_state["n_samples"], min(10, st.session_state["n_samples"]), 1, key="n_batch_size")

if st.sidebar.button("Generate"):
    base_data = tech_data.loc[filtered_indices]
    add_data = vertex_df.loc[filtered_indices, additional_cols] if additional_cols else pd.DataFrame(index=filtered_indices)
    installed_cols = [c for c in vertex_df.columns if c.startswith(INSTALLED_CAPACITY_PREFIX)]
    inst_data = vertex_df.loc[filtered_indices, installed_cols] if installed_cols else pd.DataFrame(index=filtered_indices)

    all_combos = []
    all_add = []
    all_inst = []

    batches = ceil(st.session_state["n_samples"] / st.session_state["n_batch_size"])
    for _ in range(batches):
        sample = base_data.sample(n=st.session_state["n_vertices_convex"], replace=False)
        weights = np.random.dirichlet([st.session_state["alpha_value"]] * len(sample), st.session_state["n_batch_size"])

        combo = weights @ sample.to_numpy()
        all_combos.append(pd.DataFrame(combo, columns=sample.columns))

        if not add_data.empty:
            combo_add = weights @ add_data.loc[sample.index].to_numpy()
            all_add.append(pd.DataFrame(combo_add, columns=add_data.columns))

        if not inst_data.empty:
            combo_inst = weights @ inst_data.loc[sample.index].to_numpy()
            all_inst.append(pd.DataFrame(combo_inst, columns=inst_data.columns))

    st.session_state["convex_combinations"] = pd.concat([st.session_state["convex_combinations"]] + all_combos, ignore_index=True)
    if all_add:
        st.session_state["convex_additional"] = pd.concat([st.session_state["convex_additional"]] + all_add, ignore_index=True)
    if all_inst:
        for col in inst_data.columns:
            st.session_state["convex_combinations"][col] = pd.concat(all_inst, ignore_index=True)[col]

    st.sidebar.success(f"{len(st.session_state['convex_combinations'])} combinations generated.")

if st.sidebar.button("Reset Convex"):
    st.session_state["convex_combinations"] = pd.DataFrame()
    st.session_state["convex_additional"] = pd.DataFrame()



# === Visualisierung: Zeitverläufe ===
st.markdown("### ⏳ Installed Capacities Over Time")
n_cols = st.session_state["n_cols_plots"]
plot_idx = filtered_indices[:st.session_state["max_plot_vertices"]]
n_rows = ceil(len(tech_time_map) / n_cols)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.5 * n_rows))
axs = axs.flatten()

for i, (tech, year_cols) in enumerate(sorted(tech_time_map.items())):
    if not year_cols:
        continue

    years, cols = zip(*sorted(year_cols, key=lambda x: x[0]))
    y = vertex_df.loc[plot_idx, list(cols)]
    ax = axs[i]
    ax.set_title(tech.replace('_', ' '))
    ax.set_xticks(years)
    ax.grid(True, linestyle="--", alpha=0.4)

    for _, row in y.iterrows():
        ax.plot(years, row.values, color=(0.1, 0.4, 0.8, 0.3))

    if st.session_state["show_convex_in_timeplot"] and not st.session_state["convex_combinations"].empty:
        convex_cols = [f"{INSTALLED_CAPACITY_PREFIX}{tech}_{year}" for year in years]
        if all(c in st.session_state["convex_combinations"].columns for c in convex_cols):
            y_c = st.session_state["convex_combinations"][convex_cols]
            for _, row in y_c.iterrows():
                ax.plot(years, row.values, color=(1.0, 0.3, 0.3, 0.3))

    ax.set_xlabel("Year")
    ax.set_ylabel("Installed Capacity")

# Leere Plots deaktivieren
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

fig.tight_layout()
st.pyplot(fig)

if st.session_state["show_density"]:
    st.markdown("### 🔥 Density Plots")

    n_cols_d = 2
    techs_d = [t for t, y in tech_time_map.items() if len(y) > 1]
    n_rows_d = ceil(len(techs_d) / n_cols_d)
    fig, axs = plt.subplots(n_rows_d, n_cols_d, figsize=(13, 4 * n_rows_d))
    axs = axs.flatten()

    base_df = vertex_df.loc[plot_idx]
    for i, tech in enumerate(techs_d):
        years, cols = zip(*sorted(tech_time_map[tech], key=lambda x: x[0]))
        df = base_df[list(cols)].dropna()
        if df.empty:
            axs[i].set_visible(False)
            continue

        points = []
        for row in df.itertuples(index=False):
            x_vals = np.array(years)
            y_vals = np.array(row)
            for j in range(len(x_vals) - 1):
                x_interp = np.linspace(x_vals[j], x_vals[j + 1], 3)
                y_interp = np.linspace(y_vals[j], y_vals[j + 1], 3)
                points.extend(zip(x_interp, y_interp))

        X, Y = np.meshgrid(np.linspace(min(years), max(years), 50),
                           np.linspace(0, df.max().max() + 1, 50))
        Z = gaussian_kde(np.array(points).T)(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        axs[i].contourf(X, Y, Z, levels=10, cmap="inferno")
        axs[i].set_title(tech.replace('_', ' '))
        axs[i].set_xlabel("Year")
        axs[i].set_ylabel("Installed Capacity")

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.tight_layout()
    st.pyplot(fig)

if additional_cols:
    st.markdown("### 📈 Additional Metrics")
    metrics_selected = st.multiselect("Select metrics to show", additional_cols, default=additional_cols)

    if metrics_selected:
        y_base = vertex_df.loc[filtered_indices, metrics_selected]
        y_convex = st.session_state["convex_additional"][metrics_selected] if not st.session_state["convex_additional"].empty else pd.DataFrame()

        if st.session_state["plot_type_selector"] == "Violinplot":
            fig, ax = plt.subplots(figsize=(len(metrics_selected) * 2.5, 6))
            data = [y_base[col].dropna().values for col in metrics_selected]
            vp = ax.violinplot(data, showmedians=True)
            ax.set_xticks(range(1, len(metrics_selected) + 1))
            ax.set_xticklabels([c.replace("NEW_CAPACITY_", "").replace("INSTALLED_CAPACITY_", "") for c in metrics_selected], rotation=45)
            ax.set_title("Distributions of Selected Metrics")
            st.pyplot(fig)
        else:  # Streudiagramm
            fig, ax = plt.subplots(figsize=(12, 4))
            for i, col in enumerate(metrics_selected):
                ax.scatter([i] * len(y_base), y_base[col], color="black", alpha=0.6, label="Original" if i == 0 else None)
                if not y_convex.empty:
                    ax.scatter([i] * len(y_convex), y_convex[col], color="red", alpha=0.4, label="Convex" if i == 0 else None)
                ax.set_xticks(range(len(metrics_selected)))
                ax.set_xticklabels(metrics_selected, rotation=45)
            ax.set_title("Scatterplot of Additional Metrics")
            ax.legend()
            st.pyplot(fig)

# === Tabellenansicht ===
st.markdown("### 📄 Show Remaining Vertices")

if filtered_data.empty:
    st.info("No valid vertices remain for the current selection.")
else:
    st.markdown("#### Original Vertices")
    df_full = vertex_df.loc[filtered_indices, :]
    st.dataframe(df_full, use_container_width=True)

if st.session_state["show_convex"] and not st.session_state["convex_combinations"].empty:
    st.markdown("#### Convex Combinations")
    df_convex = st.session_state["convex_combinations"]
    if not st.session_state["convex_additional"].empty:
        df_convex = pd.concat([df_convex, st.session_state["convex_additional"]], axis=1)
    st.dataframe(df_convex, use_container_width=True)
