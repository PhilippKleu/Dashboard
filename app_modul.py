# === Imports ===
import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from collections import defaultdict
import matplotlib.lines as mlines
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

# === Globale Konstanten ===
DEFAULT_FILENAME = "VERTEX_RESULTS.xlsx"
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_FILENAME)

# === Streamlit Seite konfigurieren ===
st.set_page_config(page_title="Decision Tool for Near-Optimal Transition Pathways", layout="wide")

# === UI-CSS Styling laden ===
def apply_custom_styles():
    st.markdown("""
        <style>
            body, .stApp { background-color: #f4f4f4; font-family: 'Segoe UI', sans-serif; }
            html, body, [class*="css"] { color: #2c2c2c; }
            .stContainer { background-color: #ffffff; border-radius: 16px; padding: 1.5rem;
                           box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05); }
            .stButton>button { background-color: #6e6e6e; color: white; border-radius: 8px;
                               font-weight: 500; padding: 0.5em 1.2em; border: none; }
            .stButton>button:hover { background-color: #5a5a5a; }
            .stSlider>div>div>div>div { background: #888888; }
            input[type=range]::-webkit-slider-thumb { background: #888888; }
            .stSlider [role="slider"] { background-color: #888888 !important; }
            input[type="text"] {
                background-color: #eaeaea; color: #2c2c2c; border-radius: 6px;
                padding: 0.4em; border: 1px solid #ccc;
            }
            input[type="text"]:focus { border-color: #888888; outline: none; }
        </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# === Session State initialisieren ===
def initialize_session_state():
    defaults = {
        'convex_combinations': pd.DataFrame(),
        'convex_additional': pd.DataFrame(),
        'show_convex': True,
        'show_convex_in_timeplot': True,
        'show_original_ranges': False,
        'show_density': False,
        'plot_type_selector': "Violinplot",
        'include_convex_metrics': True,
        'excel_loaded': False,
        'excel_path': '',
        'excel_error': None,
        'show_tech_info': False,
        'max_plot_vertices': 5,
        'n_cols_plots': 3,
        'n_samples': 100,
        'n_vertices_convex': 3,
        'n_batch_size': 10,
        'alpha_value': 0.1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

initialize_session_state()

# === Daten aus Excel laden ===
@st.cache_data
def load_excel_data(path):
    return pd.read_excel(path)

# === Technologien extrahieren ===
def extract_technologies(df, maa_prefix):
    maa_cols = [col for col in df.columns if col.startswith(maa_prefix)]
    techs = [col.replace(maa_prefix, '') for col in maa_cols]
    return techs, df[maa_cols].drop_duplicates(), maa_cols

# === Zeitreihen-Zuordnung für Technologien ===
def extract_time_series_map(df):
    tech_time_map = defaultdict(list)
    seen_columns = set()

    for col in df.columns:
        match_bracket = re.match(r"^(VALUE_[^[]+)\[\((\d{4}),?\)\]$", col)
        if match_bracket:
            tech = match_bracket.group(1).replace("VALUE_", "")
            year = int(match_bracket.group(2))
            tech_time_map[tech].append((year, col))
            seen_columns.add(col)
            continue

        match_suffix = re.match(r"^(.*)_(\d{4})$", col)
        if match_suffix and col not in seen_columns:
            base, year = match_suffix.groups()
            if "INSTALLED_CAPACITY_" in base:
                tech = base.split("INSTALLED_CAPACITY_")[-1]
                tech_time_map[tech].append((int(year), col))

    return tech_time_map

# === Weitere numerische Spalten nach NEW_CAPACITY finden ===
def get_additional_columns(df, new_capacity_prefix):
    indices = [i for i, col in enumerate(df.columns) if col.startswith(new_capacity_prefix)]
    if not indices:
        return []
    last_idx = max(indices)
    return [col for col in df.columns[last_idx + 1:] if pd.api.types.is_numeric_dtype(df[col])]

# === Datei-Upload & Vorverarbeitung ===
def upload_and_preprocess_excel():
    st.subheader("📂 Upload Excel File")
    uploaded_file = st.file_uploader("Upload a .xlsx file", type=["xlsx"])

    if uploaded_file:
        st.subheader("🔀 Optional Clustering Before Analysis")
        col_up1, col_up2 = st.columns(2)

        with col_up1:
            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                if st.button("📥 Read in all vertices from Excel file"):
                    try:
                        df = pd.read_excel(uploaded_file)
                        st.session_state["uploaded_excel"] = df.copy()
                        st.session_state["excel_loaded"] = True
                        st.session_state["excel_error"] = None
                        st.rerun()
                    except Exception as e:
                        st.session_state["excel_error"] = f"❌ Fehler beim Einlesen: {e}"

            with col_sub2:
                k_value = st.number_input(
                    "Number of representative vertices to retain (KMeans)",
                    min_value=50,
                    max_value=5000,
                    value=1000,
                    step=50,
                    key="clustering_k"
                )
                if st.button("📊 Apply KMeans to reduce vertex count"):
                    try:
                        df = pd.read_excel(uploaded_file)
                        clustered_df = apply_kmeans_clustering(df, k_value)
                        st.session_state["uploaded_excel"] = clustered_df
                        st.session_state["excel_loaded"] = True
                        st.session_state["excel_error"] = None
                        st.rerun()
                    except Exception as e:
                        st.session_state["excel_error"] = f"❌ Fehler beim Clustern: {e}"

    if st.session_state.get("excel_error"):
        st.error(st.session_state["excel_error"])
    st.stop()


def apply_kmeans_clustering(df, target_vertices):
    coeff_columns = [col for col in df.columns if col.startswith("COEFF_")]
    last_coeff_col = coeff_columns[-1] if coeff_columns else None

    if not last_coeff_col:
        raise ValueError("❌ Keine COEFF_-Spalten gefunden.")

    last_index = df[df[last_coeff_col] == -1].index.max()
    df_first_part = df.loc[:last_index].copy()
    df_remaining = df.loc[last_index + 1:].copy()

    cluster_cols = [col for col in df.columns if col.startswith("VALUE_") or col.startswith("MAA_")]
    df_unique = df_remaining.drop_duplicates(subset=cluster_cols)
    remaining_target = target_vertices - len(df_first_part)

    if len(df_unique) > remaining_target:
        X = df_unique[cluster_cols].fillna(0).to_numpy()
        kmeans = KMeans(n_clusters=remaining_target, random_state=42, n_init="auto")
        df_unique["cluster"] = kmeans.fit_predict(X)
        representatives = df_unique.groupby("cluster").head(1)
    else:
        representatives = df_unique

    return pd.concat([df_first_part, representatives], ignore_index=True)

# === Nach Upload: Daten vorbereiten ===
def prepare_uploaded_data():
    if "uploaded_excel" not in st.session_state:
        st.warning("⚠️ Please upload and (optionally) cluster an Excel file first.")
        st.stop()

    df = st.session_state["uploaded_excel"]

    # === Automatische Präfix-Erkennung ===
    if any(col.startswith("VALUE_") for col in df.columns):
        maa_prefix = "VALUE_"
        cap_prefix = "INSTALLED_CAPACITY_"
        new_prefix = "NEW_CAPACITY_"
    elif any(col.startswith("MAA_") for col in df.columns):
        maa_prefix = "MAA_"
        cap_prefix = "MAA_INSTALLED_CAPACITY_"
        new_prefix = "NEW_CAPACITY_"
    else:
        st.error("❌ Could not detect expected prefixes ('VALUE_' or 'MAA_') in the Excel columns.")
        st.stop()

    technologies, tech_data, maa_cols = extract_technologies(df, maa_prefix)
    tech_time_map = extract_time_series_map(df)
    additional_cols = get_additional_columns(df, new_prefix)

    return df, tech_data, technologies, maa_prefix, cap_prefix, new_prefix, tech_time_map, additional_cols



# === Globale Sidebar-Einstellungen ===
def setup_sidebar_controls(tech_data_len):
    st.sidebar.markdown("## ⚙️ Settings")
    st.sidebar.markdown("Maximum number of vertices to be displayed in plots:")

    max_plot_vertices = st.sidebar.number_input(
        "Max. number of vertices",
        min_value=1,
        max_value=tech_data_len,
        value=min(5, tech_data_len),
        step=1,
        key="max_plot_vertices"
    )

    st.sidebar.markdown("## 📊 Diagram Options")
    st.sidebar.number_input("Number of columns for plots", min_value=1, max_value=5, step=1, key="n_cols_plots")
    st.sidebar.checkbox("Show convex combinations in installed capacities", key="show_convex_in_timeplot")
    st.sidebar.checkbox("Show original flexibility ranges (red shaded)", value=False, key="show_original_ranges")

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



# === Auswahl und Filterung der Technologien ===
def technology_selection_ui(technologies, maa_prefix, tech_data):
    st.markdown("### 🧐 Select and Filter Technologies")

    # Info-Bereich zum Tool
    col_title, col_icon = st.columns([6, 2.5])
    with col_icon:
        if st.button("❓", key="show_info_button", help="Show/hide explanation"):
            st.session_state["show_tech_info"] = not st.session_state["show_tech_info"]

    if st.session_state.get("show_tech_info", False):
        display_tool_info()

    # Auswahl & Reset
    col_select, col_reset = st.columns([4, 1])
    with col_select:
        selected_techs = st.multiselect("Select variables to be constrained", technologies)
    with col_reset:
        if st.button("🔄 Reset"):
            reset_filter_sliders()

    return selected_techs


def display_tool_info():
    st.markdown(
        """
        ### 🧠 Technology Decision Tool – Overview

        **1. Filter**: Select technologies and constrain them step-by-step using sliders.
        **2. Convex Combinations**: Generate new scenarios as weighted vertex blends.
        **3. Visualizations**: Time series, density plots, additional metrics.
        **4. Results**: View all remaining scenarios in tables.

        👉 Use filters in sequence. Don't go back and change earlier sliders later.
        """, unsafe_allow_html=True
    )


def reset_filter_sliders():
    for key in list(st.session_state.keys()):
        if key.startswith("slider_"):
            del st.session_state[key]
    st.session_state["convex_combinations"] = pd.DataFrame()
    st.session_state["convex_additional"] = pd.DataFrame()
    st.rerun()



# === Filter mit Slidern anwenden ===
def apply_technology_filters(tech_data, selected_techs, maa_prefix):
    if not selected_techs:
        return tech_data.index, pd.DataFrame()

    filtered_cols = [maa_prefix + tech for tech in selected_techs]
    selected_data = tech_data[filtered_cols]
    current_indices = selected_data.index

    for i, tech in enumerate(selected_techs):
        col = maa_prefix + tech
        key = f"slider_{tech}"

        # Vorherige Filter berücksichtigen
        partial_indices = selected_data.index
        for j in range(i):
            prev_col = maa_prefix + selected_techs[j]
            min_val, max_val = st.session_state.get(f"slider_{selected_techs[j]}", (None, None))
            if min_val is not None:
                partial_indices = partial_indices[
                    (selected_data.loc[partial_indices, prev_col] >= min_val) &
                    (selected_data.loc[partial_indices, prev_col] <= max_val)
                ]

        valid_values = selected_data.loc[partial_indices, col].dropna()
        overall_min = selected_data[col].min()
        overall_max = selected_data[col].max()

        if any(f"slider_{selected_techs[j]}" not in st.session_state for j in range(i)):
            st.info(f"➡️ Please configure previous sliders to activate **{tech}**.")
            st.slider(tech, float(overall_min), float(overall_max),
                      (float(overall_min), float(overall_max)), key=key, disabled=True)
            continue

        if valid_values.empty:
            st.warning(f"⚠️ No valid vertices remaining for {tech}.")
            st.slider(tech, float(overall_min), float(overall_max),
                      (float(overall_min), float(overall_max)), key=key, disabled=True)
            continue

        min_val = valid_values.min()
        max_val = valid_values.max()
        if min_val == max_val:
            st.info(f"**{tech}**: No flexibility (constant value: {min_val:.2f})")
            st.session_state[key] = (min_val, max_val)
            current_indices = current_indices[(selected_data[col] == min_val)]
            continue

        slider_val = st.slider(
            tech,
            float(overall_min),
            float(overall_max),
            value=(float(min_val), float(max_val)),
            step=0.01,
            key=key
        )

        # Bereich prüfen
        clipped_range = (max(min_val, slider_val[0]), min(max_val, slider_val[1]))
        if slider_val != clipped_range:
            st.warning(f"⚠️ Selection for {tech} exceeds valid range ({min_val:.1f}–{max_val:.1f}). Resetting.")
            del st.session_state[key]
            st.rerun()

        st.session_state[key] = clipped_range
        current_indices = current_indices[
            (selected_data.loc[current_indices, col] >= clipped_range[0]) &
            (selected_data.loc[current_indices, col] <= clipped_range[1])
        ]

    return current_indices, selected_data.loc[current_indices]

# === Einheitliche Filterlogik für Vertex- oder Konvexdaten ===
def apply_tech_filters(data, session_state, ordered_techs, prefix):
    if data.empty:
        return pd.DataFrame()

    filtered_indices = data.index
    for tech in ordered_techs:
        key = f"slider_{tech}"
        col = f"{prefix}{tech}"
        if key in session_state and col in data.columns:
            min_val, max_val = session_state[key]
            filtered_indices = filtered_indices[
                (data.loc[filtered_indices, col] >= min_val) &
                (data.loc[filtered_indices, col] <= max_val)
            ]
    return data.loc[filtered_indices].reset_index(drop=True)

# === Sidebar-Einstellungen für konvexe Kombinationen ===
def setup_convex_combination_ui(max_vertices_available):
    st.sidebar.markdown("### ➕ Convex Combinations")

    st.sidebar.number_input(
        "Total number of convex combinations:",
        min_value=10,
        max_value=10000,
        value=100,
        step=10,
        key="n_samples"
    )

    st.sidebar.number_input(
        "Vertices per combination:",
        min_value=2,
        max_value=max_vertices_available if max_vertices_available > 1 else 2,
        value=min(5, max_vertices_available) if max_vertices_available > 1 else 2,
        step=1,
        key="n_vertices_convex"
    )

    st.sidebar.selectbox(
        "Dirichlet α (weight spread):",
        [0.01, 0.1, 1.0],
        index=[0.01, 0.1, 1.0].index(st.session_state.get('alpha_value', 0.1)),
        key="alpha_value"
    )

    st.sidebar.number_input(
        "Combinations per batch:",
        min_value=1,
        max_value=st.session_state["n_samples"],
        value=min(10, st.session_state["n_samples"]),
        step=1,
        key="n_batch_size"
    )

    col_gen, col_reset = st.sidebar.columns(2)
    generate = col_gen.button("Generate", key="generate_convex_button_sidebar")
    reset = col_reset.button("Reset", key="reset_convex_button_sidebar")

    return generate, reset


# === Konvexe Kombinationen generieren ===
def generate_convex_combinations(tech_data, vertex_df, selected_indices, additional_cols, maa_prefix, cap_prefix):
    base_data = tech_data.loc[selected_indices]
    base_additional = vertex_df.loc[selected_indices, additional_cols] if additional_cols else pd.DataFrame()

    installed_data = pd.DataFrame()
    if maa_prefix == "VALUE_":
        installed_cols = [col for col in vertex_df.columns if col.startswith(cap_prefix)]
        installed_data = vertex_df.loc[selected_indices, installed_cols]

    n_total = st.session_state["n_samples"]
    batch_size = st.session_state["n_batch_size"]
    n_vertices = st.session_state["n_vertices_convex"]
    alpha = st.session_state["alpha_value"]

    all_tech_samples = []
    all_additional_samples = []
    all_installed_samples = []

    for _ in range(ceil(n_total / batch_size)):
        sample = base_data.sample(n=n_vertices, random_state=np.random.randint(0, 99999))
        weights = np.random.dirichlet([alpha] * len(sample), size=batch_size)

        tech_comb = weights @ sample.to_numpy()
        all_tech_samples.append(pd.DataFrame(tech_comb, columns=sample.columns))

        if not base_additional.empty:
            additional_comb = weights @ base_additional.loc[sample.index].to_numpy()
            all_additional_samples.append(pd.DataFrame(additional_comb, columns=base_additional.columns))

        if not installed_data.empty:
            inst_comb = weights @ installed_data.loc[sample.index].to_numpy()
            all_installed_samples.append(pd.DataFrame(inst_comb, columns=installed_data.columns))

        if sum(len(df) for df in all_tech_samples) >= n_total:
            break

    # Zusammenfügen
    convex_df = pd.concat(all_tech_samples, ignore_index=True)
    if not installed_data.empty:
        convex_df[installed_data.columns] = pd.concat(all_installed_samples, ignore_index=True)

    st.session_state['convex_combinations'] = pd.concat(
        [st.session_state.get('convex_combinations', pd.DataFrame()), convex_df],
        ignore_index=True
    )

    if all_additional_samples:
        st.session_state['convex_additional'] = pd.concat(
            [st.session_state.get('convex_additional', pd.DataFrame()), pd.concat(all_additional_samples, ignore_index=True)],
            ignore_index=True
        )

    st.sidebar.success(f"✅ Generated {len(convex_df)} convex combinations.")


# === Reset convex combinations ===
def reset_convex_combinations():
    st.session_state['convex_combinations'] = pd.DataFrame()
    st.session_state['convex_additional'] = pd.DataFrame()


# === Zeitverlauf: VALUE_ Variablen ===
def plot_operational_variables_over_time(vertex_df, tech_time_map, current_indices, convex_data, maa_prefix, show_convex, show_ranges, max_vertices, n_cols):
    st.markdown("### Operational Variables Over Time")
for tech, year_cols in sorted(tech_time_map.items()):
    
    n_techs = sum(1 for v in value_time_map.values() if len(v) >= 1)
    n_rows = ceil(n_techs / n_cols)
    fig_width = 6 * n_cols
    fig_height = 3.5 * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('#f4f4f4')
    axes = axes.flatten() if n_techs > 1 else [axes]

    indices_to_plot = np.random.choice(current_indices, size=min(max_vertices, len(current_indices)), replace=False)

    plot_idx = 0
    for tech, year_cols in sorted(tech_time_map.items()):
        if len(year_cols) < 1:
            continue

        years_cols_sorted = sorted(year_cols, key=lambda x: x[0])
        years = [y for y, _ in years_cols_sorted]
        cols = [col for _, col in years_cols_sorted if col.startswith(maa_prefix + tech)]

        if not cols:
            continue

        vertex_subset = vertex_df.loc[indices_to_plot, cols]
        full_range = vertex_df.loc[current_indices, cols]
        ax = axes[plot_idx]
        ax.set_facecolor('#f0f0f0')

        for i in vertex_subset.index:
            values = vertex_subset.loc[i].values
            if not np.isnan(values).all():
                ax.plot(years, values, color=(0.1, 0.4, 0.8, 0.3))

        # Konvexe Kombinationen
        if show_convex and not convex_data.empty:
            if all(col in convex_data.columns for col in cols):
                for i in range(len(convex_data)):
                    values = convex_data.loc[i, cols].values
                    if not np.isnan(values).all():
                        ax.plot(years, values, color=(1.0, 0.3, 0.3, 0.3))

        # Min/Max-Bereich
        min_vals = full_range.min()
        max_vals = full_range.max()
        ax.fill_between(years, min_vals, max_vals, color=(0.1, 0.4, 0.8, 0.15))

        if show_ranges:
            original_vals = vertex_df.loc[:, cols]
            ax.fill_between(years, original_vals.min(), original_vals.max(), color=(1.0, 0.0, 0.0, 0.08))

        ax.set_title(tech.replace('_', ' ').title())
        ax.set_xticks(years)
        if plot_idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Year")
        if plot_idx % n_cols == 0:
            ax.set_ylabel("VALUE_")
        ax.grid(True, linestyle="--", alpha=0.4)

        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])

    st.pyplot(fig)



# === Zeitverlauf: INSTALLED_CAPACITY Variablen ===
def plot_installed_capacity_over_time(vertex_df, tech_time_map, current_indices, convex_data, cap_prefix, show_convex, show_ranges, max_vertices, n_cols):
    st.markdown("### ⏳ Installed Capacities Over Time")

    n_techs = len(tech_time_map)
    n_rows = ceil(n_techs / n_cols)
    fig_width = 6 * n_cols
    fig_height = 3.5 * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('#f4f4f4')
    axes = axes.flatten() if n_techs > 1 else [axes]

    indices_to_plot = np.random.choice(current_indices, size=min(max_vertices, len(current_indices)), replace=False)

    plot_idx = 0
    for tech, year_cols in sorted(tech_time_map.items()):
        if len(year_cols) < 1:
            continue

        years_cols_sorted = sorted(year_cols, key=lambda x: x[0])
        years = [y for y, _ in years_cols_sorted]
        cols = [col for _, col in years_cols_sorted]

        data_subset = vertex_df.loc[indices_to_plot, cols]
        full_range = vertex_df.loc[current_indices, cols]

        ax = axes[plot_idx]
        ax.set_facecolor('#f0f0f0')

        for i in data_subset.index:
            values = data_subset.loc[i].values
            ax.plot(years, values, color=(0.1, 0.4, 0.8, 0.3))

        # Konvexe Kombinationen
        if show_convex and not convex_data.empty:
            convex_cols = [f"{cap_prefix}{tech}_{year}" for year in years]
            if all(col in convex_data.columns for col in convex_cols):
                for i in range(len(convex_data)):
                    values = convex_data.loc[i, convex_cols].values
                    if not np.isnan(values).all():
                        ax.plot(years, values, color=(1.0, 0.3, 0.3, 0.3))

        # Min/Max-Füllung
        ax.fill_between(years, full_range.min(), full_range.max(), color=(0.1, 0.4, 0.8, 0.15))

        if show_ranges:
            orig_vals = vertex_df.loc[:, cols]
            ax.fill_between(years, orig_vals.min(), orig_vals.max(), color=(1.0, 0.0, 0.0, 0.08))

        ax.set_title(tech.replace('_', ' ').title())
        ax.set_xticks(years)
        if plot_idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Year")
        if plot_idx % n_cols == 0:
            ax.set_ylabel("Installed Capacity")
        ax.grid(True, linestyle="--", alpha=0.4)

        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])

    st.pyplot(fig)



# === Dichteplots: Entwicklungstypen über Zeit ===
def plot_density_kde_over_time(vertex_df, tech_time_map, current_indices, max_vertices_for_density=250):
    st.markdown("### 📊 Density Patterns of Installed Capacity Over Time")

    num_interp = 3
    grid_density = 50
    color_levels = 10

    techs = [tech for tech in tech_time_map if len(tech_time_map[tech]) > 1]
    n_techs = len(techs)
    n_cols = 2
    n_rows = ceil(n_techs / n_cols)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(13, 4 * n_rows))
    axs = axs.flatten()
    fig.patch.set_facecolor('#f4f4f4')

    df_base = vertex_df.loc[current_indices]
    if len(df_base) > max_vertices_for_density:
        df_base = df_base.sample(n=max_vertices_for_density, random_state=42)

    for i, tech in enumerate(techs):
        year_cols = sorted(tech_time_map[tech], key=lambda x: x[0])
        years = [y for y, _ in year_cols]
        cols = [col for _, col in year_cols]

        if not all(col in df_base.columns for col in cols):
            axs[i].set_visible(False)
            continue

        df = df_base[cols]
        if df.dropna(how='all').empty:
            axs[i].set_visible(False)
            continue

        axs[i].set_facecolor('#f0f0f0')

        x_vals = np.array(years)
        all_points = []
        for row in df.itertuples(index=False):
            y_vals = np.array(row)
            if np.isnan(y_vals).any():
                continue
            for j in range(len(x_vals) - 1):
                x_interp = np.linspace(x_vals[j], x_vals[j + 1], num_interp)
                y_interp = np.linspace(y_vals[j], y_vals[j + 1], num_interp)
                all_points.extend(zip(x_interp, y_interp))

        if not all_points:
            axs[i].set_visible(False)
            continue

        X, Y = np.meshgrid(
            np.linspace(min(years), max(years), grid_density),
            np.linspace(-0.5, df.max().max() + 0.5, grid_density)
        )

        Z = np.reshape(
            gaussian_kde(np.array(all_points).T, bw_method=0.1)(np.vstack([X.ravel(), Y.ravel()])),
            X.shape
        )

        contour = axs[i].contourf(X, Y, Z, levels=color_levels, cmap="inferno", extend="both")
        axs[i].set_title(tech.replace("_", " ").title())
        axs[i].set_xlabel("Year")
        axs[i].set_ylabel("Installed Capacity")
        axs[i].set_ylim(-1, df.max().max() * 1.05)
        axs[i].grid(True, linestyle="--", alpha=0.4)

        cbar = plt.colorbar(contour, ax=axs[i], label="Density")
        cbar.set_ticks(np.linspace(np.nanmin(Z), np.nanmax(Z), 4))
        cbar.set_ticklabels([f"{val:.3f}" for val in np.linspace(np.nanmin(Z), np.nanmax(Z), 4)])

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.subplots_adjust(top=0.95, bottom=0.07, hspace=0.44, wspace=0.3)
    st.pyplot(fig)


# === Weitere Metriken darstellen ===
def plot_additional_metrics(vertex_df, tech_data, additional_cols, selected_indices, convex_data, convex_additional, mode="Violinplot", show_convex=True):
    st.markdown("### 📈 Additional Metrics")

    selected_metrics = st.multiselect("📌 Select additional metrics to visualize", additional_cols, default=additional_cols)

    if not selected_metrics:
        st.info("Please select at least one metric to visualize.")
        return

    original_data = vertex_df.loc[tech_data.index, selected_metrics]
    filtered_original = original_data.loc[selected_indices]

    if show_convex and not convex_additional.empty:
        combined = pd.concat([filtered_original, convex_additional[selected_metrics]], axis=0)
    else:
        combined = filtered_original

    if mode == "Violinplot":
        fig, ax = plt.subplots(figsize=(len(selected_metrics) * 2.5, 6))
        ax.set_facecolor('#f0f0f0')

        data = [combined[col].dropna().values for col in selected_metrics]
        positions = list(range(len(selected_metrics)))

        vp = ax.violinplot(data, positions=positions, showmeans=False, showmedians=True, showextrema=True, widths=0.8)

        for pc in vp['bodies']:
            pc.set_facecolor('#444444')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        if 'cmedians' in vp:
            vp['cmedians'].set_color('black')

        ax.set_xticks(positions)
        ax.set_xticklabels([col.replace("NEW_CAPACITY_", "").replace("INSTALLED_CAPACITY_", "") for col in selected_metrics], rotation=45, ha="right")
        ax.set_ylabel("Metric Value")
        ax.set_title("Distributions of Selected Additional Metrics")
        ax.grid(True, linestyle="--", alpha=0.4)

        st.pyplot(fig)

    elif mode == "Streudiagramm":
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_facecolor('#f0f0f0')

        y_max = max([combined[col].max() for col in selected_metrics])
        for i, col in enumerate(selected_metrics):
            values = filtered_original[col].dropna().values
            ax.scatter([i] * len(values), values, alpha=0.7, color="#444444", label="Original" if i == 0 else None)

            if show_convex and not convex_additional.empty:
                c_vals = convex_additional[col].dropna().values
                ax.scatter([i] * len(c_vals), c_vals, alpha=0.5, color="#ff4444", label="Convex" if i == 0 else None)

            global_min = original_data[col].min()
            global_max = original_data[col].max()
            ax.fill_between([i - 0.3, i + 0.3], global_min, global_max, color='gray', alpha=0.15)

        ax.set_xticks(range(len(selected_metrics)))
        ax.set_xticklabels([col.replace("NEW_CAPACITY_", "").replace("INSTALLED_CAPACITY_", "") for col in selected_metrics], rotation=45, ha="right")
        ax.set_ylabel("Metric Value")
        ax.set_title("Scatter View of Selected Additional Metrics")
        ax.set_ylim(0, y_max * 1.1)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

        st.pyplot(fig)


# === Ergebnis-Tabellen anzeigen ===
def display_result_tables(vertex_df, tech_data, current_indices, maa_prefix, cap_prefix, additional_cols, convex_data, convex_additional, show_convex=True):
    st.markdown("### 📄 Show Remaining Vertices as Table")

    if len(current_indices) == 0 and (convex_data.empty or not show_convex):
        st.info("No valid vertices available for the current selection.")
        return

    # === Original Vertices ===
    st.markdown("#### Original Vertices")
    filtered_tech = tech_data.loc[current_indices] if not current_indices.empty else pd.DataFrame()
    result_frames = [filtered_tech.reset_index(drop=True)]

    if maa_prefix == "VALUE_":
        installed_cols = [col for col in vertex_df.columns if col.startswith(cap_prefix)]
        result_frames.append(vertex_df.loc[filtered_tech.index, installed_cols].reset_index(drop=True))

    if additional_cols:
        result_frames.append(vertex_df.loc[filtered_tech.index, additional_cols].reset_index(drop=True))

    if result_frames:
        st.dataframe(pd.concat(result_frames, axis=1), use_container_width=True)

    # === Convex Combinations ===
    if show_convex and not convex_data.empty:
        st.markdown("---")
        st.markdown("#### Convex Combinations")

        convex_frames = [convex_data.reset_index(drop=True)]

        if additional_cols and not convex_additional.empty:
            convex_frames.append(convex_additional[additional_cols].reset_index(drop=True))

        st.dataframe(pd.concat(convex_frames, axis=1), use_container_width=True)



def main():
    st.title("🔬 Technology Decision Tool")

    if not st.session_state.get("excel_loaded"):
        upload_and_preprocess_excel()

    # === Daten vorbereiten ===
    vertex_df, tech_data, technologies, maa_prefix, cap_prefix, new_prefix, tech_time_map, additional_cols = prepare_uploaded_data()
    setup_sidebar_controls(len(tech_data))

    selected_techs = technology_selection_ui(technologies, maa_prefix, tech_data)
    current_indices, filtered_data = apply_technology_filters(tech_data, selected_techs, maa_prefix)

    # === Konvexe Kombinationen UI & Berechnung ===
    generate, reset = setup_convex_combination_ui(len(current_indices))
    if reset:
        reset_convex_combinations()
    if generate and len(current_indices) > 1:
        generate_convex_combinations(tech_data, vertex_df, current_indices, additional_cols, maa_prefix, cap_prefix)

    filtered_convex = apply_tech_filters(st.session_state['convex_combinations'], st.session_state, selected_techs, maa_prefix)
    filtered_convex_add = st.session_state.get('convex_additional', pd.DataFrame())
    if not filtered_convex_add.empty and not filtered_convex.empty:
        filtered_convex_add = filtered_convex_add.loc[filtered_convex.index]

    # === Visualisierungen ===
    if maa_prefix == "VALUE_":
        plot_operational_variables_over_time(vertex_df, tech_time_map, current_indices, filtered_convex, maa_prefix,
                                             st.session_state["show_convex_in_timeplot"], st.session_state["show_original_ranges"],
                                             st.session_state["max_plot_vertices"], st.session_state["n_cols_plots"])

    plot_installed_capacity_over_time(vertex_df, tech_time_map, current_indices, filtered_convex, cap_prefix,
                                      st.session_state["show_convex_in_timeplot"], st.session_state["show_original_ranges"],
                                      st.session_state["max_plot_vertices"], st.session_state["n_cols_plots"])

    if st.session_state.get("show_density"):
        plot_density_kde_over_time(vertex_df, tech_time_map, current_indices)

    if additional_cols:
        plot_additional_metrics(vertex_df, tech_data, additional_cols, current_indices, filtered_convex,
                                filtered_convex_add, st.session_state["plot_type_selector"],
                                st.session_state["include_convex_metrics"])

    display_result_tables(vertex_df, tech_data, current_indices, maa_prefix, cap_prefix, additional_cols,
                          filtered_convex, filtered_convex_add, st.session_state["show_convex"])


if __name__ == "__main__":
    main()
