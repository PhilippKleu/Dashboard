# tech_decision_tool.py

# === Konfiguration und Imports ===
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
from io import BytesIO
from zipfile import ZipFile


DEFAULT_FILENAME = "VERTEX_RESULTS.xlsx"
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_FILENAME)

# === Streamlit Style ===
st.set_page_config(page_title="Decision Tool for Near-Optimal Transition Pathways", layout="wide")
st.markdown("""
    <style>
        body, .stApp {
            background-color: #f4f4f4;
            font-family: 'Segoe UI', sans-serif;
        }
        html, body, [class*="css"] {
            color: #2c2c2c;
        }
        .stContainer {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        }
        .stButton>button {
            background-color: #6e6e6e;
            color: white;
            border: none;
            padding: 0.5em 1.2em;
            border-radius: 8px;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #5a5a5a;
        }
        .stSlider>div>div>div>div {
            background: #888888;
        }
        input[type=range]::-webkit-slider-thumb {
            background: #888888;
        }
        .stSlider [role="slider"] {
            background-color: #888888 !important;
        }
        .stSelectbox, .stMultiSelect, .stSlider, .stButton {
            font-size: 0.95rem;
        }
        .stDataFrame, .css-1d391kg {
            border-radius: 10px;
            overflow: hidden;
        }
        .stProgress > div {
            margin-bottom: 2em;
        }
        .element-container svg {
            background-color: #f0f0f0 !important;
        }
        input[type="text"] {
            background-color: #eaeaea;
            color: #2c2c2c;
            border: 1px solid #cccccc;
            border-radius: 6px;
            padding: 0.4em;
        }
        input[type="text"]:focus {
            border-color: #888888;
            outline: none;
        }
    </style>
""", unsafe_allow_html=True)

# === Initialisiere Session State ===
def initialize_session_state():
    defaults = {
        'convex_combinations': pd.DataFrame(),
        'convex_additional': pd.DataFrame(),
        'show_convex': True,
        'excel_loaded': False,
        'excel_path': '',
        'excel_error': None,
        'n_cols_plots': 3,  # <--- HIER Standardwert für Plot-Spaltenanzahl
        'max_plot_vertices': 5,  # optional auch gleich hier
        "column_ratio" : 0.5,
        'layout_mode': "Two-column layout",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

initialize_session_state()

# === Lade Excel Datei ===
@st.cache_data
def load_excel_data(path):
    return pd.read_excel(path)

# === Extrahiere Technologien aus DataFrame ===
def extract_technologies(df):
    maa_cols = [col for col in df.columns if col.startswith(MAA_PREFIX)]
    techs = [col.replace(MAA_PREFIX, '') for col in maa_cols]
    return techs, df[maa_cols].drop_duplicates(), maa_cols

# === Extrahiere Zeitreihe pro Technologie ===
def extract_time_series_map(df, maa_prefix, mode="installed"):
    """
    Extrahiert eine Zeitreihen-Mapping-Technologie → [(Jahr, Spaltenname)].

    :param df: Pandas DataFrame
    :param maa_prefix: "VALUE_" oder "MAA_"
    :param mode: "installed" oder "operational"
    :return: Dictionary {tech: [(year, column)]}
    """
    tech_time_map = defaultdict(list)
    seen_keys = set()

    for col in df.columns:
        # === Für MAA-Daten: alle INSTALLED_CAPACITY_* Spalten zulassen ===
        if maa_prefix == "MAA_":
            match = re.match(r"^MAA_INSTALLED_CAPACITY_(.+)_(\d{4})$", col)
            if match:
                tech, year = match.groups()
                key = (tech, int(year))
                if key not in seen_keys:
                    tech_time_map[tech].append((int(year), col))
                    seen_keys.add(key)

        # === Für VALUE-Daten: abhängig vom Modus ===
        elif maa_prefix == "VALUE_":
            if mode == "operational":
                # Format: VALUE_Tech[(2025,)]
                match = re.match(r"^(VALUE_)([^[]+)\[\((\d{4}),?\)\]$", col)
                if match:
                    _, tech, year = match.groups()
                    key = (tech, int(year))
                    if key not in seen_keys:
                        tech_time_map[tech].append((int(year), col))
                        seen_keys.add(key)
            elif mode == "installed":
                # Format: INSTALLED_CAPACITY_Tech_2025
                match = re.match(r"^(INSTALLED_CAPACITY_)(.+)_(\d{4})$", col)
                if match:
                    _, tech, year = match.groups()
                    key = (tech, int(year))
                    if key not in seen_keys:
                        tech_time_map[tech].append((int(year), col))
                        seen_keys.add(key)

    # Sortiere Jahre innerhalb jeder Technologie
    return {
        tech: sorted(entries, key=lambda x: x[0])
        for tech, entries in tech_time_map.items()
    }

# === Bestimme zusätzliche Metrikspalten ===
def get_additional_columns(df):
    new_cap_indices = [i for i, col in enumerate(df.columns) if col.startswith(NEW_CAPACITY_PREFIX)]
    if not new_cap_indices:
        return []
    last_new_cap_idx = max(new_cap_indices)
    return [
        col for col in df.columns[last_new_cap_idx + 1:]
        if pd.api.types.is_numeric_dtype(df[col])
    ]

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

# === Titel & Initialisierung ===
st.title("🔬 Technology Decision Tool")
# === Excel-Datei Ladebereich via Upload ===
if not st.session_state.get("excel_loaded", False):
    st.subheader("📂 Upload Excel File")
    uploaded_file = st.file_uploader("Upload a .xlsx file", type=["xlsx"])

    if uploaded_file is not None:
        st.subheader("🔀 Optional Clustering Before Analysis")
        st.markdown("Choose Option:")

        col_up1, col_up2 = st.columns(2)

        with col_up1:
            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                if st.button("📥 Read in all vertices from excel file"):
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
                if st.button("📊 Apply KMeans to Excel vertices to retain a set number of representatives."):
                    try:
                        df = pd.read_excel(uploaded_file)
                        amount_vertices_remaining = int(k_value)
    
                        coeff_columns = [col for col in df.columns if col.startswith("COEFF_")]
                        last_coeff_col = coeff_columns[-1] if coeff_columns else None
    
                        if not last_coeff_col:
                            raise ValueError("❌ Keine COEFF_-Spalten gefunden.")
    
                        last_index_with_minus1 = df[df[last_coeff_col] == -1].index.max()
                        df_first_part = df.loc[:last_index_with_minus1].copy()
                        df_remaining = df.loc[last_index_with_minus1 + 1:].copy()
    
                        cluster_columns = [col for col in df.columns if col.startswith("VALUE_") or col.startswith("MAA_")]
                        df_remaining_unique = df_remaining.drop_duplicates(subset=cluster_columns)
                        remaining_target = amount_vertices_remaining - len(df_first_part)
    
                        if len(df_remaining_unique) > remaining_target:
                            X = df_remaining_unique[cluster_columns].fillna(0).to_numpy()
                            kmeans = KMeans(n_clusters=remaining_target, random_state=42, n_init="auto")
                            df_remaining_unique["cluster"] = kmeans.fit_predict(X)
                            representative_indices = df_remaining_unique.groupby("cluster").head(1).index
                            df_clustered = df_remaining.loc[representative_indices].copy()
                        else:
                            df_clustered = df_remaining_unique.copy()
    
                        df_final = pd.concat([df_first_part, df_clustered], ignore_index=True)
                        st.session_state["uploaded_excel"] = df_final.copy()
                        st.session_state["excel_loaded"] = True
                        st.session_state["excel_error"] = None
                        st.rerun()
    
                    except Exception as e:
                        st.session_state["excel_error"] = f"❌ Fehler beim Clustern: {e}"

    if st.session_state.get("excel_error"):
        st.error(st.session_state["excel_error"])

    st.stop()

# === Daten laden & vorbereiten ===
if "uploaded_excel" not in st.session_state:
    st.warning("⚠️ Please upload and (optionally) cluster an Excel file first.")
    st.stop()

vertex_df = st.session_state["uploaded_excel"]

# Präfixe automatisch bestimmen
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
technologies, tech_data, maa_cols = extract_technologies(vertex_df)
tech_time_map = extract_time_series_map(vertex_df,MAA_PREFIX)
additional_cols = get_additional_columns(vertex_df)

# === Sidebar: Einstellungen ===
st.sidebar.markdown("## ⚙️ Settings")


# === Sidebar: Strukturierte Einstellungen ===

with st.sidebar.expander("⚙️ General Settings", expanded=True):
    
    total_vertices_available = len(tech_data)
    st.radio(
        "Select layout mode",
        ["Two-column layout", "Full-width layout"],
        index=0,
        key="layout_mode"
    )
    if st.session_state.layout_mode == "Two-column layout":
        st.slider(
            "Column ratio (Right vs Left)",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.01,
            key="column_ratio"
        )
    st.number_input(
        "Max vertices to display in plots",
        min_value=1,
        max_value=total_vertices_available,
        value=min(5, total_vertices_available),
        step=1,
        key="max_plot_vertices"
    )

with st.sidebar.expander("📊 Plot Options"):
    st.number_input(
        "Number of columns in plot layout",
        min_value=1,
        max_value=10,
        step=1,
        key="n_cols_plots"
    )
    st.checkbox("Show convex combinations in all plots", value=True, key="show_convex")
    st.checkbox("Show original flexibility ranges (red shaded)", value=False, key="show_original_ranges")

with st.sidebar.expander("🌈 Density Plots"):
    st.checkbox("Enable density plots", key="show_density")

with st.sidebar.expander("📌 Additional Metrics"):
    st.radio(
        "Plot type for additional metrics",
        ["Violinplot", "Streudiagramm"],
        index=0,
        key="plot_type_selector"
    )


# === Different Tabs ===
tab1, tab2,tab3 = st.tabs(["📊 Decision Tool", "⚙️ Explanation","📥 Download Results"])

# === Auswahl & Filter-UI ===
with tab1: 
    if st.session_state["layout_mode"] == "Two-column layout":
        ratio = st.session_state.get("column_ratio")
        col1, spacer, col2 = st.columns([0.5+ratio, 0.1, 1.5-ratio])
        with col1:
            st.markdown("### 🧐 Select and Filter Technologies")
            col_select, col_reset = st.columns([4, 1])
            with col_select:
                selected_techs_raw = st.multiselect("Select variables to be constrained", technologies)
                ordered_techs = selected_techs_raw.copy()
        
            with col_reset:
                st.markdown("""
                    <style>
                        div[data-testid="stButton"] > button {
                            padding: 0.3rem 0.6rem;
                            font-size: 0.75rem;
                        }
                    </style>
                """, unsafe_allow_html=True)
                if st.button("🔄 Reset"):
                    for key in list(st.session_state.keys()):
                        if key.startswith("slider_"):
                            del st.session_state[key]
                    st.rerun()
        
            filtered_data = pd.DataFrame()
            convex_data = pd.DataFrame()
        
            filtered_cols = [MAA_PREFIX + tech for tech in ordered_techs]
            selected_data = tech_data[filtered_cols] if ordered_techs else pd.DataFrame(index=tech_data.index)
            current_indices = selected_data.index if ordered_techs else tech_data.index
        
            # === Slider-Filter anwenden ===
            for i, tech in enumerate(ordered_techs):
                col = MAA_PREFIX + tech
                key = f"slider_{tech}"
        
                partial_indices = selected_data.index
                for j in range(i):
                    prev_col = MAA_PREFIX + ordered_techs[j]
                    prev_range = st.session_state.get(f"slider_{ordered_techs[j]}", (selected_data[prev_col].min(), selected_data[prev_col].max()))
                    partial_indices = partial_indices[
                        (selected_data.loc[partial_indices, prev_col] >= prev_range[0]) &
                        (selected_data.loc[partial_indices, prev_col] <= prev_range[1])
                    ]
        
                valid_values = selected_data.loc[partial_indices, col].dropna()
                overall_min = selected_data[col].min()
                overall_max = selected_data[col].max()
        
                missing_slider = any(
                    f"slider_{ordered_techs[j]}" not in st.session_state
                    or st.session_state[f"slider_{ordered_techs[j]}"] is None
                    for j in range(i)
                )
                if missing_slider:
                    st.info(f"➡️ Please configure previous sliders to activate **{tech}**.")
                    st.slider(f"{tech}", float(overall_min), float(overall_max),
                              (float(overall_min), float(overall_max)), key=key, disabled=True)
                    continue
        
                if valid_values.empty:
                    st.warning(f"⚠️ No valid vertices remaining for {tech}.")
                    st.slider(f"{tech}", float(overall_min), float(overall_max),
                              (float(overall_min), float(overall_max)), key=key, disabled=True)
                    continue
        
                min_val = valid_values.min()
                max_val = valid_values.max()
                if min_val == max_val:
                    st.info(f"**{tech}**: No decision flexibility (constant value: {min_val:.2f})")
                    st.session_state[key] = (min_val, max_val)
                    current_indices = current_indices[
                        (selected_data.loc[current_indices, col] >= min_val) &
                        (selected_data.loc[current_indices, col] <= max_val)
                    ]
                    continue
        
                default_val = (float(min_val), float(max_val))
                value = st.session_state.get(key, default_val)
                value = tuple(map(float, value))
        
                slider_value = st.slider(
                    f"{tech}",
                    float(overall_min),
                    float(overall_max),
                    value=value,
                    step=0.01
                )
                clipped_range = (max(min_val, slider_value[0]), min(max_val, slider_value[1]))
        
                if slider_value != clipped_range:
                    st.warning(f"⚠️ Selection for {tech} exceeds valid range ({min_val:.1f}–{max_val:.1f}). Resetting.")
                    if key in st.session_state:
                        del st.session_state[key]
                    st.rerun()
        
                st.session_state[key] = clipped_range
                current_indices = current_indices[
                    (selected_data.loc[current_indices, col] >= clipped_range[0]) &
                    (selected_data.loc[current_indices, col] <= clipped_range[1])
                ]
        
            filtered_data = selected_data.loc[current_indices] if ordered_techs else pd.DataFrame(index=current_indices)
            # === Konvexe Kombinationen ===
            with st.sidebar.expander("➕ Convex Combination Settings"):
                st.number_input(
                    "Total number of convex combinations",
                    min_value=10,
                    max_value=10000,
                    value=100,
                    step=10,
                    key="n_samples"
                )
            
                max_vertices = len(current_indices) if not current_indices.empty else 0
                st.number_input(
                    "Vertices used per combination",
                    min_value=2,
                    max_value=max_vertices if max_vertices > 1 else 2,
                    value=max_vertices if max_vertices > 1 else 2,
                    step=1,
                    key="n_vertices_convex"
                )
            
                st.selectbox(
                    "Dirichlet α (weight spread)",
                    [0.01, 0.1, 1.0],
                    index=[0.01, 0.1, 1.0].index(st.session_state.get('alpha_value', 0.1)),
                    key="alpha_value"
                )
            
                st.number_input(
                    "Combinations per batch",
                    min_value=1,
                    max_value=st.session_state["n_samples"],
                    value=min(10, st.session_state["n_samples"]),
                    step=1,
                    key="n_batch_size"
                )
            
                col_gen, col_reset = st.columns(2)
                with col_gen:
                    generate_convex=st.button("Generate", key="generate_convex_button_sidebar")
                with col_reset:
                    reset_convex=st.button("Reset", key="reset_convex_button_sidebar")
                
                if reset_convex:
                    st.session_state['convex_combinations'] = pd.DataFrame()
                    st.session_state['convex_additional'] = pd.DataFrame()
            
                if generate_convex and not current_indices.empty:
                    base_data_full = tech_data.loc[current_indices]
                    base_additional_data = vertex_df.loc[current_indices, additional_cols] if additional_cols else pd.DataFrame(index=current_indices)
            
                    # Nur wenn MAA_PREFIX == "VALUE_", auch INSTALLED_CAPACITY einbeziehen
                    include_installed_capacity = MAA_PREFIX == "VALUE_"
                    installed_data = pd.DataFrame(index=current_indices)
            
                    if include_installed_capacity:
                        installed_cols = [col for col in vertex_df.columns if col.startswith(INSTALLED_CAPACITY_PREFIX)]
                        installed_data = vertex_df.loc[current_indices, installed_cols]
                    n_total = st.session_state["n_samples"]
                    batch_size = st.session_state["n_batch_size"]
                    n_vertices = st.session_state["n_vertices_convex"]
                    alpha = st.session_state["alpha_value"]
            
                    all_samples = []
                    all_additional_samples = []
                    all_installed_samples = [] if include_installed_capacity else None
                    n_batches = int(np.ceil(n_total / batch_size))
            
                    for _ in range(n_batches):
                        base_sample = base_data_full.sample(
                            n=n_vertices if len(base_data_full) > n_vertices else len(base_data_full),
                            random_state=np.random.randint(0, 999999)
                        )
            
                        base_additional_sample = base_additional_data.loc[base_sample.index] if not base_additional_data.empty else pd.DataFrame(index=base_sample.index)
            
                        effective_batch_size = min(batch_size, n_total - len(all_samples) * batch_size)
                        weights = np.random.dirichlet([alpha] * len(base_sample), size=effective_batch_size)
            
                        batch_samples = weights @ base_sample.values
                        if include_installed_capacity and not installed_data.empty and MAA_PREFIX == "VALUE_":
                            installed_sample = installed_data.loc[base_sample.index]
                            batch_installed = weights @ installed_sample.values
                            all_installed_samples.append(pd.DataFrame(batch_installed, columns=installed_sample.columns))
                        elif include_installed_capacity and not installed_data.empty:
                            batch_installed = weights @ installed_data.values
                            all_installed_samples.append(pd.DataFrame(batch_installed, columns=installed_data.columns))
                        all_samples.append(pd.DataFrame(batch_samples, columns=base_sample.columns))
            
                        if not base_additional_sample.empty:
                            batch_additional = weights @ base_additional_sample.values
                            all_additional_samples.append(pd.DataFrame(batch_additional, columns=base_additional_sample.columns))
            
                        if sum(len(df) for df in all_samples) >= n_total:
                            break
            
                    convex_df = pd.concat(all_samples, ignore_index=True)
                    st.session_state['convex_combinations'] = pd.concat(
                        [st.session_state['convex_combinations'], convex_df],
                        ignore_index=True
                    )
                    if include_installed_capacity and all_installed_samples:
                        installed_comb_df = pd.concat(all_installed_samples, ignore_index=True)
                        st.session_state['convex_combinations'][installed_data.columns] = installed_comb_df
            
                    if all_additional_samples:
                        additional_comb_df = pd.concat(all_additional_samples, ignore_index=True)
                        st.session_state['convex_additional'] = pd.concat(
                            [st.session_state.get('convex_additional', pd.DataFrame()), additional_comb_df],
                            ignore_index=True
                        )
            
                    n_convex = len(st.session_state['convex_combinations'])
                    st.sidebar.info(f"**Currently {n_convex} convex combination(s)** generated.")
        
            # === Konvexe Kombinationen filtern ===
            filtered_convex_data = apply_tech_filters(
                st.session_state['convex_combinations'],
                st.session_state,
                ordered_techs,
                prefix=MAA_PREFIX
            )
        
            filtered_convex_additional = st.session_state.get('convex_additional', pd.DataFrame())
            if not filtered_convex_additional.empty and not filtered_convex_data.empty:
                filtered_convex_additional = filtered_convex_additional.loc[filtered_convex_data.index]
        with col2:
            # === Matplotlib-Style für Diagramme ===
            mpl.rcParams.update({
                'axes.titlesize': 16,
                'axes.labelsize': 14,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 14,
            })
             # === Layout-Optionen für Diagramme ===
              
        
            # === Plot-Vorbereitung ===
            n_techs = sum(1 for v in tech_time_map.values() if len(v) >= 1)
            n_rows = ceil(n_techs / st.session_state.get("n_cols_plots", 3))
            plot_width_per_col = 6
            plot_height_per_row = 3.5
            if MAA_PREFIX == "VALUE_":
                
                st.markdown("### Operational Variables Over Time")
            
                value_time_map = extract_time_series_map(vertex_df,MAA_PREFIX,mode="operational")
            
                n_techs_value = sum(1 for v in value_time_map.values() if len(v) >= 1)
                n_rows_value = ceil(n_techs_value / st.session_state.get("n_cols_plots", 3))
                fig_width_value = plot_width_per_col * st.session_state.get("n_cols_plots", 3)
                fig_height_value = plot_height_per_row * n_rows_value
            
                fig_value, axes_value = plt.subplots(n_rows_value, st.session_state.get("n_cols_plots", 3), figsize=(fig_width_value, fig_height_value))
                fig_value.patch.set_facecolor('#f4f4f4')
                axes_value = axes_value.flatten() if n_techs_value > 1 else [axes_value]
            
                if len(current_indices) > st.session_state["max_plot_vertices"]:
                    plot_indices_val = np.random.choice(current_indices, size=st.session_state["max_plot_vertices"], replace=False)
                    st.caption(f"⚡️ Hinweis: Zeige zufällig {st.session_state['max_plot_vertices']} von {len(current_indices)} gültigen Vertices.")
                else:
                    plot_indices_val = current_indices
            
                plot_idx_val = 0
                for tech, year_cols in sorted(value_time_map.items()):
                    if len(year_cols) < 1 or not any(col.startswith(MAA_PREFIX + tech) for _, col in year_cols):
                        continue
            
                    years_cols_sorted = sorted(year_cols, key=lambda x: x[0])
                    years = [y for y, _ in years_cols_sorted]
                    cols = [col for _, col in years_cols_sorted if col.startswith(MAA_PREFIX + tech)]
            
                    if not cols:
                        continue
            
                    full_values_matrix = vertex_df.loc[current_indices, cols]
                    values_matrix = vertex_df.loc[plot_indices_val, cols]
            
                    if values_matrix.dropna(how='all').empty:
                        continue
            
                    ax = axes_value[plot_idx_val]
                    ax.set_facecolor('#f0f0f0')
            
                    if len(years) == 1:
                        year = years[0]
                        col = cols[0]
                        y_values = values_matrix[col]
                        ax.scatter([year] * len(y_values), y_values, color=(0.1, 0.4, 0.8, 0.4))
            
                        if st.session_state['show_convex'] and not st.session_state['convex_combinations'].empty:
                            if col in filtered_convex_data.columns:
                                convex_vals = filtered_convex_data[col].dropna()
                                ax.scatter([year] * len(convex_vals), convex_vals, color=(1.0, 0.3, 0.3, 0.4))
            
                        min_val = full_values_matrix[col].min()
                        max_val = full_values_matrix[col].max()
                        ax.fill_between([year - 0.4, year + 0.4], min_val, max_val, color=(0.1, 0.4, 0.8, 0.15))
            
                        if st.session_state['show_original_ranges']:
                            original_vals = vertex_df.loc[tech_data.index, col]
                            orig_min = original_vals.min()
                            orig_max = original_vals.max()
                            ax.fill_between([year - 0.4, year + 0.4], orig_min, orig_max, color=(1.0, 0.0, 0.0, 0.08))
            
                        ax.set_xlim(year - 1, year + 1)
                        ax.set_xticks([year])
                    else:
                        for i in values_matrix.index:
                            values = values_matrix.loc[i].values
                            if len(values) != len(years):
                                st.warning(
                                    f"⚠️ Mismatch for tech: **{tech}**\n"
                                    f"- years: {years}\n"
                                    f"- values: {values}\n"
                                    f"- len(years): {len(years)}, len(values): {len(values)}"
                                )
                            ax.plot(years, values, color=(0.1, 0.4, 0.8, 0.3))
            
                        if st.session_state['show_convex'] and not st.session_state['convex_combinations'].empty:
                            if all(col in filtered_convex_data.columns for col in cols):
                                for idx in range(len(filtered_convex_data)):
                                    values = filtered_convex_data.loc[idx, cols].values
                                    if not np.isnan(values).all():
                                        ax.plot(years, values, color=(1.0, 0.3, 0.3, 0.3))
            
                        min_vals = full_values_matrix.min()
                        max_vals = full_values_matrix.max()
                        ax.fill_between(years, min_vals, max_vals, color=(0.1, 0.4, 0.8, 0.15))
            
                        if st.session_state['show_original_ranges']:
                            original_matrix = vertex_df.loc[tech_data.index, cols]
                            original_min = original_matrix.min()
                            original_max = original_matrix.max()
                            ax.fill_between(years, original_min, original_max, color=(1.0, 0.0, 0.0, 0.08))
            
                        ax.set_xticks(years)
            
                    ax.set_title(tech.replace('_', ' ').title())
                    if plot_idx_val >= (n_rows_value - 1) * st.session_state.get("n_cols_plots", 3):
                        ax.set_xlabel("Year")
                    if plot_idx_val % st.session_state.get("n_cols_plots", 3) == 0:
                        ax.set_ylabel("VALUE_")
                    ax.grid(True, linestyle="--", alpha=0.4)
            
                    plot_idx_val += 1
            
                for i in range(plot_idx_val, len(axes_value)):
                    fig_value.delaxes(axes_value[i])
            
                if plot_idx_val > 0:
                    value_line = mlines.Line2D([], [], color=(0.1, 0.4, 0.8), alpha=0.8, label='Vertex')
            
                    legend_anchor_y = 1.2 - 0.02 * max(st.session_state.get("n_cols_plots", 3) - 2, 0)
                    top_margin = legend_anchor_y - 0.12
            
                    fig_value.legend(
                        [value_line],
                        ['Vertex'],
                        loc='upper center',
                        bbox_to_anchor=(0.5, legend_anchor_y),
                        ncol=1,
                        frameon=True,
                        fancybox=True,
                        fontsize=14
                    )
            
                    fig_value.subplots_adjust(
                        top=top_margin,
                        hspace=0.3,
                        wspace=0.18
                    )
            
                st.pyplot(fig_value)
            st.markdown("### ⏳ Installed Capacities Over Time")
        
            n_techs = sum(1 for v in tech_time_map.values() if len(v) >= 1)
            n_rows = ceil(n_techs / st.session_state.get("n_cols_plots", 3))
            plot_width_per_col = 6
            plot_height_per_row = 3.5
            fig_width = plot_width_per_col * st.session_state.get("n_cols_plots", 3)
            fig_height = plot_height_per_row * n_rows
        
            fig, axes = plt.subplots(n_rows,st.session_state.get("n_cols_plots", 3), figsize=(fig_width, fig_height))
            fig.patch.set_facecolor('#f4f4f4')
            axes = axes.flatten() if n_techs > 1 else [axes]
        
            if len(current_indices) > st.session_state["max_plot_vertices"]:
                plot_indices = np.random.choice(current_indices, size=st.session_state["max_plot_vertices"], replace=False)
                st.caption(f"⚡️ Hinweis: Zeige zufällig {st.session_state['max_plot_vertices']} von {len(current_indices)} gültigen Vertices.")
            else:
                plot_indices = current_indices
        
            plot_idx = 0
            for tech, year_cols in sorted(tech_time_map.items()):
                if len(year_cols) < 1:
                    continue
        
                years_cols_sorted = sorted(year_cols, key=lambda x: x[0])
                years = [y for y, _ in years_cols_sorted]
                cols = [col for _, col in years_cols_sorted]
        
                full_values_matrix = vertex_df.loc[current_indices, cols]
                values_matrix = vertex_df.loc[plot_indices, cols]
        
                if values_matrix.dropna(how='all').empty:
                    continue
        
                ax = axes[plot_idx]
                ax.set_facecolor('#f0f0f0')
        
                if len(years) == 1:
                    year = years[0]
                    col = cols[0]
        
                    # Original Punkte
                    y_values = values_matrix[col]
                    ax.scatter([year] * len(y_values), y_values, color=(0.1, 0.4, 0.8, 0.4))
        
                    # Konvex Punkte
                    if st.session_state['show_convex'] and not st.session_state['convex_combinations'].empty:
                        convex_col = f"{INSTALLED_CAPACITY_PREFIX}{tech}_{year}"
                        if convex_col in filtered_convex_data.columns:
                            convex_vals = filtered_convex_data[convex_col].dropna()
                            ax.scatter([year] * len(convex_vals), convex_vals, color=(1.0, 0.3, 0.3, 0.4))
        
                    # Min/Max-Bereich
                    min_val = full_values_matrix[col].min()
                    max_val = full_values_matrix[col].max()
                    ax.fill_between([year - 0.4, year + 0.4], min_val, max_val, color=(0.1, 0.4, 0.8, 0.15))
        
                    if st.session_state['show_original_ranges']:
                        original_vals = vertex_df.loc[tech_data.index, col]
                        orig_min = original_vals.min()
                        orig_max = original_vals.max()
                        ax.fill_between([year - 0.4, year + 0.4], orig_min, orig_max, color=(1.0, 0.0, 0.0, 0.08))
        
                    ax.set_xlim(year - 1, year + 1)
                    ax.set_xticks([year])
                else:
                    for i in values_matrix.index:
                        values = values_matrix.loc[i].values
                        ax.plot(years, values, color=(0.1, 0.4, 0.8, 0.3))
        
                    if st.session_state['show_convex'] and not st.session_state['convex_combinations'].empty:
                        convex_cols = [f"{INSTALLED_CAPACITY_PREFIX}{tech}_{year}" for year in years]
                        if all(col in filtered_convex_data.columns for col in convex_cols):
                            for idx in range(len(filtered_convex_data)):
                                values = filtered_convex_data.loc[idx, convex_cols].values
                                if not np.isnan(values).all():
                                    ax.plot(years, values, color=(1.0, 0.3, 0.3, 0.3))
        
                    min_vals = full_values_matrix.min()
                    max_vals = full_values_matrix.max()
        
                    if st.session_state['show_convex'] and not st.session_state['convex_combinations'].empty:
                        convex_cols = [f"{INSTALLED_CAPACITY_PREFIX}{tech}_{year}" for year in years]
                        if all(col in filtered_convex_data.columns for col in convex_cols):
                            convex_min = filtered_convex_data[convex_cols].min()
                            convex_max = filtered_convex_data[convex_cols].max()
                            min_vals = np.minimum(min_vals, convex_min)
                            max_vals = np.maximum(max_vals, convex_max)
        
                    ax.fill_between(years, min_vals, max_vals, color=(0.1, 0.4, 0.8, 0.15))
        
                    if st.session_state['show_original_ranges']:
                        original_matrix = vertex_df.loc[tech_data.index, cols]
                        original_min = original_matrix.min()
                        original_max = original_matrix.max()
                        ax.fill_between(years, original_min, original_max, color=(1.0, 0.0, 0.0, 0.08))
        
                    ax.set_xticks(years)
        
                ax.set_title(tech.replace('_', ' ').title())
                if plot_idx >= (n_rows - 1) * st.session_state.get("n_cols_plots", 3):
                    ax.set_xlabel("Year")
                if plot_idx % st.session_state.get("n_cols_plots", 3) == 0:
                    ax.set_ylabel("Installed Capacity")
                ax.grid(True, linestyle="--", alpha=0.4)
        
                if plot_idx == 0:
                    handles_labels = ax.get_legend_handles_labels()
        
                plot_idx += 1
        
            for i in range(plot_idx, len(axes)):
                fig.delaxes(axes[i])
        
            if 'handles_labels' in locals():
                handles, labels = handles_labels
                vertex_line = mlines.Line2D([], [], color=(0.1, 0.4, 0.8), alpha=0.8, label='Vertex')
                all_handles = [vertex_line] + handles
                all_labels = ['Vertex'] + labels
        
                legend_anchor_y = 1.2 - 0.02 * max(st.session_state.get("n_cols_plots", 3) - 2, 0)
                top_margin = legend_anchor_y - 0.06
        
                fig.legend(
                    all_handles,
                    all_labels,
                    loc='upper center',
                    bbox_to_anchor=(0.5, legend_anchor_y),
                    ncol=len(all_labels),
                    frameon=True,
                    fancybox=True,
                    fontsize=14
                )
        
                fig.subplots_adjust(
                    top=top_margin,
                    hspace=0.3,
                    wspace=0.18
                )
        
            st.pyplot(fig)
            # === Dichteplots: Kernel Density Estimation über Zeitverläufe ===
            
            
            if st.session_state.get("show_density"):
                st.divider()
                num_interpolated_points = 3
                grid_density = 50
                color_levels = 10
                max_vertices_for_density = 250
        
                techs_with_time_data = [tech for tech in tech_time_map if len(tech_time_map[tech]) > 1]
                n_techs = len(techs_with_time_data)
                n_cols = 2
                n_rows = ceil(n_techs / n_cols)
        
                fig_dichte, axs = plt.subplots(n_rows, n_cols, figsize=(13, 4 * n_rows))
                axs = axs.flatten()
                fig_dichte.patch.set_facecolor('#f4f4f4')
        
                df_base = vertex_df.loc[current_indices]
                if len(df_base) > max_vertices_for_density:
                    df_base = df_base.sample(n=max_vertices_for_density, random_state=42)
        
                for i, tech in enumerate(techs_with_time_data):
                    year_cols = sorted(tech_time_map[tech], key=lambda x: x[0])
                    years = [y for y, _ in year_cols]
                    cols = [col for _, col in year_cols]
        
                    if not all(c in df_base.columns for c in cols):
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
                            x_interp = np.linspace(x_vals[j], x_vals[j + 1], num_interpolated_points)
                            y_interp = np.linspace(y_vals[j], y_vals[j + 1], num_interpolated_points)
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
        
                    Z_masked = np.where((Y >= -0.5) & (Y <= df.max().max() + 0.5), Z, np.nan)
        
                    contour = axs[i].contourf(X, Y, Z_masked, levels=color_levels, cmap="inferno", extend="both")
                    axs[i].set_facecolor("white")
                    axs[i].set_title(tech.replace("_", " ").title())
                    axs[i].set_xlabel("Year")
                    axs[i].set_ylabel("Installed Capacity")
                    axs[i].set_ylim(-1, df.max().max() * 1.05)
        
                    min_vals = df.min()
                    max_vals = df.max()
                    axs[i].fill_between(years, max_vals + 0.5, max_vals.max() + 0.5, facecolor='#f4f4f4', alpha=1)
                    axs[i].fill_between(years, min_vals - 0.5, -0.5, facecolor='#f4f4f4', alpha=1)
        
                    cbar = plt.colorbar(contour, ax=axs[i], label="Density")
                    cbar.set_ticks(np.linspace(np.nanmin(Z_masked), np.nanmax(Z_masked), 4))
                    cbar.set_ticklabels([f"{val:.3f}" for val in np.linspace(np.nanmin(Z_masked), np.nanmax(Z_masked), 4)])
        
                for j in range(i + 1, len(axs)):
                    fig_dichte.delaxes(axs[j])
        
                fig_dichte.subplots_adjust(
                    top=0.95,
                    bottom=0.07,
                    hspace=0.44,
                    wspace=0.3
                )
                st.pyplot(fig_dichte)
        st.session_state["stored_figures"] = [("Operational_Variables", fig_value), ("Installed_Capacities", fig)]
    else:
        
        if "show_tech_info" not in st.session_state:
            st.session_state["show_tech_info"] = False
    
        col_title, col_icon = st.columns([6, 2.5])
        with col_title:
            st.markdown("### 🧐 Select and Filter Technologies")
        with col_icon:
            st.markdown("<div style='margin-top: -1.0rem;'>", unsafe_allow_html=True)
            if st.button("❓", key="show_info_button", help="Show/hide explanation"):
                st.session_state["show_tech_info"] = not st.session_state["show_tech_info"]
            st.markdown("</div>", unsafe_allow_html=True)
    
        if st.session_state["show_tech_info"]:
            st.markdown(
                """
                ### 🧠 Technology Decision Tool – Overview & Usage
    
                This tool supports the exploration and evaluation of **technology transition pathways**. It enables interactive filtering, visualization, and generation of new combinations. The main features include:
    
                ---
    
                #### 🔹 1. Selection & Filtering
                - Choose any number of technologies from the dataset.
                - For each selected technology, a slider will appear to restrict its value range.
                - Filtering is **sequential**: adjust one slider at a time in order. Avoid changing earlier sliders later.
                - Result: only the vertices that meet **all constraints** are used for analysis.
    
                ---
    
                #### 🔹 2. Convex Combinations
                - Generate **new in-between scenarios** based on valid vertices.
                - These are created as convex combinations — weighted averages of selected vertices.
                - Adjustable settings:
                    - Number of total combinations
                    - Number of vertices per combination
                    - Dirichlet alpha (controls weight spread)
                - Installed capacities and additional metrics are also interpolated.
    
                ---
    
                #### 🔹 3. Visualization
                - Time series plots show installed capacities over the years for each technology.
                - Comparison between:
                    - Original valid vertices
                    - Convex combinations (if enabled)
                    - Original min/max value ranges (optional)
                - Additionally: **density plots (KDE)** to reveal typical development patterns.
    
                ---
    
                #### 🔹 4. Additional Metrics
                - Select and visualize additional numeric indicators (e.g. costs, emissions).
                - A scatterplot compares:
                    - Original filtered vertices
                    - Convex combinations (optional)
                    - Global value range (min/max band)
    
                ---
    
                #### 🔹 5. Results Table
                - Full display of remaining valid vertices:
                    - Selected technology values
                    - Installed capacities
                    - Additional metrics
                - Convex combinations are listed separately (if activated).
    
                ---
    
                #### 📌 Notes
                - Filtering is strictly **step-by-step** – apply constraints in order.
                - Empty plots usually indicate over-filtering or missing data.
                - The **maximum number of displayed vertices** can be limited in the sidebar for performance.
    
                ---
                
                """
                ,
                unsafe_allow_html=True
            )
    
        col_select, col_reset = st.columns([4, 1])
        with col_select:
            selected_techs_raw = st.multiselect("Select variables to be constrained", technologies)
            ordered_techs = selected_techs_raw.copy()
    
        with col_reset:
            st.markdown("""
                <style>
                    div[data-testid="stButton"] > button {
                        padding: 0.3rem 0.6rem;
                        font-size: 0.75rem;
                    }
                </style>
            """, unsafe_allow_html=True)
            if st.button("🔄 Reset"):
                for key in list(st.session_state.keys()):
                    if key.startswith("slider_"):
                        del st.session_state[key]
                st.rerun()
    
        filtered_data = pd.DataFrame()
        convex_data = pd.DataFrame()
    
        filtered_cols = [MAA_PREFIX + tech for tech in ordered_techs]
        selected_data = tech_data[filtered_cols] if ordered_techs else pd.DataFrame(index=tech_data.index)
        current_indices = selected_data.index if ordered_techs else tech_data.index
    
        # === Slider-Filter anwenden ===
        for i, tech in enumerate(ordered_techs):
            col = MAA_PREFIX + tech
            key = f"slider_{tech}"
    
            partial_indices = selected_data.index
            for j in range(i):
                prev_col = MAA_PREFIX + ordered_techs[j]
                prev_range = st.session_state.get(f"slider_{ordered_techs[j]}", (selected_data[prev_col].min(), selected_data[prev_col].max()))
                partial_indices = partial_indices[
                    (selected_data.loc[partial_indices, prev_col] >= prev_range[0]) &
                    (selected_data.loc[partial_indices, prev_col] <= prev_range[1])
                ]
    
            valid_values = selected_data.loc[partial_indices, col].dropna()
            overall_min = selected_data[col].min()
            overall_max = selected_data[col].max()
    
            missing_slider = any(
                f"slider_{ordered_techs[j]}" not in st.session_state
                or st.session_state[f"slider_{ordered_techs[j]}"] is None
                for j in range(i)
            )
            if missing_slider:
                st.info(f"➡️ Please configure previous sliders to activate **{tech}**.")
                st.slider(f"{tech}", float(overall_min), float(overall_max),
                          (float(overall_min), float(overall_max)), key=key, disabled=True)
                continue
    
            if valid_values.empty:
                st.warning(f"⚠️ No valid vertices remaining for {tech}.")
                st.slider(f"{tech}", float(overall_min), float(overall_max),
                          (float(overall_min), float(overall_max)), key=key, disabled=True)
                continue
    
            min_val = valid_values.min()
            max_val = valid_values.max()
            if min_val == max_val:
                st.info(f"**{tech}**: No decision flexibility (constant value: {min_val:.2f})")
                st.session_state[key] = (min_val, max_val)
                current_indices = current_indices[
                    (selected_data.loc[current_indices, col] >= min_val) &
                    (selected_data.loc[current_indices, col] <= max_val)
                ]
                continue
    
            default_val = (float(min_val), float(max_val))
            value = st.session_state.get(key, default_val)
            value = tuple(map(float, value))
    
            slider_value = st.slider(
                f"{tech}",
                float(overall_min),
                float(overall_max),
                value=value,
                step=0.01
            )
            clipped_range = (max(min_val, slider_value[0]), min(max_val, slider_value[1]))
    
            if slider_value != clipped_range:
                st.warning(f"⚠️ Selection for {tech} exceeds valid range ({min_val:.1f}–{max_val:.1f}). Resetting.")
                if key in st.session_state:
                    del st.session_state[key]
                st.rerun()
    
            st.session_state[key] = clipped_range
            current_indices = current_indices[
                (selected_data.loc[current_indices, col] >= clipped_range[0]) &
                (selected_data.loc[current_indices, col] <= clipped_range[1])
            ]
    
        filtered_data = selected_data.loc[current_indices] if ordered_techs else pd.DataFrame(index=current_indices)
        # === Konvexe Kombinationen ===
        with st.sidebar.expander("➕ Convex Combination Settings"):
            st.number_input(
                "Total number of convex combinations",
                min_value=10,
                max_value=10000,
                value=100,
                step=10,
                key="n_samples"
            )
        
            max_vertices = len(current_indices) if not current_indices.empty else 0
            st.number_input(
                "Vertices used per combination",
                min_value=2,
                max_value=max_vertices if max_vertices > 1 else 2,
                value=max_vertices if max_vertices > 1 else 2,
                step=1,
                key="n_vertices_convex"
            )
        
            st.selectbox(
                "Dirichlet α (weight spread)",
                [0.01, 0.1, 1.0],
                index=[0.01, 0.1, 1.0].index(st.session_state.get('alpha_value', 0.1)),
                key="alpha_value"
            )
        
            st.number_input(
                "Combinations per batch",
                min_value=1,
                max_value=st.session_state["n_samples"],
                value=min(10, st.session_state["n_samples"]),
                step=1,
                key="n_batch_size"
            )
        
            col_gen, col_reset = st.columns(2)
            with col_gen:
                generate_convex=st.button("Generate", key="generate_convex_button_sidebar")
            with col_reset:
                reset_convex=st.button("Reset", key="reset_convex_button_sidebar")
            
            if reset_convex:
                st.session_state['convex_combinations'] = pd.DataFrame()
                st.session_state['convex_additional'] = pd.DataFrame()
        
            if generate_convex and not current_indices.empty:
                base_data_full = tech_data.loc[current_indices]
                base_additional_data = vertex_df.loc[current_indices, additional_cols] if additional_cols else pd.DataFrame(index=current_indices)
        
                # Nur wenn MAA_PREFIX == "VALUE_", auch INSTALLED_CAPACITY einbeziehen
                include_installed_capacity = MAA_PREFIX == "VALUE_"
                installed_data = pd.DataFrame(index=current_indices)
        
                if include_installed_capacity:
                    installed_cols = [col for col in vertex_df.columns if col.startswith(INSTALLED_CAPACITY_PREFIX)]
                    installed_data = vertex_df.loc[current_indices, installed_cols]
                n_total = st.session_state["n_samples"]
                batch_size = st.session_state["n_batch_size"]
                n_vertices = st.session_state["n_vertices_convex"]
                alpha = st.session_state["alpha_value"]
        
                all_samples = []
                all_additional_samples = []
                all_installed_samples = [] if include_installed_capacity else None
                n_batches = int(np.ceil(n_total / batch_size))
        
                for _ in range(n_batches):
                    base_sample = base_data_full.sample(
                        n=n_vertices if len(base_data_full) > n_vertices else len(base_data_full),
                        random_state=np.random.randint(0, 999999)
                    )
        
                    base_additional_sample = base_additional_data.loc[base_sample.index] if not base_additional_data.empty else pd.DataFrame(index=base_sample.index)
        
                    effective_batch_size = min(batch_size, n_total - len(all_samples) * batch_size)
                    weights = np.random.dirichlet([alpha] * len(base_sample), size=effective_batch_size)
        
                    batch_samples = weights @ base_sample.values
                    if include_installed_capacity and not installed_data.empty and MAA_PREFIX == "VALUE_":
                        installed_sample = installed_data.loc[base_sample.index]
                        batch_installed = weights @ installed_sample.values
                        all_installed_samples.append(pd.DataFrame(batch_installed, columns=installed_sample.columns))
                    elif include_installed_capacity and not installed_data.empty:
                        batch_installed = weights @ installed_data.values
                        all_installed_samples.append(pd.DataFrame(batch_installed, columns=installed_data.columns))
                    all_samples.append(pd.DataFrame(batch_samples, columns=base_sample.columns))
        
                    if not base_additional_sample.empty:
                        batch_additional = weights @ base_additional_sample.values
                        all_additional_samples.append(pd.DataFrame(batch_additional, columns=base_additional_sample.columns))
        
                    if sum(len(df) for df in all_samples) >= n_total:
                        break
        
                convex_df = pd.concat(all_samples, ignore_index=True)
                st.session_state['convex_combinations'] = pd.concat(
                    [st.session_state['convex_combinations'], convex_df],
                    ignore_index=True
                )
                if include_installed_capacity and all_installed_samples:
                    installed_comb_df = pd.concat(all_installed_samples, ignore_index=True)
                    st.session_state['convex_combinations'][installed_data.columns] = installed_comb_df
        
                if all_additional_samples:
                    additional_comb_df = pd.concat(all_additional_samples, ignore_index=True)
                    st.session_state['convex_additional'] = pd.concat(
                        [st.session_state.get('convex_additional', pd.DataFrame()), additional_comb_df],
                        ignore_index=True
                    )
        
                n_convex = len(st.session_state['convex_combinations'])
                st.sidebar.info(f"**Currently {n_convex} convex combination(s)** generated.")
    
        # === Konvexe Kombinationen filtern ===
        filtered_convex_data = apply_tech_filters(
            st.session_state['convex_combinations'],
            st.session_state,
            ordered_techs,
            prefix=MAA_PREFIX
        )
    
        filtered_convex_additional = st.session_state.get('convex_additional', pd.DataFrame())
        if not filtered_convex_additional.empty and not filtered_convex_data.empty:
            filtered_convex_additional = filtered_convex_additional.loc[filtered_convex_data.index]
        st.divider()
        # === Matplotlib-Style für Diagramme ===
        mpl.rcParams.update({
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 14,
        })
         # === Layout-Optionen für Diagramme ===
          
    
        # === Plot-Vorbereitung ===
        if "n_cols_plots" not in st.session_state:
            st.session_state["n_cols_plots"] = 3
        n_techs = sum(1 for v in tech_time_map.values() if len(v) >= 1)
        n_rows = ceil(n_techs / st.session_state.get("n_cols_plots", 3))
        plot_width_per_col = 6
        plot_height_per_row = 3.5
        if MAA_PREFIX == "VALUE_":
            
            st.markdown("### Operational Variables Over Time")
        
            value_time_map = extract_time_series_map(vertex_df,MAA_PREFIX,mode="operational")
        
            n_techs_value = sum(1 for v in value_time_map.values() if len(v) >= 1)
            n_rows_value = ceil(n_techs_value / st.session_state.get("n_cols_plots", 3))
            fig_width_value = plot_width_per_col * st.session_state.get("n_cols_plots", 3)
            fig_height_value = plot_height_per_row * n_rows_value
        
            fig_value, axes_value = plt.subplots(n_rows_value, st.session_state.get("n_cols_plots", 3), figsize=(fig_width_value, fig_height_value))
            fig_value.patch.set_facecolor('#f4f4f4')
            axes_value = axes_value.flatten() if n_techs_value > 1 else [axes_value]
        
            if len(current_indices) > st.session_state["max_plot_vertices"]:
                plot_indices_val = np.random.choice(current_indices, size=st.session_state["max_plot_vertices"], replace=False)
                st.caption(f"⚡️ Hinweis: Zeige zufällig {st.session_state['max_plot_vertices']} von {len(current_indices)} gültigen Vertices.")
            else:
                plot_indices_val = current_indices
        
            plot_idx_val = 0
            for tech, year_cols in sorted(value_time_map.items()):
                if len(year_cols) < 1 or not any(col.startswith(MAA_PREFIX + tech) for _, col in year_cols):
                    continue
        
                years_cols_sorted = sorted(year_cols, key=lambda x: x[0])
                years = [y for y, _ in years_cols_sorted]
                cols = [col for _, col in years_cols_sorted if col.startswith(MAA_PREFIX + tech)]
        
                if not cols:
                    continue
        
                full_values_matrix = vertex_df.loc[current_indices, cols]
                values_matrix = vertex_df.loc[plot_indices_val, cols]
        
                if values_matrix.dropna(how='all').empty:
                    continue
        
                ax = axes_value[plot_idx_val]
                ax.set_facecolor('#f0f0f0')
        
                if len(years) == 1:
                    year = years[0]
                    col = cols[0]
                    y_values = values_matrix[col]
                    ax.scatter([year] * len(y_values), y_values, color=(0.1, 0.4, 0.8, 0.4))
        
                    if st.session_state['show_convex'] and not st.session_state['convex_combinations'].empty:
                        if col in filtered_convex_data.columns:
                            convex_vals = filtered_convex_data[col].dropna()
                            ax.scatter([year] * len(convex_vals), convex_vals, color=(1.0, 0.3, 0.3, 0.4))
        
                    min_val = full_values_matrix[col].min()
                    max_val = full_values_matrix[col].max()
                    ax.fill_between([year - 0.4, year + 0.4], min_val, max_val, color=(0.1, 0.4, 0.8, 0.15))
        
                    if st.session_state['show_original_ranges']:
                        original_vals = vertex_df.loc[tech_data.index, col]
                        orig_min = original_vals.min()
                        orig_max = original_vals.max()
                        ax.fill_between([year - 0.4, year + 0.4], orig_min, orig_max, color=(1.0, 0.0, 0.0, 0.08))
        
                    ax.set_xlim(year - 1, year + 1)
                    ax.set_xticks([year])
                else:
                    for i in values_matrix.index:
                        values = values_matrix.loc[i].values
                        if len(values) != len(years):
                            st.warning(
                                f"⚠️ Mismatch for tech: **{tech}**\n"
                                f"- years: {years}\n"
                                f"- values: {values}\n"
                                f"- len(years): {len(years)}, len(values): {len(values)}"
                            )
                        ax.plot(years, values, color=(0.1, 0.4, 0.8, 0.3))
        
                    if st.session_state['show_convex'] and not st.session_state['convex_combinations'].empty:
                        if all(col in filtered_convex_data.columns for col in cols):
                            for idx in range(len(filtered_convex_data)):
                                values = filtered_convex_data.loc[idx, cols].values
                                if not np.isnan(values).all():
                                    ax.plot(years, values, color=(1.0, 0.3, 0.3, 0.3))
        
                    min_vals = full_values_matrix.min()
                    max_vals = full_values_matrix.max()
                    ax.fill_between(years, min_vals, max_vals, color=(0.1, 0.4, 0.8, 0.15))
        
                    if st.session_state['show_original_ranges']:
                        original_matrix = vertex_df.loc[tech_data.index, cols]
                        original_min = original_matrix.min()
                        original_max = original_matrix.max()
                        ax.fill_between(years, original_min, original_max, color=(1.0, 0.0, 0.0, 0.08))
        
                    ax.set_xticks(years)
        
                ax.set_title(tech.replace('_', ' ').title())
                if plot_idx_val >= (n_rows_value - 1) * st.session_state.get("n_cols_plots", 3):
                    ax.set_xlabel("Year")
                if plot_idx_val % st.session_state.get("n_cols_plots", 3) == 0:
                    ax.set_ylabel("VALUE_")
                ax.grid(True, linestyle="--", alpha=0.4)
        
                plot_idx_val += 1
        
            for i in range(plot_idx_val, len(axes_value)):
                fig_value.delaxes(axes_value[i])
        
            if plot_idx_val > 0:
                value_line = mlines.Line2D([], [], color=(0.1, 0.4, 0.8), alpha=0.8, label='Vertex')
        
                legend_anchor_y = 1.2 - 0.02 * max(st.session_state.get("n_cols_plots", 3) - 2, 0)
                top_margin = legend_anchor_y - 0.12
        
                fig_value.legend(
                    [value_line],
                    ['Vertex'],
                    loc='upper center',
                    bbox_to_anchor=(0.5, legend_anchor_y),
                    ncol=1,
                    frameon=True,
                    fancybox=True,
                    fontsize=14
                )
        
                fig_value.subplots_adjust(
                    top=top_margin,
                    hspace=0.3,
                    wspace=0.18
                )
        
            st.pyplot(fig_value)
        st.markdown("### ⏳ Installed Capacities Over Time")
    
        n_techs = sum(1 for v in tech_time_map.values() if len(v) >= 1)
        n_rows = ceil(n_techs / st.session_state.get("n_cols_plots", 3))
        plot_width_per_col = 6
        plot_height_per_row = 3.5
        fig_width = plot_width_per_col * st.session_state.get("n_cols_plots", 3)
        fig_height = plot_height_per_row * n_rows
    
        fig, axes = plt.subplots(n_rows,st.session_state.get("n_cols_plots", 3), figsize=(fig_width, fig_height))
        fig.patch.set_facecolor('#f4f4f4')
        axes = axes.flatten() if n_techs > 1 else [axes]
    
        if len(current_indices) > st.session_state["max_plot_vertices"]:
            plot_indices = np.random.choice(current_indices, size=st.session_state["max_plot_vertices"], replace=False)
            st.caption(f"⚡️ Hinweis: Zeige zufällig {st.session_state['max_plot_vertices']} von {len(current_indices)} gültigen Vertices.")
        else:
            plot_indices = current_indices
    
        plot_idx = 0
        for tech, year_cols in sorted(tech_time_map.items()):
            if len(year_cols) < 1:
                continue
    
            years_cols_sorted = sorted(year_cols, key=lambda x: x[0])
            years = [y for y, _ in years_cols_sorted]
            cols = [col for _, col in years_cols_sorted]
    
            full_values_matrix = vertex_df.loc[current_indices, cols]
            values_matrix = vertex_df.loc[plot_indices, cols]
    
            if values_matrix.dropna(how='all').empty:
                continue
    
            ax = axes[plot_idx]
            ax.set_facecolor('#f0f0f0')
    
            if len(years) == 1:
                year = years[0]
                col = cols[0]
    
                # Original Punkte
                y_values = values_matrix[col]
                ax.scatter([year] * len(y_values), y_values, color=(0.1, 0.4, 0.8, 0.4))
    
                # Konvex Punkte
                if st.session_state['show_convex'] and not st.session_state['convex_combinations'].empty:
                    convex_col = f"{INSTALLED_CAPACITY_PREFIX}{tech}_{year}"
                    if convex_col in filtered_convex_data.columns:
                        convex_vals = filtered_convex_data[convex_col].dropna()
                        ax.scatter([year] * len(convex_vals), convex_vals, color=(1.0, 0.3, 0.3, 0.4))
    
                # Min/Max-Bereich
                min_val = full_values_matrix[col].min()
                max_val = full_values_matrix[col].max()
                ax.fill_between([year - 0.4, year + 0.4], min_val, max_val, color=(0.1, 0.4, 0.8, 0.15))
    
                if st.session_state['show_original_ranges']:
                    original_vals = vertex_df.loc[tech_data.index, col]
                    orig_min = original_vals.min()
                    orig_max = original_vals.max()
                    ax.fill_between([year - 0.4, year + 0.4], orig_min, orig_max, color=(1.0, 0.0, 0.0, 0.08))
    
                ax.set_xlim(year - 1, year + 1)
                ax.set_xticks([year])
            else:
                for i in values_matrix.index:
                    values = values_matrix.loc[i].values
                    ax.plot(years, values, color=(0.1, 0.4, 0.8, 0.3))
    
                if st.session_state['show_convex'] and not st.session_state['convex_combinations'].empty:
                    convex_cols = [f"{INSTALLED_CAPACITY_PREFIX}{tech}_{year}" for year in years]
                    if all(col in filtered_convex_data.columns for col in convex_cols):
                        for idx in range(len(filtered_convex_data)):
                            values = filtered_convex_data.loc[idx, convex_cols].values
                            if not np.isnan(values).all():
                                ax.plot(years, values, color=(1.0, 0.3, 0.3, 0.3))
    
                min_vals = full_values_matrix.min()
                max_vals = full_values_matrix.max()
    
                if st.session_state['show_convex'] and not st.session_state['convex_combinations'].empty:
                    convex_cols = [f"{INSTALLED_CAPACITY_PREFIX}{tech}_{year}" for year in years]
                    if all(col in filtered_convex_data.columns for col in convex_cols):
                        convex_min = filtered_convex_data[convex_cols].min()
                        convex_max = filtered_convex_data[convex_cols].max()
                        min_vals = np.minimum(min_vals, convex_min)
                        max_vals = np.maximum(max_vals, convex_max)
    
                ax.fill_between(years, min_vals, max_vals, color=(0.1, 0.4, 0.8, 0.15))
    
                if st.session_state['show_original_ranges']:
                    original_matrix = vertex_df.loc[tech_data.index, cols]
                    original_min = original_matrix.min()
                    original_max = original_matrix.max()
                    ax.fill_between(years, original_min, original_max, color=(1.0, 0.0, 0.0, 0.08))
    
                ax.set_xticks(years)
    
            ax.set_title(tech.replace('_', ' ').title())
            if plot_idx >= (n_rows - 1) * st.session_state.get("n_cols_plots", 3):
                ax.set_xlabel("Year")
            if plot_idx % st.session_state.get("n_cols_plots", 3) == 0:
                ax.set_ylabel("Installed Capacity")
            ax.grid(True, linestyle="--", alpha=0.4)
    
            if plot_idx == 0:
                handles_labels = ax.get_legend_handles_labels()
    
            plot_idx += 1
    
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])
    
        if 'handles_labels' in locals():
            handles, labels = handles_labels
            vertex_line = mlines.Line2D([], [], color=(0.1, 0.4, 0.8), alpha=0.8, label='Vertex')
            all_handles = [vertex_line] + handles
            all_labels = ['Vertex'] + labels
    
            legend_anchor_y = 1.2 - 0.02 * max(st.session_state.get("n_cols_plots", 3) - 2, 0)
            top_margin = legend_anchor_y - 0.06
    
            fig.legend(
                all_handles,
                all_labels,
                loc='upper center',
                bbox_to_anchor=(0.5, legend_anchor_y),
                ncol=len(all_labels),
                frameon=True,
                fancybox=True,
                fontsize=14
            )
    
            fig.subplots_adjust(
                top=top_margin,
                hspace=0.3,
                wspace=0.18
            )
    
        st.pyplot(fig)
        # === Dichteplots: Kernel Density Estimation über Zeitverläufe ===
        
        
        if st.session_state.get("show_density"):
            st.divider()
            num_interpolated_points = 3
            grid_density = 50
            color_levels = 10
            max_vertices_for_density = 250
    
            techs_with_time_data = [tech for tech in tech_time_map if len(tech_time_map[tech]) > 1]
            n_techs = len(techs_with_time_data)
            n_cols = 2
            n_rows = ceil(n_techs / n_cols)
    
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(13, 4 * n_rows))
            axs = axs.flatten()
            fig.patch.set_facecolor('#f4f4f4')
    
            df_base = vertex_df.loc[current_indices]
            if len(df_base) > max_vertices_for_density:
                df_base = df_base.sample(n=max_vertices_for_density, random_state=42)
    
            for i, tech in enumerate(techs_with_time_data):
                year_cols = sorted(tech_time_map[tech], key=lambda x: x[0])
                years = [y for y, _ in year_cols]
                cols = [col for _, col in year_cols]
    
                if not all(c in df_base.columns for c in cols):
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
                        x_interp = np.linspace(x_vals[j], x_vals[j + 1], num_interpolated_points)
                        y_interp = np.linspace(y_vals[j], y_vals[j + 1], num_interpolated_points)
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
    
                Z_masked = np.where((Y >= -0.5) & (Y <= df.max().max() + 0.5), Z, np.nan)
    
                contour = axs[i].contourf(X, Y, Z_masked, levels=color_levels, cmap="inferno", extend="both")
                axs[i].set_facecolor("white")
                axs[i].set_title(tech.replace("_", " ").title())
                axs[i].set_xlabel("Year")
                axs[i].set_ylabel("Installed Capacity")
                axs[i].set_ylim(-1, df.max().max() * 1.05)
    
                min_vals = df.min()
                max_vals = df.max()
                axs[i].fill_between(years, max_vals + 0.5, max_vals.max() + 0.5, facecolor='#f4f4f4', alpha=1)
                axs[i].fill_between(years, min_vals - 0.5, -0.5, facecolor='#f4f4f4', alpha=1)
    
                cbar = plt.colorbar(contour, ax=axs[i], label="Density")
                cbar.set_ticks(np.linspace(np.nanmin(Z_masked), np.nanmax(Z_masked), 4))
                cbar.set_ticklabels([f"{val:.3f}" for val in np.linspace(np.nanmin(Z_masked), np.nanmax(Z_masked), 4)])
    
            for j in range(i + 1, len(axs)):
                fig.delaxes(axs[j])
    
            fig.subplots_adjust(
                top=0.95,
                bottom=0.07,
                hspace=0.44,
                wspace=0.3
            )
            st.pyplot(fig)
    
    st.divider()
    # === Weitere Metriken anzeigen ===
    # Beispiel-Daten
    st.markdown("### 📈 Additional Metrics")
    
    # Auswahl des Plottyps
    
    
    # Checkbox für Konvexe Kombinationen
    
    if additional_cols:
        selected_metrics = st.multiselect(
            "📌 Select additional metrics to visualize",
            additional_cols,
            default=additional_cols[:5] if len(additional_cols) > 5 else additional_cols
        )
    
        if selected_metrics:
            additional_data = vertex_df.loc[tech_data.index, selected_metrics]
            filtered_additional = additional_data.loc[current_indices]
    
            if st.session_state.get("show_convex") and not filtered_convex_additional.empty:
                filtered_combined = pd.concat([filtered_additional, filtered_convex_additional[selected_metrics]], axis=0)
            else:
                filtered_combined = filtered_additional
    
            if st.session_state.get("plot_type_selector") == "Violinplot":
                max_cols = 8
                n_metrics = len(selected_metrics)
                n_cols = min(max_cols, n_metrics)
                n_rows = -(-n_metrics // max_cols)  # Ceiling division
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 5))
                fig.patch.set_facecolor('#f4f4f4')  # Hintergrund der gesamten Figur
                
                if n_metrics == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()
                
                for i, col in enumerate(selected_metrics):
                    ax = axes[i]
                    ax.set_facecolor('#f0f0f0')  # Hintergrund pro Plot-Achse
                
                    values = filtered_combined[col].dropna().values
                    if len(values) == 0:
                        ax.set_visible(False)
                        continue
                
                    vp = ax.violinplot(
                        [values],
                        showmeans=False,
                        showmedians=True,
                        showextrema=True,
                        widths=0.8
                    )
                
                    for pc in vp['bodies']:
                        pc.set_facecolor((0.1, 0.4, 0.8, 0.7))  # Blau, leicht transparent
                        pc.set_edgecolor('black')
                        pc.set_alpha(0.7)
                
                    if 'cmedians' in vp:
                        vp['cmedians'].set_color('black')
                
                    clean_label = (
                        col.replace("installed_capacity_", "")
                           .replace("INSTALLED_CAPACITY_", "")
                           .replace("NEW_CAPACITY_", "")
                    )
                    ax.set_title(clean_label, fontsize=14)
                    ax.set_xticks([])
                    ax.set_ylabel("Metric Value", fontsize=13)
                    ax.tick_params(axis='y', labelsize=12)
                    ax.grid(True, linestyle="--", alpha=0.4)
                
                # Unsichtbare Achsen ausblenden
                for j in range(i + 1, len(axes)):
                    axes[j].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
    
            elif st.session_state.get("plot_type_selector") == "Streudiagramm":
                fig_scatter, ax_scatter = plt.subplots(figsize=(12, 3))
                fig_scatter.patch.set_facecolor('#f4f4f4')
                ax_scatter.set_facecolor('#f0f0f0')
    
                y_max = max([filtered_combined[col].max() for col in selected_metrics])
    
                for i, col in enumerate(selected_metrics):
                    values = filtered_additional[col].dropna().values
                    x_vals = [i] * len(values)
                    ax_scatter.scatter(x_vals, values, alpha=0.7, color="#444444", label="Original" if i == 0 else None)
    
                    if st.session_state.get("show_convex") and not filtered_convex_additional.empty:
                        convex_vals = filtered_convex_additional[col].dropna().values
                        cx_vals = [i] * len(convex_vals)
                        ax_scatter.scatter(cx_vals, convex_vals, alpha=0.5, color="#ff4444", label="Convex" if i == 0 else None)
    
                    global_min = additional_data[col].min()
                    global_max = additional_data[col].max()
    
                    ax_scatter.fill_between(
                        [i - 0.3, i + 0.3],
                        global_min,
                        global_max,
                        color='gray',
                        alpha=0.15
                    )
    
                    ax_scatter.text(
                        i,
                        global_max + (y_max * 0.02),
                        f"{global_min:.1f}–{global_max:.1f}",
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        color='black'
                    )
    
                handles, labels = ax_scatter.get_legend_handles_labels()
                if handles:
                    ax_scatter.legend(handles, labels)
    
                clean_labels = [
                    col.replace("installed_capacity_", "")
                        .replace("INSTALLED_CAPACITY_", "")
                        .replace("NEW_CAPACITY_", "")
                    for col in selected_metrics
                ]
                ax_scatter.set_xticks(range(len(selected_metrics)))
                ax_scatter.set_xticklabels(clean_labels, rotation=45, ha="right")
                ax_scatter.set_ylabel("Metric Value")
                ax_scatter.set_title("Remaining Values for Selected Additional Metrics")
                ax_scatter.set_ylim(0, y_max * 1.1)
                ax_scatter.grid(True, linestyle="--", alpha=0.4)
    
                st.pyplot(fig_scatter)
    
        else:
            st.info("Please select at least one metric to visualize.")
    else:
        st.info("No numeric columns found after the last 'NEW_CAPACITY' column.")
        
        
    # === Daten als Tabelle anzeigen ===
    st.divider()
    st.markdown("### 📄 Show remaining vertices as table")
    
    if filtered_data.empty and (
        st.session_state['convex_combinations'].empty 
        or not st.session_state['show_convex']
    ):
        st.info("No valid vertices available for the current selection.")
    else:
        # === Original-Vertices ===
        st.markdown("#### Original Vertices")
        filtered_full_data = tech_data.loc[current_indices] if not current_indices.empty else pd.DataFrame()
        if filtered_data.empty:
            st.dataframe(filtered_full_data, use_container_width=True)
        else:
            frames_to_concat = [filtered_full_data.reset_index(drop=True)]
    
            # Füge Installed Capacity-Spalten hinzu, falls VALUE_-Modus
            if MAA_PREFIX == "VALUE_":
                installed_cols = [col for col in vertex_df.columns if col.startswith(INSTALLED_CAPACITY_PREFIX)]
                installed_part = vertex_df.loc[filtered_full_data.index, installed_cols]
                frames_to_concat.append(installed_part.reset_index(drop=True))
    
            if additional_cols:
                additional_metrics_part = vertex_df.loc[filtered_full_data.index, additional_cols]
                frames_to_concat.append(additional_metrics_part.reset_index(drop=True))
    
            full_with_all = pd.concat(frames_to_concat, axis=1)
            st.dataframe(full_with_all, use_container_width=True)
    
        # === Konvexe Kombinationen ===
        if (
            st.session_state['show_convex'] 
            and not st.session_state['convex_combinations'].empty
        ):
            st.markdown("---")
            st.markdown("#### Convex Combinations")
            if filtered_convex_data.empty:
                st.dataframe(filtered_convex_data, use_container_width=True)
            else:
                frames_to_concat = [filtered_convex_data.reset_index(drop=True)]
    
                
    
                if additional_cols and not filtered_convex_additional.empty:
                    additional_convex_part = filtered_convex_additional[additional_cols].reset_index(drop=True)
                    frames_to_concat.append(additional_convex_part)
    
                convex_with_all = pd.concat(frames_to_concat, axis=1)
                st.dataframe(convex_with_all, use_container_width=True)

with tab2: 
         """
    ### 🧠 Technology Decision Tool – Overview & Usage

    This tool supports the exploration and evaluation of **technology transition pathways**. It enables interactive filtering, visualization, and generation of new combinations. The main features include:

    ---

    #### 🔹 1. Selection & Filtering
    - Choose any number of technologies from the dataset.
    - For each selected technology, a slider will appear to restrict its value range.
    - Filtering is **sequential**: adjust one slider at a time in order. Avoid changing earlier sliders later.
    - Result: only the vertices that meet **all constraints** are used for analysis.

    ---

    #### 🔹 2. Convex Combinations
    - Generate **new in-between scenarios** based on valid vertices.
    - These are created as convex combinations — weighted averages of selected vertices.
    - Adjustable settings:
        - Number of total combinations
        - Number of vertices per combination
        - Dirichlet alpha (controls weight spread)
    - Installed capacities and additional metrics are also interpolated.

    ---

    #### 🔹 3. Visualization
    - Time series plots show installed capacities over the years for each technology.
    - Comparison between:
        - Original valid vertices
        - Convex combinations (if enabled)
        - Original min/max value ranges (optional)
    - Additionally: **density plots (KDE)** to reveal typical development patterns.

    ---

    #### 🔹 4. Additional Metrics
    - Select and visualize additional numeric indicators (e.g. costs, emissions).
    - A scatterplot compares:
        - Original filtered vertices
        - Convex combinations (optional)
        - Global value range (min/max band)

    ---

    #### 🔹 5. Results Table
    - Full display of remaining valid vertices:
        - Selected technology values
        - Installed capacities
        - Additional metrics
    - Convex combinations are listed separately (if activated).

    ---

    #### 📌 Notes
    - Filtering is strictly **step-by-step** – apply constraints in order.
    - Empty plots usually indicate over-filtering or missing data.
    - The **maximum number of displayed vertices** can be limited in the sidebar for performance.

    ---
    
    """
with tab3:
    
    
    st.header("⬇️ Export Plots as PDF")
    
    if "stored_figures" in st.session_state and st.session_state["stored_figures"]:
        if st.button("📄 Generate ZIP with all Plots"):
            zip_buffer = BytesIO()
            with ZipFile(zip_buffer, "w") as zip_file:
                for name, fig in st.session_state["stored_figures"]:
                    pdf_bytes = BytesIO()
                    fig.savefig(pdf_bytes, format="pdf", bbox_inches="tight")
                    pdf_bytes.seek(0)
                    filename = f"{name.replace(' ', '_')}.pdf"
                    zip_file.writestr(filename, pdf_bytes.read())
            zip_buffer.seek(0)
    
            st.download_button(
                label="⬇️ Download ZIP",
                data=zip_buffer,
                file_name="exported_plots.zip",
                mime="application/zip"
            )
    else:
        st.info("⚠️ No plots available for download yet. Please generate plots in Tab 1.")
