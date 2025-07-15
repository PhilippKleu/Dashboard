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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.graph_objs.layout import XAxis, YAxis
import requests


DEFAULT_FILENAME = "VERTEX_RESULTS.xlsx"
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_FILENAME)

# === Streamlit Style ===
st.set_page_config(page_title="Decision Tool for Near-Optimal Transition Pathways", layout="wide")
# === body, .stApp background color website ===
# === .stButton>button Reset Button ===

st.markdown("""
    <style>
        body, .stApp {
            background-color: #f4f4f4;
            font-family: 'Segoe UI', sans-serif;
        } 
        .stButton>button {
            background-color: #C0C6D2;
            color: black;
            border: none;
            padding: 0.5em 1.2em;
            border-radius: 8px;
            font-weight: 500;
        }
        div[data-testid="stFileUploader"]:hover,
        div[data-testid^="stSelectbox"]:hover,
        div[data-testid^="stMultiSelect"]:hover,
        div[data-testid^="stNumberInput"]:hover,
        div[data-testid^="stRadio"]:hover,
        div[data-testid^="stSlider"]:hover {
            transform: scale(1.01);
            transition: transform 0.2s ease;
        }
        
    </style>
""", unsafe_allow_html=True)
def clean_plot_indices(raw_indices):
    """
    Wandelt eine Liste wie ['np.int64(22)', 'np.int64(738)'] in [22, 738] um.
    Funktioniert auch bei echten ints und np.int64-Objekten.
    """
    cleaned = []
    for idx in raw_indices:
        try:
            if isinstance(idx, int):
                cleaned.append(idx)
            elif isinstance(idx, str):
                match = re.search(r"\d+", idx)
                if match:
                    cleaned.append(int(match.group()))
            else:
                cleaned.append(int(idx))  # z. B. np.int64
        except Exception as e:
            print(f"⚠️ Error converting index {idx}: {e}")
    return cleaned
def select_representative_vertices_by_kmeans(
    df: pd.DataFrame,
    cols: list,
    n_vertices: int,
    index_subset=None,
    random_state: int = 42
) -> list:
    """
    Wählt repräsentative Vertices per k-Means-Clustering basierend auf den angegebenen Spalten.

    Args:
        df (pd.DataFrame): Vollständiges DataFrame mit Vertex-Daten.
        cols (list): Spalten, die für das Clustering verwendet werden (z. B. Zeitreihendaten).
        n_vertices (int): Gewünschte Anzahl an Vertices.
        index_subset (optional): Teilmenge von Indizes zur Auswahl (z. B. current_indices).
        random_state (int): Reproduzierbarkeit des Clusters.

    Returns:
        list: Liste mit ausgewählten Vertex-Indizes.
    """
    if index_subset is not None:
        data = df.loc[index_subset, cols]
    else:
        data = df[cols]

    # Fehlende Werte mit 0 ersetzen (oder ggf. eine andere Strategie verwenden)
    data = data.fillna(0)

    # Begrenze Anzahl Cluster auf max. Anzahl verfügbarer Zeilen
    n_clusters = min(len(data), n_vertices)

    if n_clusters == 0:
        return []

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(data)

    selected_indices = []
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        if not cluster_points.empty:
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_points.values - center, axis=1)
            closest_idx = cluster_points.index[np.argmin(distances)]
            selected_indices.append(closest_idx)

    return selected_indices
    
def select_and_show_vertex_info(plot_indices, current_indices, vertex_df, tech_data, additional_cols):
    vertex_placeholder = "— Please select —"
    options_with_placeholder = [vertex_placeholder] + list(plot_indices)
    selected_label = st.selectbox("Choose a displayed Vertex to highlight.", options=options_with_placeholder)
    selected_vertex = selected_label if selected_label != vertex_placeholder else None

    filtered_additional = vertex_df.loc[current_indices, additional_cols[:5]]
    full_additional = vertex_df.loc[tech_data.index, additional_cols[:5]]

    if selected_vertex is not None and selected_vertex in filtered_additional.index:
        st.markdown("### ℹ️ Zusatzinformationen")
        extra_data = filtered_additional.loc[selected_vertex]
        col1_inner, col2_inner = st.columns(2)
        extra_items = list(extra_data.items())

        for i in range(0, len(extra_items), 2):
            for col, (key, val) in zip([col1_inner, col2_inner], extra_items[i:i+2]):
                val_display = f"{val:.2f}" if pd.notna(val) else "n/a"
                col.markdown(f"**{key}**: {val_display}")

                if pd.notna(val):
                    col_vals = full_additional[key].dropna()
                    if not col_vals.empty:
                        q1 = col_vals.quantile(0.25)
                        q2 = col_vals.quantile(0.5)
                        q3 = col_vals.quantile(0.75)
                        min_val = col_vals.min()
                        max_val = col_vals.max()

                        fig, ax = plt.subplots(figsize=(3.5, 0.3))
                        ax.hlines(0, min_val, max_val, color="lightgray", linewidth=6)
                        for q in [q1, q2, q3]:
                            ax.vlines(q, -0.1, 0.1, color="gray", linewidth=1)
                        ax.plot(val, 0, 'o', color='blue')
                        ax.set_xlim(min_val, max_val)
                        ax.set_yticks([])
                        ax.set_xticks([])
                        for spine in ax.spines.values():
                            spine.set_visible(False)
                        col.pyplot(fig)

    return selected_vertex

def plot_violin_values(
    vertex_df,
    valid_techs_value,
    value_time_map,
    plot_indices_val,
    current_indices,
    filtered_convex_data,
    MAA_PREFIX="MAA_"
):
    n_techs_value = len(valid_techs_value)
    n_cols_val = st.session_state.get("n_cols_plots", 3)
    n_rows_val = ceil(n_techs_value / n_cols_val)

    plot_width_per_col = 6
    plot_height_per_row = 3.5
    fig_width_val = plot_width_per_col * n_cols_val
    fig_height_val = plot_height_per_row * n_rows_val

    fig_val, axes_val = plt.subplots(n_rows_val, n_cols_val, figsize=(fig_width_val, fig_height_val))
    fig_val.patch.set_facecolor('#f4f4f4')
    axes_val = axes_val.flatten() if n_techs_value > 1 else [axes_val]

    plot_idx_val = 0
    for tech in valid_techs_value:
        year_cols = value_time_map[tech]
        years_cols_sorted = sorted(year_cols, key=lambda x: x[0])
        years = [y for y, c in years_cols_sorted if c.startswith(MAA_PREFIX + tech)]
        cols = [c for y, c in years_cols_sorted if c.startswith(MAA_PREFIX + tech)]

        if not cols:
            continue

        values_matrix = vertex_df.loc[plot_indices_val, cols]

        # === Konvexe Kombinationen ===
        if st.session_state.get('show_convex', False) and not st.session_state['convex_combinations'].empty:
            if all(col in filtered_convex_data.columns for col in cols):
                convex_matrix = filtered_convex_data[cols]
                values_matrix = pd.concat([values_matrix, convex_matrix], axis=0)

        if values_matrix.dropna(how='all').empty:
            continue

        ax = axes_val[plot_idx_val]
        ax.set_facecolor('#f0f0f0')

        data = [values_matrix[col].dropna().values for col in cols]

        if all(len(d) > 0 for d in data):
            ax.violinplot(data, positions=years, showmeans=False, showmedians=True, widths=2.0)

            # === Vertex Highlight ===
            selected_vertex = st.session_state.get("selected_vertex")
            if selected_vertex is not None and selected_vertex in vertex_df.index:
                try:
                    highlight_vals = vertex_df.loc[selected_vertex, cols]
                    for y, val in zip(years, highlight_vals):
                        if pd.notnull(val):
                            ax.scatter(y, val, color="black", s=60, zorder=3,
                                       label="Selected Vertex" if plot_idx_val == 0 else None)
                except Exception as e:
                    st.warning(f"⚠️ Error highlighting selected vertex for {tech}: {e}")

        # === Originalbereich ===
        if st.session_state.get('show_original_ranges', False):
            try:
                original_matrix = vertex_df.loc[current_indices, cols]
            except Exception:
                original_matrix = vertex_df[cols]

            original_min = original_matrix.min()
            original_max = original_matrix.max()

            for y, omin, omax in zip(years, original_min, original_max):
                if not np.isnan(omin) and not np.isnan(omax):
                    ax.fill_between([y - 0.4, y + 0.4], omin, omax, color=(1.0, 0.0, 0.0, 0.08))

        ax.set_title(tech.replace('_', ' ').title())
        ax.set_xticks(years)
        ax.set_xticklabels([str(y) for y in years])

        if plot_idx_val >= (n_rows_val - 1) * n_cols_val:
            ax.set_xlabel("Year")
        if plot_idx_val % n_cols_val == 0:
            ax.set_ylabel("VALUE_")

        ax.grid(True, linestyle="--", alpha=0.4)
        plot_idx_val += 1

    # Entferne leere Subplots
    for i in range(plot_idx_val, len(axes_val)):
        if axes_val[i] in fig_val.axes:
            fig_val.delaxes(axes_val[i])

    # === Legende ===
    if plot_idx_val > 0:
        legend_items = []
        convex_line = mlines.Line2D([], [], color=(0.1, 0.4, 0.8), alpha=0.8, label='Values incl. Convex')
        legend_items.append(convex_line)

        if st.session_state.get("selected_vertex") is not None:
            vertex_dot = mlines.Line2D([], [], color="black", marker='o', linestyle='None', markersize=8,
                                       label="Selected Vertex")
            legend_items.append(vertex_dot)

        fig_val.legend(
            legend_items,
            [item.get_label() for item in legend_items],
            loc='upper center',
            bbox_to_anchor=(0.5, 1.2 - 0.02 * max(n_cols_val - 2, 0)),
            ncol=1,
            frameon=True,
            fancybox=True,
            fontsize=14
        )

        fig_val.subplots_adjust(
            top=1.14 - 0.02 * max(n_cols_val - 2, 0),
            hspace=0.3,
            wspace=0.18
        )

    st.pyplot(fig_val)

def plot_operational_variables_over_time(
    vertex_df,
    current_indices,
    plot_indices_val,
    time_column_map,
    selected_vertex,
    n_cols_val=3,
    show_convex=False,
    st_convex=None,
    filtered_convex_data=None,
    show_original_ranges=False,
    max_plot_vertices=20,
    maa_prefix="MAA_",
    apply_prefix=True,
    plot_title="Operational Variables Over Time"
):
    valid_techs_value = sorted([tech for tech, v in time_column_map.items() if len(v) >= 1])
    n_techs_value = len(valid_techs_value)
    n_rows_val = ceil(n_techs_value / n_cols_val)

    fig_width_val = 9 * (1 / n_cols_val) * n_cols_val
    fig_height_val = fig_width_val

    fig_val = make_subplots(
        rows=n_rows_val,
        cols=n_cols_val,
        subplot_titles=[tech.replace("_", " ").title() for tech in valid_techs_value],
        horizontal_spacing=0.08,
        vertical_spacing=0.09
    )

    for idx, tech in enumerate(valid_techs_value):
        year_cols = time_column_map[tech]
        if not year_cols:
            st.write(f"⚠️ No columns found for technology: {tech}")
            continue

        years_cols_sorted = sorted(year_cols, key=lambda x: x[0])
        years = [y for y, c in years_cols_sorted if (not apply_prefix or c.startswith(maa_prefix + tech))]
        cols = [c for y, c in years_cols_sorted if (not apply_prefix or c.startswith(maa_prefix + tech))]
        plot_mode = 'markers' if len(years) == 1 else 'lines'

        if not cols:
            st.write(f"🚫 Skipping technology {tech} – no matching columns found.")
            continue

        row, col = divmod(idx, n_cols_val)
        row += 1
        col += 1

        full_values_matrix = vertex_df.loc[current_indices, cols]
        values_matrix = vertex_df.loc[plot_indices_val, cols]

        if values_matrix.dropna(how='all').empty:
            st.write(f"⚠️ Empty values for {tech}, skipping.")
            continue

        for i in values_matrix.index:
            values = values_matrix.loc[i].values
            if len(values) != len(years):
                st.write(f"⚠️ Row {i}: Mismatch between number of years and values ({len(years)} vs {len(values)})")
                continue

            is_sel = selected_vertex is not None and i == selected_vertex

            hover_text = f"<b>Vertex {i}</b><br>" + "<br>".join(
                [f"{year}: {val:.2f}" if not np.isnan(val) else f"{year}: n/a" for year, val in zip(years, values)]
            )

            fig_val.add_trace(go.Scatter(
                x=years,
                y=values,
                mode=plot_mode,
                line=dict(
                    color="rgba(26, 102, 204, 0.8)" if is_sel else "rgba(26, 102, 204, 0.3)",
                    width=4 if is_sel else 1
                ) if plot_mode == "lines" else None,
                marker=dict(
                    color="rgba(26, 102, 204, 0.8)" if is_sel else "rgba(26, 102, 204, 0.3)",
                    size=10
                ) if plot_mode == "markers" else None,
                text=[hover_text] * len(years),
                hoverinfo='text',
                showlegend=False
            ), row=row, col=col)

        # ✅ Konvexe Kombinationen (korrekter Spaltenabgleich)
        
        if show_convex and not st_convex.empty:
            if len(years) == 1:
                
                convex_col = f"{INSTALLED_CAPACITY_PREFIX}{tech}_{years[0]}"
                
                if convex_col in filtered_convex_data.columns:
                    convex_vals = filtered_convex_data[convex_col].dropna()
                    fig_val.add_trace(go.Scatter(
                        x=[years[0]] * len(convex_vals),
                        y=convex_vals,
                        mode='markers',
                        marker=dict(color="rgba(255,0,0,0.4)", size=10),
                        hoverinfo="skip",
                        showlegend=False
                    ), row=row, col=col)
            else:
               
                convex_cols = [f"{INSTALLED_CAPACITY_PREFIX}{tech}_{year}" for year in years]
                
                if all(col in filtered_convex_data.columns for col in convex_cols):
                    for idx in filtered_convex_data.index:
                        values = filtered_convex_data.loc[idx, convex_cols].values
                        if not np.isnan(values).all():
                            fig_val.add_trace(go.Scatter(
                                x=years,
                                y=values,
                                mode='lines',
                                line=dict(color="rgba(255,0,0,0.3)", width=1),
                                hoverinfo="skip",
                                showlegend=False
                            ), row=row, col=col)

        # Min/Max Bereich für aktuelle Vertices
        try:
            min_vals = full_values_matrix.min()
            max_vals = full_values_matrix.max()

            if len(years) == 1:
                year = years[0]
                x_vals = [year - 0.25, year + 0.25, year + 0.25, year - 0.25]
                y_vals = [min_vals.iloc[0], min_vals.iloc[0], max_vals.iloc[0], max_vals.iloc[0]]
            else:
                x_vals = list(years) + list(reversed(years))
                y_vals = min_vals.tolist() + max_vals.tolist()[::-1]

            fig_val.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                fill='toself',
                fillcolor="rgba(26, 102, 204, 0.15)",
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False
            ), row=row, col=col)
        except Exception as e:
            st.warning(f"⚠️ Error adding min/max fill for {tech}: {e}")

        # X-Achse bei Ein-Jahres-Daten
        if len(years) == 1:
            fig_val.update_xaxes(
                tickvals=[years[0]],
                ticktext=[str(years[0])],
                row=row,
                col=col
            )

        # Originalbereiche (rot)
        if show_original_ranges:
            try:
                original_matrix = vertex_df.loc[current_indices, cols]
                min_vals = original_matrix.min()
                max_vals = original_matrix.max()

                if len(years) == 1:
                    x_vals = [years[0] - 0.25, years[0] + 0.25, years[0] + 0.25, years[0] - 0.25]
                    y_vals = [min_vals.iloc[0], min_vals.iloc[0], max_vals.iloc[0], max_vals.iloc[0]]
                else:
                    x_vals = list(years) + list(reversed(years))
                    y_vals = min_vals.tolist() + max_vals.tolist()[::-1]

                fig_val.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    fill='toself',
                    fillcolor="rgba(255, 0, 0, 0.08)",
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    showlegend=False
                ), row=row, col=col)
            except Exception as e:
                st.write(f"❌ Error displaying original range for {tech}: {e}")

    # Layout
    fig_val.update_layout(
        height=fig_height_val * 100,
        width=fig_width_val * 100,
        title=dict(
            text=plot_title,
            font=dict(size=18, family="Arial", color="#333"),
            x=0,
            xanchor="left"
        ),
        font=dict(size=12, family="Arial", color="#333"),
        paper_bgcolor='#f4f4f4',
        plot_bgcolor='#f4f4f4',
        hovermode="closest",
        margin=dict(l=40, r=40, t=80, b=50),
        showlegend=False
    )

    # Achsenstyling
    for i in range(1, len(valid_techs_value) + 1):
        suffix = "" if i == 1 else str(i)
        xaxis = getattr(fig_val.layout, f"xaxis{suffix}", None)
        yaxis = getattr(fig_val.layout, f"yaxis{suffix}", None)

        if isinstance(xaxis, XAxis):
            xaxis.update(
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)",
                mirror=True,
                showline=True,
                linecolor="rgba(0,0,0,0.3)",
                linewidth=1,
                ticks="outside"
            )
        if isinstance(yaxis, YAxis):
            yaxis.update(
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)",
                mirror=True,
                showline=True,
                linecolor="rgba(0,0,0,0.3)",
                linewidth=1,
                ticks="outside"
            )

    for ann in fig_val['layout']['annotations']:
        ann['y'] += 0.01
        ann['font'] = dict(size=12, color='#222', family="Arial")

    st.plotly_chart(fig_val, use_container_width=True)

def prepare_vertex_selection(
    MAA_PREFIX,
    vertex_df,
    tech_time_map,
    extract_time_series_map,
    select_representative_vertices_by_kmeans,
    current_indices,
    max_plot_vertices,
    st
):
    """
    Bereitet die Vertex-Auswahl basierend auf dem MAA_PREFIX vor.
    Gibt: time_map, valid_techs, plot_indices, selected_tech, cols zurück
    """
    if MAA_PREFIX == "VALUE_":
        source = "value_time_map"
        mode = "operational"
        time_map = extract_time_series_map(vertex_df, MAA_PREFIX, mode=mode)
        valid_techs = sorted([tech for tech, v in time_map.items() if len(v) >= 1])
    elif MAA_PREFIX == "MAA_":
        source = "tech_time_map"
        time_map = tech_time_map
        valid_techs = sorted([tech for tech, v in time_map.items() if len(v) >= 1])
    else:
        st.error(f"❌ Unknown MAA_PREFIX: '{MAA_PREFIX}'. Expected 'VALUE_' or 'MAA_'.")
        return {}, [], [], None, []

    if not valid_techs:
        st.warning(f"⚠️ No valid technologies found in `{source}`.")
        return time_map, valid_techs, [], None, []

    tech = valid_techs[0]
    year_cols = time_map[tech]

    try:
        # Sortiere nach Jahr
        years_cols_sorted = sorted(year_cols, key=lambda x: x[0])

        if source == "value_time_map":
            cols = [c for y, c in years_cols_sorted if c.startswith(MAA_PREFIX + tech)]
        else:
            _, cols = zip(*years_cols_sorted)
            cols = list(cols)

        # Auswahl repräsentativer Vertices nur wenn nötig
        if len(current_indices) > max_plot_vertices:
            plot_indices = select_representative_vertices_by_kmeans(
                df=vertex_df,
                cols=cols,
                n_vertices=max_plot_vertices,
                index_subset=current_indices
            )
        else:
            plot_indices = current_indices

    except Exception as e:
        st.error(f"❌ Error processing technology '{tech}': {e}")
        return time_map, valid_techs, [], tech, []

    return time_map, valid_techs, plot_indices, tech, cols
    
# === Initialisiere Session State ===
def initialize_session_state():
    defaults = {
        'convex_combinations': pd.DataFrame(),
        'convex_additional': pd.DataFrame(),
        'show_convex': True,
        'show_original_ranges': False,          # <--- ergänzt
        'plot_type_selector': 'Violinplot',     # <--- ergänzt
        'stored_figures': [],                   # <--- ergänzt
        'excel_loaded': False,
        'excel_path': '',
        'excel_error': None,
        'n_cols_plots': 3,                      # Standardanzahl Subplot-Spalten
        'max_plot_vertices': 5,
        "column_ratio" : 0.3,
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
def apply_tech_filters(data, data_additional, session_state, ordered_techs, prefix):
    import streamlit as st

    if data.empty:
        
        return pd.DataFrame(), pd.DataFrame()

    # Sicherstellen, dass beide DataFrames die gleiche Indizierung haben
    data = data.copy()
    data_additional = data_additional.copy()
    data.index = data.index.astype(str)
    data_additional.index = data_additional.index.astype(str)
    common_indices = data.index.intersection(data_additional.index)
    data = data.loc[common_indices]
    data_additional = data_additional.loc[common_indices]

    filtered_indices = data.index

    for tech in ordered_techs:
        key = f"slider_{tech}"
        col_data = f"{prefix}{tech}"
        col_additional = tech

        if key in session_state:
            min_val, max_val = session_state[key]
            
            if col_data in data.columns:
                col_values = data.loc[filtered_indices, [col_data]]
                

                mask = (col_values[col_data] >= min_val) & (col_values[col_data] <= max_val)
                filtered_indices = filtered_indices[mask]

              

            elif col_additional in data_additional.columns:
                col_values = data_additional.loc[filtered_indices, [col_additional]]
               

                mask = (col_values[col_additional] >= min_val) & (col_values[col_additional] <= max_val)
                filtered_indices = filtered_indices[mask]

               

    return data.loc[filtered_indices].reset_index(drop=True), data_additional.loc[filtered_indices].reset_index(drop=True)

# === Titel & Initialisierung ===
st.title(" Technology Decision Tool")
# === Excel-Datei Ladebereich via Upload ===
DEFAULT_EXCEL_URL = "https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/vertex_results_default.xlsx"

@st.cache_data(show_spinner="📥 Load Default-Excel ...")
def read_default_excel_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("❌ Could not load the default file.")
    return pd.read_excel(BytesIO(response.content))



if not st.session_state.get("excel_loaded", False):

    st.subheader("📂 Select Excel File")
    col1, spacer, col2 = st.columns([2, 0.3, 1])

    with col1:
        upload_file = st.file_uploader("📤 Upload your own Excel file (.xlsx)", type=["xlsx"])
        # Upload in Session speichern, falls vorhanden
        if upload_file is not None:
            st.session_state["uploaded_file"] = upload_file
            st.session_state["use_default_excel"] = False

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📁 Use default Excel file"):
            st.session_state["use_default_excel"] = True
            # Entferne eventuell vorhandene Upload-Datei, damit keine Verwirrung entsteht
            if "uploaded_file" in st.session_state:
                del st.session_state["uploaded_file"]

    uploaded_file = st.session_state.get("uploaded_file", None)
    use_default_excel = st.session_state.get("use_default_excel", False)

    # Nur fortfahren, wenn eine der beiden Optionen gewählt wurde
    if use_default_excel or uploaded_file is not None:
        st.subheader("Excel Read-In method")

        option = st.selectbox(
            "Choose Option:",
            ["", "📥 Read-in all vertices", "📊 Apply clustering to retain representative vertices"]
        )

        if option == "📥 Read-in all vertices":
            if st.button("Read-in Excel File"):
                try:
                    df = read_default_excel_from_url(DEFAULT_EXCEL_URL) if use_default_excel else load_excel_data(uploaded_file)
                    st.session_state["uploaded_excel"] = df.copy()
                    st.session_state["excel_loaded"] = True
                    st.session_state["excel_error"] = None
                    st.rerun()
                except Exception as e:
                    st.session_state["excel_error"] = f"❌ Error while reading the file: {e}"

        elif option == "📊 Apply clustering to retain representative vertices":
            k_value = st.number_input(
                "Number of representative vertices to retain (KMeans)",
                min_value=50,
                max_value=5000,
                value=1000,
                step=50,
                key="clustering_k"
            )

            if st.button("Apply Clustering and Read-in"):
                try:
                    df = read_default_excel_from_url(DEFAULT_EXCEL_URL) if use_default_excel else load_excel_data(uploaded_file)
                    amount_vertices_requested = int(k_value)

                    coeff_columns = [col for col in df.columns if col.startswith("COEFF_")]
                    if not coeff_columns:
                        raise ValueError("❌ No COEFF_ columns found.")

                    last_coeff_col = coeff_columns[-1]
                    last_index_with_minus1 = df[df[last_coeff_col] == -1].index.max()
                    df_first_part = df.loc[:last_index_with_minus1].copy()
                    df_remaining = df.loc[last_index_with_minus1 + 1:].copy()

                    if amount_vertices_requested >= len(df):
                        st.session_state["uploaded_excel"] = df.copy()
                        st.session_state["excel_loaded"] = True
                        st.session_state["excel_error"] = None
                        st.rerun()

                    elif amount_vertices_requested <= len(df_first_part):
                        st.session_state["uploaded_excel"] = df_first_part.copy()
                        st.session_state["excel_loaded"] = True
                        st.session_state["excel_error"] = None
                        st.rerun()

                    else:
                        cluster_columns = [col for col in df.columns if col.startswith("VALUE_") or col.startswith("MAA_")]
                        df_remaining_unique = df_remaining.drop_duplicates(subset=cluster_columns)
                        remaining_target = amount_vertices_requested - len(df_first_part)

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
                    st.session_state["excel_error"] = f"❌ Error during clustering: {e}"

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

with st.sidebar.expander("Layout Options", expanded=True):
    
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
            value=0.3,
            step=0.01,
            key="column_ratio"
        )
    st.number_input(
        "Number of columns in plot layout",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        key="n_cols_plots"
    )

with st.sidebar.expander("Plot Options"):
    st.radio(
        "Choose plot type",
        options=["Line Plot", "Violin Plot"],
        index=0,
        key="plot_type_selector2"
    )
    if st.session_state.plot_type_selector2 == "Line Plot":
        st.number_input(
            "Max vertices to display in plots",
            min_value=1,
            max_value=total_vertices_available,
            value=min(5, total_vertices_available),
            step=1,
            key="max_plot_vertices"
        )
    
    st.checkbox("Show convex combinations in all plots", value=True, key="show_convex")
    st.checkbox("Show original flexibility ranges (red shaded)", value=False, key="show_original_ranges")
    st.checkbox("Enable density plots", key="show_density")  

with st.sidebar.expander("Additional Metrics"):
    st.radio(
        "Plot type for additional metrics",
        ["Violin Plot", "Scatter Plot"],
        index=0,
        key="plot_type_selector"
    )

# === Different Tabs ===
tab1, tab2,tab3 = st.tabs(["📊 Decision Tool", "❓ Explanation","📥 Download Results"])

# === Auswahl & Filter-UI ===
with tab1: 
    if st.session_state["layout_mode"] == "Two-column layout":
        ratio = st.session_state.get("column_ratio")
        col1, spacer, col2 = st.columns([0.5+ratio, 0.1, 1.5-ratio])
        with col1:
            st.markdown("### Select and Filter Technologies")
            st.caption(f"⚡️ **Note:** Proceed sequential.")
            col_select, col_reset = st.columns([4, 1])
            with col_select:
                all_filter_options = technologies + additional_cols
                selected_techs_raw = st.multiselect("Select variables to be constrained", all_filter_options)
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
        
            selected_techs = [t for t in ordered_techs if t in technologies]
            selected_metrics = [m for m in ordered_techs if m in additional_cols]
        
            selected_data = pd.DataFrame(index=tech_data.index)
            
            if selected_techs:
                
                selected_data = pd.concat([selected_data, tech_data[[f"{MAA_PREFIX}{t}" for t in selected_techs]]], axis=1)
            if selected_metrics:
                selected_data = pd.concat([selected_data, vertex_df.loc[tech_data.index, selected_metrics]], axis=1)
        
            current_indices = selected_data.index
            for i, tech in enumerate(ordered_techs):
                key = f"slider_{tech}"
                col = f"{MAA_PREFIX}{tech}" if tech in technologies else tech
        
                partial_indices = selected_data.index
                for j in range(i):
                    prev = ordered_techs[j]
                    prev_col = f"{MAA_PREFIX}{prev}" if prev in technologies else prev
                    prev_range = st.session_state.get(f"slider_{prev}", (selected_data[prev_col].min(), selected_data[prev_col].max()))
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
        
            with st.sidebar.expander("Convex Combination Settings"):
                st.number_input("Total number of convex combinations", min_value=10, max_value=10000, value=100, step=10, key="n_samples")
                max_vertices = len(current_indices) if not current_indices.empty else 0
                st.number_input("Vertices used per combination", min_value=2, max_value=max_vertices if max_vertices > 1 else 2,
                                value=max_vertices if max_vertices > 1 else 2, step=1, key="n_vertices_convex")
                st.selectbox("Dirichlet α (weight spread)", [0.01, 0.1, 1.0],
                             index=[0.01, 0.1, 1.0].index(st.session_state.get('alpha_value', 0.1)), key="alpha_value")
                st.number_input("Combinations per batch", min_value=1, max_value=st.session_state["n_samples"],
                                value=min(10, st.session_state["n_samples"]), step=1, key="n_batch_size")
        
                col_gen, col_reset = st.columns(2)
                with col_gen:
                    generate_convex = st.button("Generate", key="generate_convex_button_sidebar")
                with col_reset:
                    reset_convex = st.button("Reset", key="reset_convex_button_sidebar")
        
                if reset_convex:
                    st.session_state['convex_combinations'] = pd.DataFrame()
                    st.session_state['convex_additional'] = pd.DataFrame()
        
                if generate_convex and not current_indices.empty:
                    base_data_full = tech_data.loc[current_indices]
                    base_additional_data = vertex_df.loc[current_indices, additional_cols] if additional_cols else pd.DataFrame(index=current_indices)
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
        
                        if include_installed_capacity and not installed_data.empty:
                            installed_sample = installed_data.loc[base_sample.index]
                            batch_installed = weights @ installed_sample.values
                            all_installed_samples.append(pd.DataFrame(batch_installed, columns=installed_sample.columns))
        
                        all_samples.append(pd.DataFrame(batch_samples, columns=base_sample.columns))
        
                        if not base_additional_sample.empty:
                            batch_additional = weights @ base_additional_sample.values
                            all_additional_samples.append(pd.DataFrame(batch_additional, columns=base_additional_sample.columns))
        
                        if sum(len(df) for df in all_samples) >= n_total:
                            break
        
                    convex_df = pd.concat(all_samples, ignore_index=True)
                    st.session_state['convex_combinations'] = pd.concat([st.session_state['convex_combinations'], convex_df], ignore_index=True)
        
                    if include_installed_capacity and all_installed_samples:
                        installed_comb_df = pd.concat(all_installed_samples, ignore_index=True)
                        st.session_state['convex_combinations'][installed_data.columns] = installed_comb_df
        
                    if all_additional_samples:
                        additional_comb_df = pd.concat(all_additional_samples, ignore_index=True)
                        st.session_state['convex_additional'] = pd.concat(
                            [st.session_state.get('convex_additional', pd.DataFrame()), additional_comb_df], ignore_index=True
                        )
        
                    n_convex = len(st.session_state['convex_combinations'])
                    st.sidebar.info(f"**Currently {n_convex} convex combination(s)** generated.")
        
            filtered_convex_data, filtered_convex_additional = apply_tech_filters(
                st.session_state['convex_combinations'],
                st.session_state['convex_additional'],
                st.session_state,
                ordered_techs,
                prefix=MAA_PREFIX
            )
                
            time_map, valid_techs, plot_indices, tech,cols = prepare_vertex_selection(
                MAA_PREFIX=MAA_PREFIX,
                vertex_df=vertex_df,
                tech_time_map=tech_time_map,
                extract_time_series_map=extract_time_series_map,
                select_representative_vertices_by_kmeans=select_representative_vertices_by_kmeans,
                current_indices=current_indices,
                max_plot_vertices=st.session_state["max_plot_vertices"],
                st=st
            )
            
            if valid_techs:
                st.markdown("---")
                st.markdown("### Highlight Vertex & View Details")
                selected_vertex = select_and_show_vertex_info(
                    plot_indices=plot_indices,
                    current_indices=current_indices,
                    vertex_df=vertex_df,
                    tech_data=tech_data,
                    additional_cols=additional_cols
                )
                st.session_state["selected_vertex"] = (
                    selected_vertex if selected_vertex != "— Please select —" else None
                )
            else:
                selected_vertex = None
            
        
        with col2:        
            if MAA_PREFIX == "VALUE_":
                if st.session_state.get("plot_type_selector2") == "Line Plot":
                    st.markdown("### Operational Variables Over Time")
                    if len(current_indices) > st.session_state["max_plot_vertices"]:
                        plot_indices_val = select_representative_vertices_by_kmeans(
                            df=vertex_df,
                            cols=list(cols),  # <- Stelle sicher, dass `cols` korrekt definiert ist
                            n_vertices=st.session_state["max_plot_vertices"],
                            index_subset=current_indices
                        )
                        st.caption(f"⚡️ **Note:** Displaying a clustered sample of {st.session_state['max_plot_vertices']} out of {len(current_indices)} valid vertices.")
                    else:
                        plot_indices_val = current_indices
                        st.caption(f"⚡️ **Note:** {len(current_indices)} valid vertices remaining.")
                    value_time_map = extract_time_series_map(vertex_df, MAA_PREFIX, mode="operational")
                    
                    valid_techs_value = sorted([tech for tech, v in value_time_map.items() if len(v) >= 1])
                    
                    plot_operational_variables_over_time(
                        vertex_df=vertex_df,
                        current_indices=current_indices,
                        plot_indices_val=plot_indices_val,
                        time_column_map=value_time_map,  # statt value_time_map
                        selected_vertex=st.session_state.get("selected_vertex"),
                        n_cols_val=st.session_state.get("n_cols_plots", 3),
                        show_convex=st.session_state["show_convex"],
                        st_convex =st.session_state["convex_combinations"],
                        filtered_convex_data=filtered_convex_data,
                        show_original_ranges=st.session_state["show_original_ranges"],
                        max_plot_vertices=st.session_state["max_plot_vertices"],
                        maa_prefix=MAA_PREFIX,
                        apply_prefix=True,  # wichtig: MAA_ Prefix verwenden
                        plot_title="Operational Variables Over Time"
                    )


                else:
                    plot_violin_values(
                        vertex_df,
                        valid_techs_value,
                        value_time_map,
                        plot_indices_val,
                        current_indices,
                        filtered_convex_data,
                        MAA_PREFIX="MAA_"  # optional
                    )

                # === Installed Capacities Plot ===
            st.markdown("### Installed Capacities Over Time")

            # === Auswahl Mapping je nach Prefix
            source_map = extract_time_series_map(vertex_df, MAA_PREFIX, mode="installed")
                        
            # === Gültige Technologien + Auswahl
            valid_techs_all = sorted([tech for tech, v in source_map.items() if len(v) >= 1])
                        
            selected_techs = st.multiselect(
                "Select technologies to display:",
                options=valid_techs_all,
                default=valid_techs_all,
                help="Select one or more technologies to display."
            )
            
            valid_techs = selected_techs          
            
            # === Initiale Spaltenauswahl für KMeans
            if MAA_PREFIX == "VALUE_":
                cols = []
                for tech in valid_techs:
                    year_cols = source_map.get(tech, [])
                    cols.extend([col for _, col in year_cols])
                cols = list(set(cols))  # Doppelte entfernen
            else:
                # Fallback: Nur erste Technologie wie gehabt
                year_cols = source_map[valid_techs_all[0]]
                years_cols_sorted = sorted(year_cols, key=lambda x: x[0])
                try:
                    _, cols = zip(*years_cols_sorted)
                    cols = list(cols)
                   
                except ValueError:
                    st.error("❌ Could not extract columns from years_cols_sorted.")
                    st.stop()
            
            if not cols:
                st.error("❌ No matching columns found.")
                st.stop()
            
            # === Data Check vor KMeans
            subset_df = vertex_df.loc[current_indices, cols]
            
            
            # === Auswahl von Vertices mittels KMeans mit Fehlerbehandlung
            if MAA_PREFIX == "VALUE_":
                plot_indices=plot_indices_val
                if len(current_indices) > st.session_state["max_plot_vertices"] and st.session_state.get("plot_type_selector2") == "Line Plot":
                    st.caption(f"⚡️ **Note:** Displaying a clustered sample of {st.session_state['max_plot_vertices']} out of {len(current_indices)} valid vertices.")
                else:
                    st.caption(f"⚡️ **Note:** {len(current_indices)} valid vertices remaining.")

            
            else:
                try:
                    raw_plot_indices = select_representative_vertices_by_kmeans(
                        df=vertex_df,
                        cols=cols,
                        n_vertices=st.session_state["max_plot_vertices"],
                        index_subset=current_indices
                    )
                    plot_indices = clean_plot_indices(raw_plot_indices)
                    if len(current_indices) > st.session_state["max_plot_vertices"] and st.session_state.get("plot_type_selector2") == "Line Plot":
                        st.caption(f"⚡️ **Note:** Displaying a clustered sample of {st.session_state['max_plot_vertices']} out of {len(current_indices)} valid vertices.")
                    else:
                        st.caption(f"⚡️ **Note:** {len(current_indices)} valid vertices remaining.")
                except Exception as e:
                    st.exception(f"❌ Error selecting vertices using KMeans: {e}")
                    st.stop()
                    
            
            
            # === Plot-Erstellung
            if st.session_state.get("plot_type_selector2") == "Line Plot":
                st.markdown("### Installed Capacities Over Time")
            
                plot_operational_variables_over_time(
                    vertex_df=vertex_df,
                    current_indices=current_indices,
                    plot_indices_val=plot_indices,
                    time_column_map=source_map,
                    selected_vertex=st.session_state.get("selected_vertex"),
                    n_cols_val=st.session_state.get("n_cols_plots", 3),
                    show_convex=st.session_state["show_convex"],
                    st_convex =st.session_state["convex_combinations"],
                    filtered_convex_data=filtered_convex_data,
                    show_original_ranges=st.session_state.get("show_original_ranges", False),
                    apply_prefix=False,
                    plot_title="Installed Capacities Over Time"
                )

            else:
                n_techs = sum(1 for v in tech_time_map.values() if len(v) >= 1)
                n_rows = ceil(n_techs / st.session_state.get("n_cols_plots", 3))
                plot_width_per_col = 6
                plot_height_per_row = 3.5
                fig_width = plot_width_per_col * st.session_state.get("n_cols_plots", 3)
                fig_height = plot_height_per_row * n_rows
            
                fig, axes = plt.subplots(n_rows,st.session_state.get("n_cols_plots", 3), figsize=(fig_width, fig_height))
                fig.patch.set_facecolor('#f4f4f4')
                axes = axes.flatten() if n_techs > 1 else [axes]
                plot_idx = 0
                for tech, year_cols in sorted(tech_time_map.items()):
                    if len(year_cols) < 1:
                        continue
            
                    years_cols_sorted = sorted(year_cols, key=lambda x: x[0])
                    years = [y for y, _ in years_cols_sorted]
                    cols = [col for _, col in years_cols_sorted]
            
                    values_matrix = vertex_df.loc[plot_indices, cols]
                    if values_matrix.dropna(how='all').empty:
                        continue
            
                    # ==== Konvexe Kombinationen einbeziehen ====
                    if st.session_state.get('show_convex', False) and not st.session_state['convex_combinations'].empty:
                        convex_cols = [f"{INSTALLED_CAPACITY_PREFIX}{tech}_{year}" for year in years]
                        if all(col in filtered_convex_data.columns for col in convex_cols):
                            convex_matrix = filtered_convex_data[convex_cols]
                            values_matrix = pd.concat([values_matrix, convex_matrix], axis=0)
            
                    ax = axes[plot_idx]
                    ax.set_facecolor('#f0f0f0')
            
                    # ==== Violinplot-Daten vorbereiten ====
                    data = [values_matrix[col].dropna().values for col in cols]
            
                    if all(len(d) > 0 for d in data):
                        ax.violinplot(data, positions=years, showmeans=False, showmedians=True, widths=2.0)
            
                    # ==== Ursprünglicher Wertebereich als rote Fläche ====
                    if st.session_state.get('show_original_ranges', False):
                        try:
                            original_matrix = vertex_df.loc[tech_data.index, cols]
                        except Exception:
                            original_matrix = vertex_df[cols]
            
                        original_min = original_matrix.min()
                        original_max = original_matrix.max()
            
                        for y, omin, omax in zip(years, original_min, original_max):
                            if not np.isnan(omin) and not np.isnan(omax):
                                ax.fill_between([y - 0.4, y + 0.4], omin, omax, color=(1.0, 0.0, 0.0, 0.08))
            
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
            
                # ==== Legende ====
                if plot_idx > 0:
                    combined_line = mlines.Line2D([], [], color=(0.1, 0.4, 0.8), alpha=0.8, label='Values incl. Convex')
            
                    fig.legend(
                        [combined_line],
                        ['Values incl. Convex'],
                        loc='upper center',
                        bbox_to_anchor=(0.5, 1.2 - 0.02 * max(st.session_state.get("n_cols_plots", 3) - 2, 0)),
                        ncol=1,
                        frameon=True,
                        fancybox=True,
                        fontsize=14
                    )
            
                    fig.subplots_adjust(
                        top=1.14 - 0.02 * max(st.session_state.get("n_cols_plots", 3) - 2, 0),
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
            
                n_techs = sum(1 for v in tech_time_map.values() if len(v) >= 1)
                n_rows = ceil(n_techs / st.session_state.get("n_cols_plots", 3))
                plot_width_per_col = 6
                plot_height_per_row = 3.5
                fig_width = plot_width_per_col * st.session_state.get("n_cols_plots", 3)
                fig_height = plot_height_per_row * n_rows
            
                fig_dichte, axs = plt.subplots(n_rows, st.session_state.get("n_cols_plots", 3), figsize=(fig_width, fig_height))
                axs = axs.flatten()
                fig_dichte.patch.set_facecolor('#f4f4f4')
            
                # Gleiche Datenbasis wie oben
                plot_indices = st.session_state.get("plot_indices", current_indices)
                df_base = vertex_df.loc[plot_indices]
                if len(df_base) > max_vertices_for_density:
                    df_base = df_base.sample(n=max_vertices_for_density, random_state=42)
            
                i = 0  # manueller Index für Achsen
                for tech, year_cols in sorted(tech_time_map.items()):
                    if len(year_cols) < 1:
                        continue
            
                    year_cols = sorted(year_cols, key=lambda x: x[0])
                    years = [y for y, _ in year_cols]
                    cols = [col for _, col in year_cols]
            
                    if not all(c in df_base.columns for c in cols):
                        axs[i].set_visible(False)
                        i += 1
                        continue
            
                    df = df_base[cols]
                    if df.dropna(how='all').empty:
                        axs[i].set_visible(False)
                        i += 1
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
                        i += 1
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
            
                    i += 1
            
                for j in range(i, len(axs)):
                    fig_dichte.delaxes(axs[j])
            
                fig_dichte.subplots_adjust(
                    top=0.95,
                    bottom=0.07,
                    hspace=0.44,
                    wspace=0.3
                )
                st.pyplot(fig_dichte)
                
        
    else:
       
        st.markdown("### Select and Filter Technologies")
        
        col_select, col_reset = st.columns([4, 1])
        with col_select:
            all_filter_options = technologies + additional_cols
            selected_techs_raw = st.multiselect("Select variables to be constrained", all_filter_options)
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
    
        # Separiere Technologien und Metriken
        selected_techs = [t for t in ordered_techs if t in technologies]
        selected_metrics = [m for m in ordered_techs if m in additional_cols]
        
        # Basisdaten vorbereiten
        selected_data = pd.DataFrame(index=tech_data.index)
        if selected_techs:
            selected_data = pd.concat([selected_data, tech_data[[f"{MAA_PREFIX}{t}" for t in selected_techs]]], axis=1)
        if selected_metrics:
            selected_data = pd.concat([selected_data, vertex_df.loc[tech_data.index, selected_metrics]], axis=1)
        current_indices = selected_data.index
        for i, tech in enumerate(ordered_techs):
            key = f"slider_{tech}"
            col = f"{MAA_PREFIX}{tech}" if tech in technologies else tech
        
            # Vorherige Einschränkungen anwenden
            partial_indices = selected_data.index
            for j in range(i):
                prev = ordered_techs[j]
                prev_col = f"{MAA_PREFIX}{prev}" if prev in technologies else prev
                prev_range = st.session_state.get(f"slider_{prev}", (selected_data[prev_col].min(), selected_data[prev_col].max()))
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
        with st.sidebar.expander("Convex Combination Settings"):
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
        filtered_convex_data,filtered_convex_additional = apply_tech_filters(
            st.session_state['convex_combinations'],
            st.session_state['convex_additional'],
            st.session_state,
            ordered_techs,
            prefix=MAA_PREFIX
        )
    
        
       
        # === Highlight ===
        time_map, valid_techs, plot_indices, tech,cols = prepare_vertex_selection(
                MAA_PREFIX=MAA_PREFIX,
                vertex_df=vertex_df,
                tech_time_map=tech_time_map,
                extract_time_series_map=extract_time_series_map,
                select_representative_vertices_by_kmeans=select_representative_vertices_by_kmeans,
                current_indices=current_indices,
                max_plot_vertices=st.session_state["max_plot_vertices"],
                st=st
            )
            
        if valid_techs:
            st.markdown("---")
            st.markdown("### Highlight Vertex & View Details")
            selected_vertex = select_and_show_vertex_info(
                plot_indices=plot_indices,
                current_indices=current_indices,
                vertex_df=vertex_df,
                tech_data=tech_data,
                additional_cols=additional_cols
            )
            st.session_state["selected_vertex"] = (
                selected_vertex if selected_vertex != "— Please select —" else None
            )
        else:
            selected_vertex = None
        
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

            value_time_map = extract_time_series_map(vertex_df, MAA_PREFIX, mode="operational")
            
            n_techs_value = sum(1 for v in value_time_map.values() if len(v) >= 1)
            n_cols_val = st.session_state.get("n_cols_plots", 3)
            n_rows_value = ceil(n_techs_value / n_cols_val)
            
            fig_width_val = 9 * (1 / n_cols_val) * n_cols_val
            fig_height_val = fig_width_val
            
            if len(current_indices) > st.session_state["max_plot_vertices"] and st.session_state.get("plot_type_selector2") == "Line Plot":
                plot_indices_val = plot_indices
                st.caption(f"⚡️ **Note:** Displaying a clustered sample of {st.session_state['max_plot_vertices']} out of {len(current_indices)} valid vertices.")
            else:
                st.caption(f"⚡️ **Note:** {len(current_indices)} valid vertices remaining.")
                plot_indices_val = plot_indices
                
            if st.session_state.get("plot_type_selector2") == "Line Plot":
                selected_vertex = st.session_state.get("selected_vertex")
                
                fig_val = make_subplots(
                    rows=n_rows_value,
                    cols=n_cols_val,
                    subplot_titles=[tech.replace("_", " ").title() for tech in sorted(value_time_map.keys()) if len(value_time_map[tech]) >= 1],
                    horizontal_spacing=0.08,
                    vertical_spacing=0.09
                )
                
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
                
                    row, col = divmod(plot_idx_val, n_cols_val)
                    row += 1
                    col += 1
                
                    plot_mode = 'markers' if len(years) == 1 else 'lines'
                
                    for i in values_matrix.index:
                        values = values_matrix.loc[i].values
                        if len(values) != len(years):
                            st.warning(
                                f"⚠️ Mismatch for tech: **{tech}**\n"
                                f"- years: {years}\n"
                                f"- values: {values}\n"
                                f"- len(years): {len(years)}, len(values): {len(values)}"
                            )
                            continue
                
                        is_sel = selected_vertex is not None and i == selected_vertex
                        hover_text = f"<b>Vertex {i}</b><br>" + "<br>".join(
                            [f"{year}: {val:.2f}" for year, val in zip(years, values)]
                        )
                
                        fig_val.add_trace(go.Scatter(
                            x=years,
                            y=values,
                            mode=plot_mode,
                            line=dict(
                                color="rgba(26, 102, 204, 0.8)" if is_sel else "rgba(26, 102, 204, 0.3)",
                                width=4 if is_sel else 1
                            ),
                            text=[hover_text] * len(years),
                            hoverinfo='text',
                            showlegend=False
                        ), row=row, col=col)
                
                    if st.session_state['show_convex'] and not st.session_state['convex_combinations'].empty:
                        if all(col in filtered_convex_data.columns for col in cols):
                            for idx in filtered_convex_data.index:
                                values = filtered_convex_data.loc[idx, cols].values
                                if not np.isnan(values).all():
                                    fig_val.add_trace(go.Scatter(
                                        x=years,
                                        y=values,
                                        mode=plot_mode,
                                        line=dict(color="rgba(255, 50, 50, 0.4)"),
                                        hovertemplate='Year: %{x}<br>Convex: %{y}<extra></extra>',
                                        showlegend=False
                                    ), row=row, col=col)
                
                    fig_val.add_trace(go.Scatter(
                        x=list(years) + list(reversed(years)),
                        y=list(full_values_matrix.min()) + list(full_values_matrix.max())[::-1],
                        fill='toself',
                        fillcolor="rgba(26, 102, 204, 0.15)",
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo='skip',
                        showlegend=False
                    ), row=row, col=col)
                
                    if st.session_state['show_original_ranges']:
                        try:
                            original_matrix = vertex_df.loc[current_indices, cols]
                            min_vals = original_matrix.min()
                            max_vals = original_matrix.max()
                
                            if len(years) == 1:
                                year = years[0]
                                x_vals = [year - 0.25, year + 0.25, year + 0.25, year - 0.25]
                                y_vals = [min_vals.iloc[0], min_vals.iloc[0], max_vals.iloc[0], max_vals.iloc[0]]
                
                                fig_val.update_xaxes(
                                    tickvals=[year],
                                    ticktext=[str(year)],
                                    row=row,
                                    col=col
                                )
                            else:
                                x_vals = list(years) + list(reversed(years))
                                y_vals = min_vals.tolist() + max_vals.tolist()[::-1]
                
                            fig_val.add_trace(go.Scatter(
                                x=x_vals,
                                y=y_vals,
                                fill='toself',
                                fillcolor="rgba(255, 0, 0, 0.08)",
                                line=dict(color='rgba(255,255,255,0)'),
                                hoverinfo='skip',
                                showlegend=False
                            ), row=row, col=col)
                        except Exception as e:
                            st.write(f"❌ Error displaying original range for {tech}: {e}")
                
                    plot_idx_val += 1
                
                fig_val.update_layout(
                    height=fig_height_val * 100,
                    width=fig_width_val * 100,
                    title=dict(
                        text="Operational Variables Over Time",
                        font=dict(size=18, family="Arial", color="#333"),
                        x=0,
                        xanchor="left"
                    ),
                    font=dict(size=12, family="Arial", color="#333"),
                    paper_bgcolor='#f4f4f4',
                    plot_bgcolor='#f4f4f4',
                    hovermode="closest",
                    margin=dict(l=40, r=40, t=80, b=50),
                    showlegend=False
                )
                
                for i in range(1, plot_idx_val + 1):
                    suffix = "" if i == 1 else str(i)
                    xaxis = getattr(fig_val.layout, f"xaxis{suffix}", None)
                    yaxis = getattr(fig_val.layout, f"yaxis{suffix}", None)
                
                    if isinstance(xaxis, XAxis):
                        xaxis.update(
                            showgrid=True,
                            gridcolor="rgba(0,0,0,0.1)",
                            mirror=True,
                            showline=True,
                            linecolor="rgba(0,0,0,0.3)",
                            linewidth=1,
                            ticks="outside"
                        )
                    if isinstance(yaxis, YAxis):
                        yaxis.update(
                            showgrid=True,
                            gridcolor="rgba(0,0,0,0.1)",
                            mirror=True,
                            showline=True,
                            linecolor="rgba(0,0,0,0.3)",
                            linewidth=1,
                            ticks="outside"
                        )
                
                for ann in fig_val['layout']['annotations']:
                    ann['y'] += 0.01
                    ann['font'] = dict(size=12, color='#222', family="Arial")
                
                st.plotly_chart(fig_val, use_container_width=True)
            else:
                # === Layout definieren ===
                n_techs_value = sum(1 for v in value_time_map.values() if len(v) >= 1)
                n_cols = st.session_state.get("n_cols_plots", 3)
                n_rows_value = ceil(n_techs_value / n_cols)
                
                plot_width_per_col = 6
                plot_height_per_row = 3.5
                fig_width = plot_width_per_col * n_cols
                fig_height = plot_height_per_row * n_rows_value
                
                fig_value, axes_value = plt.subplots(n_rows_value, n_cols, figsize=(fig_width, fig_height))
                fig_value.patch.set_facecolor('#f4f4f4')
                axes_value = axes_value.flatten() if n_techs_value > 1 else [axes_value]
                
                # === Plotten ===
                plot_idx_val = 0
                for tech, year_cols in sorted(value_time_map.items()):
                    if len(year_cols) < 1 or not any(col.startswith(MAA_PREFIX + tech) for _, col in year_cols):
                        continue
                
                    years_cols_sorted = sorted(year_cols, key=lambda x: x[0])
                    years = [y for y, _ in years_cols_sorted]
                    cols = [col for _, col in years_cols_sorted if col.startswith(MAA_PREFIX + tech)]
                
                    if not cols:
                        continue
                
                    values_matrix = vertex_df.loc[plot_indices_val, cols]
                
                    # === Konvexe Kombinationen einbeziehen ===
                    if st.session_state.get('show_convex', False) and not st.session_state['convex_combinations'].empty:
                        if all(col in filtered_convex_data.columns for col in cols):
                            convex_matrix = filtered_convex_data[cols]
                            values_matrix = pd.concat([values_matrix, convex_matrix], axis=0)
                
                    if values_matrix.dropna(how='all').empty:
                        continue
                
                    ax = axes_value[plot_idx_val]
                    ax.set_facecolor('#f0f0f0')
                
                    # === Violinplot-Daten vorbereiten ===
                    data = [values_matrix[col].dropna().values for col in cols]
                
                    if all(len(d) > 0 for d in data):
                        ax.violinplot(data, positions=years, showmeans=False, showmedians=True, widths=2.0)
                
                        # === Vertex Highlight als Punkt ===
                        selected_vertex = st.session_state.get("selected_vertex", None)
                        if selected_vertex is not None and selected_vertex in vertex_df.index:
                            try:
                                highlight_vals = vertex_df.loc[selected_vertex, cols]
                                for y, val in zip(years, highlight_vals):
                                    if pd.notnull(val):
                                        show_label = plot_idx_val == 0  # nur beim ersten Plot in Legende
                                        ax.scatter(y, val, color="black", s=60, zorder=3,
                                                   label="Selected Vertex" if show_label else None)
                            except Exception as e:
                                st.warning(f"❌ Error drawing highlighted vertex for {tech}: {e}")
                
                    # === Ursprünglicher Wertebereich (rote Fläche) ===
                    if st.session_state.get('show_original_ranges', False):
                        try:
                            original_matrix = vertex_df.loc[tech_data.index, cols]
                        except Exception:
                            original_matrix = vertex_df[cols]
                
                        original_min = original_matrix.min()
                        original_max = original_matrix.max()
                
                        for y, omin, omax in zip(years, original_min, original_max):
                            if not np.isnan(omin) and not np.isnan(omax):
                                ax.fill_between([y - 0.4, y + 0.4], omin, omax, color=(1.0, 0.0, 0.0, 0.08))
                
                    # === Achsenbeschriftung & Formatierung ===
                    ax.set_title(tech.replace('_', ' ').title())
                    ax.set_xticks(years)
                    ax.set_xticklabels([str(y) for y in years])
                
                    if plot_idx_val >= (n_rows_value - 1) * n_cols:
                        ax.set_xlabel("Year")
                    if plot_idx_val % n_cols == 0:
                        ax.set_ylabel("VALUE_")
                    ax.grid(True, linestyle="--", alpha=0.4)
                
                    if plot_idx_val == 0:
                        handles_labels_val = ax.get_legend_handles_labels()
                
                    plot_idx_val += 1
                
                # === Leere Subplots entfernen ===
                for i in range(plot_idx_val, len(axes_value)):
                    if axes_value[i] in fig_value.axes:
                        fig_value.delaxes(axes_value[i])
                
                # === Legende einfügen ===
                if plot_idx_val > 0:
                    legend_items = []
                
                    # Basislinie
                    combined_line = mlines.Line2D([], [], color=(0.1, 0.4, 0.8), alpha=0.8, label='Values incl. Convex')
                    legend_items.append(combined_line)
                
                    # Highlight-Punkt
                    if st.session_state.get("selected_vertex") is not None:
                        highlight_point = mlines.Line2D([], [], color="black", marker='o', linestyle='None',
                                                        markersize=8, label="Selected Vertex")
                        legend_items.append(highlight_point)
                
                    fig_value.legend(
                        legend_items,
                        [line.get_label() for line in legend_items],
                        loc='upper center',
                        bbox_to_anchor=(0.5, 1.2 - 0.02 * max(n_cols - 2, 0)),
                        ncol=1,
                        frameon=True,
                        fancybox=True,
                        fontsize=14
                    )
                
                    fig_value.subplots_adjust(
                        top=1.14 - 0.02 * max(n_cols - 2, 0),
                        hspace=0.3,
                        wspace=0.18
                    )
                
                st.pyplot(fig_value)
                
        st.markdown("### Installed Capacities Over Time")
    
        n_techs = sum(1 for v in tech_time_map.values() if len(v) >= 1)
        n_rows = ceil(n_techs / st.session_state.get("n_cols_plots", 3))
        plot_width_per_col = 6
        plot_height_per_row = 3.5
        fig_width = plot_width_per_col * st.session_state.get("n_cols_plots", 3)
        fig_height = plot_height_per_row * n_rows
    
        fig, axes = plt.subplots(n_rows,st.session_state.get("n_cols_plots", 3), figsize=(fig_width, fig_height))
        fig.patch.set_facecolor('#f4f4f4')
        axes = axes.flatten() if n_techs > 1 else [axes]
    
        
                
    
        
        if st.session_state.get("plot_type_selector2") == "Line Plot":
            st.markdown("### Installed Capacities Over Time")
        
            n_techs = sum(1 for v in tech_time_map.values() if len(v) >= 1)
            n_cols = st.session_state.get("n_cols_plots", 3)
            n_rows = ceil(n_techs / n_cols)
            fig_width = 9 * (1 / n_cols) * n_cols
            fig_height = fig_width
        
            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                subplot_titles=[tech.replace("_", " ").title() for tech in tech_time_map if len(tech_time_map[tech]) >= 1],
                horizontal_spacing=0.08,
                vertical_spacing=0.09
            )
        
            if len(current_indices) > st.session_state["max_plot_vertices"] and st.session_state.get("plot_type_selector2") == "Line Plot":
                
                st.caption(f"⚡️ **Note:** Displaying a clustered sample of {st.session_state['max_plot_vertices']} out of {len(current_indices)} valid vertices.")
            else:
                st.caption(f"⚡️ **Note:** {len(current_indices)} valid vertices remaining.")
        
            selected_vertex = st.session_state.get("selected_vertex")
            plot_idx = 0
        
            for tech, year_cols in sorted(tech_time_map.items()):
                if len(year_cols) < 1:
                    continue
        
                years_cols_sorted = sorted(year_cols, key=lambda x: x[0])
                years = [y for y, _ in years_cols_sorted]
                cols = [c for _, c in years_cols_sorted]
        
                full_values_matrix = vertex_df.loc[current_indices, cols]
                values_matrix = vertex_df.loc[plot_indices, cols]
        
                row, col = divmod(plot_idx, n_cols)
                row += 1
                col += 1
        
                if values_matrix.dropna(how='all').empty:
                    plot_idx += 1
                    continue
        
                for idx in values_matrix.index:
                    values = values_matrix.loc[idx].values
                    if np.all(np.isnan(values)):
                        continue
        
                    is_sel = selected_vertex is not None and idx == selected_vertex
                    hover_text = f"<b>Vertex {idx}</b><br>" + "<br>".join(
                        [f"{year}: {float(val):.2f}" if pd.notnull(val) else f"{year}: n/a"
                         for year, val in zip(years, values)]
                    )
        
                    fig.add_trace(go.Scatter(
                        x=years,
                        y=values,
                        mode='lines' if len(years) > 1 else 'markers',
                        line=dict(
                            color="rgba(26,102,204,0.8)" if is_sel else "rgba(26,102,204,0.3)",
                            width=4 if is_sel else 1
                        ) if len(years) > 1 else None,
                        marker=dict(
                            color="rgba(26,102,204,0.8)" if is_sel else "rgba(26,102,204,0.3)",
                            size=10
                        ) if len(years) == 1 else None,
                        text=[hover_text] * len(years),
                        hoverinfo='text',
                        showlegend=False
                    ), row=row, col=col)
        
                # Konvexe Kombinationen
                if st.session_state.get('show_convex', False) and not st.session_state['convex_combinations'].empty:
                    if len(years) == 1:
                        convex_col = f"{INSTALLED_CAPACITY_PREFIX}{tech}_{years[0]}"
                        if convex_col in filtered_convex_data.columns:
                            convex_vals = filtered_convex_data[convex_col].dropna()
                            fig.add_trace(go.Scatter(
                                x=[years[0]] * len(convex_vals),
                                y=convex_vals,
                                mode='markers',
                                marker=dict(color="rgba(255,0,0,0.4)", size=10),
                                hoverinfo="skip",
                                showlegend=False
                            ), row=row, col=col)
                    else:
                        convex_cols = [f"{INSTALLED_CAPACITY_PREFIX}{tech}_{year}" for year in years]
                        if all(col in filtered_convex_data.columns for col in convex_cols):
                            for idx in filtered_convex_data.index:
                                values = filtered_convex_data.loc[idx, convex_cols].values
                                if not np.isnan(values).all():
                                    fig.add_trace(go.Scatter(
                                        x=years,
                                        y=values,
                                        mode='lines',
                                        line=dict(color="rgba(255,0,0,0.3)", width=1),
                                        hoverinfo="skip",
                                        showlegend=False
                                    ), row=row, col=col)
        
                # Min/Max Bereich für aktuelle Vertices
                try:
                    min_vals = full_values_matrix.min()
                    max_vals = full_values_matrix.max()
        
                    if len(years) == 1:
                        year = years[0]
                        x_vals = [year - 0.25, year + 0.25, year + 0.25, year - 0.25]
                        y_vals = [min_vals.iloc[0], min_vals.iloc[0], max_vals.iloc[0], max_vals.iloc[0]]
                    else:
                        x_vals = list(years) + list(reversed(years))
                        y_vals = min_vals.tolist() + max_vals.tolist()[::-1]
        
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        fill='toself',
                        fillcolor="rgba(26,102,204,0.15)",
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo='skip',
                        showlegend=False
                    ), row=row, col=col)
                except Exception as e:
                    st.warning(f"❌ Error plotting min/max ranges for {tech}: {e}")
        
                # Originalbereiche anzeigen (z.B. für alle Datenpunkte, nicht nur aktuelle Auswahl)
                if st.session_state.get('show_original_ranges', False):
                    try:
                        original_matrix = vertex_df.loc[tech_data.index, cols]
                        orig_min = original_matrix.min()
                        orig_max = original_matrix.max()
        
                        if len(years) == 1:
                            year = years[0]
                            x_vals = [year - 0.25, year + 0.25, year + 0.25, year - 0.25]
                            y_vals = [orig_min.iloc[0], orig_min.iloc[0], orig_max.iloc[0], orig_max.iloc[0]]
                        else:
                            x_vals = list(years) + list(reversed(years))
                            y_vals = orig_min.tolist() + orig_max.tolist()[::-1]
        
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            fill='toself',
                            fillcolor="rgba(255,0,0,0.08)",
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo='skip',
                            showlegend=False
                        ), row=row, col=col)
                    except Exception as e:
                        st.warning(f"❌ Error displaying original range for {tech}: {e}")
        
                plot_idx += 1
        
            # Layout
            fig.update_layout(
                height=fig_height * 100,
                width=fig_width * 100,
                title=dict(
                    text="Installed Capacities Over Time",
                    font=dict(size=18, family="Arial", color="#333"),
                    x=0,
                    xanchor="left"
                ),
                font=dict(size=12, family="Arial", color="#333"),
                paper_bgcolor='#f4f4f4',
                plot_bgcolor='#f4f4f4',
                hovermode="closest",
                margin=dict(l=40, r=40, t=80, b=50),
                showlegend=False
            )
        
            # Achsenstyling + X-Achsen auf Jahresbeschriftung setzen
            for i in range(1, plot_idx + 1):
                suffix = "" if i == 1 else str(i)
                xaxis = getattr(fig.layout, f"xaxis{suffix}", None)
                yaxis = getattr(fig.layout, f"yaxis{suffix}", None)
            
                if xaxis:
                    # Tick-Labels auf echte Jahre begrenzen
                    subplot_years = [y for tech, year_cols in sorted(tech_time_map.items())
                                     for y, _ in year_cols][:1]  # fallback bei Fehler
                    if plot_idx >= i:
                        tech_idx = i - 1
                        subplot_tech = list(sorted(tech_time_map.keys()))[tech_idx]
                        subplot_years = [y for y, _ in sorted(tech_time_map[subplot_tech])]
            
                    xaxis.update(
                        tickvals=subplot_years,
                        ticktext=[str(y) for y in subplot_years],
                        showgrid=True,
                        gridcolor="rgba(0,0,0,0.1)",
                        mirror=True,
                        showline=True,
                        linecolor="rgba(0,0,0,0.3)",
                        linewidth=1,
                        ticks="outside"
                    )
                if yaxis:
                    yaxis.update(
                        showgrid=True,
                        gridcolor="rgba(0,0,0,0.1)",
                        mirror=True,
                        showline=True,
                        linecolor="rgba(0,0,0,0.3)",
                        linewidth=1,
                        ticks="outside"
                    )
        
            # Annotation-Titel etwas nach oben verschieben
            for ann in fig['layout']['annotations']:
                ann['y'] += 0.01
                ann['font'] = dict(size=12, color='#222', family="Arial")
        
            # Anzeigen
            st.plotly_chart(fig, use_container_width=True)
        else:
            plot_idx = 0
            for tech, year_cols in sorted(tech_time_map.items()):
                if len(year_cols) < 1:
                    continue
            
                years_cols_sorted = sorted(year_cols, key=lambda x: x[0])
                years = [y for y, _ in years_cols_sorted]
                cols = [col for _, col in years_cols_sorted]
            
                values_matrix = vertex_df.loc[plot_indices, cols]
                if values_matrix.dropna(how='all').empty:
                    continue
            
                # ==== Konvexe Kombinationen einbeziehen ====
                if st.session_state.get('show_convex', False) and not st.session_state['convex_combinations'].empty:
                    convex_cols = [f"{INSTALLED_CAPACITY_PREFIX}{tech}_{year}" for year in years]
                    if all(col in filtered_convex_data.columns for col in convex_cols):
                        convex_matrix = filtered_convex_data[convex_cols]
                        values_matrix = pd.concat([values_matrix, convex_matrix], axis=0)
            
                ax = axes[plot_idx]
                ax.set_facecolor('#f0f0f0')
            
                # ==== Violinplot-Daten ====
                data = [values_matrix[col].dropna().values for col in cols]
            
                if all(len(d) > 0 for d in data):
                    ax.violinplot(data, positions=years, showmeans=False, showmedians=True, widths=2.0)
            
                    # ==== Selektierter Vertex als Punkt ====
                    selected_vertex = st.session_state.get("selected_vertex", None)
                    if selected_vertex is not None and selected_vertex in vertex_df.index:
                        try:
                            highlight_vals = vertex_df.loc[selected_vertex, cols]
                            for y, val in zip(years, highlight_vals):
                                if pd.notnull(val):
                                    show_label = plot_idx == 0  # Nur erstes Subplot zeigt Legenden-Eintrag
                                    ax.scatter(y, val, color="black", s=60, zorder=3,
                                               label="Selected Vertex" if show_label else None)
                        except Exception as e:
                            st.warning(f"❌ Error drawing highlighted vertex for {tech}: {e}")
            
                # ==== Ursprünglicher Wertebereich (rote Fläche) ====
                if st.session_state.get('show_original_ranges', False):
                    try:
                        original_matrix = vertex_df.loc[tech_data.index, cols]
                    except Exception:
                        original_matrix = vertex_df[cols]
            
                    original_min = original_matrix.min()
                    original_max = original_matrix.max()
            
                    for y, omin, omax in zip(years, original_min, original_max):
                        if not np.isnan(omin) and not np.isnan(omax):
                            ax.fill_between([y - 0.4, y + 0.4], omin, omax, color=(1.0, 0.0, 0.0, 0.08))
            
                ax.set_title(tech.replace('_', ' ').title())
                ax.set_xticks(years)
                ax.set_xticklabels([str(y) for y in years])
            
                if plot_idx >= (n_rows - 1) * st.session_state.get("n_cols_plots", 3):
                    ax.set_xlabel("Year")
                if plot_idx % st.session_state.get("n_cols_plots", 3) == 0:
                    ax.set_ylabel("Installed Capacity")
                ax.grid(True, linestyle="--", alpha=0.4)
            
                if plot_idx == 0:
                    handles_labels = ax.get_legend_handles_labels()
            
                plot_idx += 1
            
            # ==== Leere Achsen entfernen ====
            for i in range(plot_idx, len(axes)):
                if axes[i] in fig.axes:
                    fig.delaxes(axes[i])
            
            # ==== Legende ====
            if plot_idx > 0:
                legend_items = []
            
                # Konvexe Daten-Linie
                combined_line = mlines.Line2D([], [], color=(0.1, 0.4, 0.8), alpha=0.8, label='Values incl. Convex')
                legend_items.append(combined_line)
            
                # Highlight-Punkt, falls vorhanden
                if st.session_state.get("selected_vertex") is not None:
                    highlight_point = mlines.Line2D([], [], color="black", marker='o', linestyle='None',
                                                    markersize=8, label="Selected Vertex")
                    legend_items.append(highlight_point)
            
                fig.legend(
                    legend_items,
                    [line.get_label() for line in legend_items],
                    loc='upper center',
                    bbox_to_anchor=(0.5, 1.2 - 0.02 * max(st.session_state.get("n_cols_plots", 3) - 2, 0)),
                    ncol=1,
                    frameon=True,
                    fancybox=True,
                    fontsize=14
                )
            
                fig.subplots_adjust(
                    top=1.14 - 0.02 * max(st.session_state.get("n_cols_plots", 3) - 2, 0),
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
           
        
    st.divider()
    # === Weitere Metriken anzeigen ===
    # Beispiel-Daten
    st.markdown("### Additional Metrics")

    if additional_cols:
        # === Trennung in Jahresmetriken und Einzelmetriken ===
        yearly_metrics = [col for col in additional_cols if re.search(r"\b\d{4}\b", col)]
        single_metrics = [col for col in additional_cols if not re.search(r"\b\d{4}\b", col)]
    
        # === Gruppierung der Jahresmetriken nach Basismetrik ===
        base_metric_dict = defaultdict(list)
        for col in yearly_metrics:
            match = re.search(r"\b(\d{4})\b", col)
            if match:
                year = int(match.group(1))
                base_name = re.sub(r"\b\d{4}\b", "", col).strip(" _()-")
                base_metric_dict[base_name].append((year, col))
    
        # === Dropdown für beide Typen
        options_dropdown = (
            [f"🔹 {m}" for m in single_metrics] +
            [f"📈 {base}" for base in sorted(base_metric_dict.keys())]
        )
        default_selection = options_dropdown[:3] if len(options_dropdown) >= 3 else options_dropdown

        selected_combined = st.multiselect(
            "📊 Select metrics to plot",
            options_dropdown,
            default=default_selection
        )
    
        # === Auswahl trennen
        selected_single = [item.replace("🔹 ", "") for item in selected_combined if item.startswith("🔹")]
        selected_base_metrics = [item.replace("📈 ", "") for item in selected_combined if item.startswith("📈")]
    
        total_plots = len(selected_single) + len(selected_base_metrics)
    
        if total_plots > 0:
            max_cols = 3
            n_cols = min(max_cols, total_plots)
            n_rows = -(-total_plots // max_cols)
    
            fig_combined, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
            fig_combined.patch.set_facecolor('#f4f4f4')
            axes = axes.flatten() if total_plots > 1 else [axes]
            ax_idx = 0
    
            # === Einzelmetriken ===
            if selected_single:
                additional_data = vertex_df.loc[tech_data.index, selected_single]
                filtered_additional = additional_data.loc[current_indices]
    
                if st.session_state.get("show_convex") and not filtered_convex_additional.empty:
                    filtered_combined = pd.concat(
                        [filtered_additional, filtered_convex_additional[selected_single]],
                        axis=0
                    )
                else:
                    filtered_combined = filtered_additional
    
                for col in selected_single:
                    ax = axes[ax_idx]
                    ax.set_facecolor('#f0f0f0')
                    values = filtered_combined[col].dropna().values
                    if len(values) == 0:
                        ax.set_visible(False)
                        ax_idx += 1
                        continue
    
                    if st.session_state.get("show_original_ranges", False):
                        try:
                            original_values = vertex_df[col].dropna()
                            omin, omax = original_values.min(), original_values.max()
                            ax.fill_between([1 - 0.4, 1 + 0.4], omin, omax, color=(1.0, 0.0, 0.0, 0.08), zorder=1)
                        except:
                            pass
    
                    if st.session_state.get("plot_type_selector") == "Violin Plot":
                        vp = ax.violinplot([values], positions=[1], showmeans=False, showmedians=True, showextrema=True, widths=0.8)
                        for pc in vp['bodies']:
                            pc.set_facecolor((0.1, 0.4, 0.8, 0.7))
                            pc.set_edgecolor('black')
                            pc.set_alpha(0.7)
                        if 'cmedians' in vp:
                            vp['cmedians'].set_color('black')
                        ax.set_xlim(0.5, 1.5)
                        ax.set_xticks([])
    
                        # Konvexwerte (falls aktiv)
                        if st.session_state.get("show_convex") and not filtered_convex_additional.empty:
                            if col in filtered_convex_additional.columns:
                                cvals = filtered_convex_additional[col].dropna().values
                                ax.scatter([1] * len(cvals), cvals, color='crimson', alpha=0.5, marker='x', label='Convex')
    
                    else:
                        x_vals = [0] * len(values)
                        ax.scatter(x_vals, values, alpha=0.7, color="#444444")
                        if st.session_state.get("show_convex") and not filtered_convex_additional.empty:
                            if col in filtered_convex_additional.columns:
                                cx_vals = [0] * len(filtered_convex_additional[col].dropna())
                                cy_vals = filtered_convex_additional[col].dropna().values
                                ax.scatter(cx_vals, cy_vals, alpha=0.5, color='crimson', marker='x', label='Convex')
                        ax.set_xlim(-0.5, 0.5)
                        ax.set_xticks([])
    
                    ax.set_title(col, fontsize=12)
                    ax.set_ylabel("Value")
                    ax.grid(True, linestyle="--", alpha=0.4)
                    ax_idx += 1
    
            # === Jahresmetriken ===
            # === Jahresmetriken ===
            for base in selected_base_metrics:
                ax = axes[ax_idx]
                ax.set_facecolor('#f0f0f0')
            
                entries = sorted(base_metric_dict[base])
                years = [year for year, _ in entries]
                columns = [col for _, col in entries]
            
                           
                data = vertex_df.loc[tech_data.index, columns]
                filtered_data = data.loc[current_indices]
            
                # === Konvexe Daten: KEIN current_indices!
                available_cols = [col for col in columns if col in filtered_convex_additional.columns]
                convex_data_available = (
                    st.session_state.get("show_convex") and
                    not filtered_convex_additional.empty and
                    len(available_cols) > 0
                )
                if convex_data_available:
                    filtered_convex_data = filtered_convex_additional[available_cols]  # Keine .loc[current_indices]
            
                if st.session_state.get("plot_type_selector") == "Violin Plot":
                    year_values = [filtered_data[col].dropna().values for col in columns]
            
                    if convex_data_available:
                        convex_year_values = [filtered_convex_data[col].dropna().values for col in available_cols]
            
                    if st.session_state.get("show_original_ranges", False):
                        for i, col in enumerate(columns):
                            try:
                                ovals = vertex_df[col].dropna()
                                omin, omax = ovals.min(), ovals.max()
                                ax.fill_between([i + 0.6, i + 1.4], omin, omax, color=(1.0, 0.0, 0.0, 0.08), zorder=1)
                            except:
                                pass
            
                    vp = ax.violinplot(year_values, positions=range(1, len(years) + 1), showmeans=False, showmedians=True, showextrema=True)
                    for pc in vp['bodies']:
                        pc.set_facecolor((0.1, 0.4, 0.8, 0.7))
                        pc.set_edgecolor('black')
                        pc.set_alpha(0.7)
                    if 'cmedians' in vp:
                        vp['cmedians'].set_color('black')
            
                    if convex_data_available:
                        for i, cvals in enumerate(convex_year_values):
                            ax.scatter([i + 1] * len(cvals), cvals, color='crimson', alpha=0.5, marker='x', label='Convex' if i == 0 else "")
            
                    ax.set_xticks(range(1, len(years) + 1))
                    ax.set_xticklabels([str(y) for y in years])
            
                else:  # Scatter für Jahresmetriken
                    for i, col in enumerate(columns):
                        values = filtered_data[col].dropna().values
                        x_vals = [i + 1] * len(values)
                        ax.scatter(x_vals, values, alpha=0.7, color="#444444")
            
                        if convex_data_available and col in filtered_convex_data.columns:
                            cvals = filtered_convex_data[col].dropna().values
                            cx_vals = [i + 1] * len(cvals)
                            ax.scatter(cx_vals, cvals, alpha=0.5, color='crimson', marker='x')
            
                        if st.session_state.get("show_original_ranges", False):
                            try:
                                ovals = vertex_df[col].dropna()
                                omin, omax = ovals.min(), ovals.max()
                                ax.fill_between([i + 0.8, i + 1.2], omin, omax, color=(1.0, 0.0, 0.0, 0.08))
                            except:
                                pass
            
                    ax.set_xticks(range(1, len(years) + 1))
                    ax.set_xticklabels([str(y) for y in years])
            
                ax.set_title(base, fontsize=12)
                ax.set_xlabel("Year")
                ax.set_ylabel("Value")
                ax.grid(True, linestyle="--", alpha=0.4)
                ax_idx += 1
    
            for i in range(ax_idx, len(axes)):
                axes[i].set_visible(False)
    
            plt.tight_layout()
            st.pyplot(fig_combined)
            st.session_state["stored_figures"].append(("Combined_Metric_Subplots", fig_combined))
    
    else:
        st.info("No numeric columns found after the last 'NEW_CAPACITY' column.")
        
        
    # === Daten als Tabelle anzeigen ===
    st.divider()
    st.markdown("### Show remaining vertices as table")
    
    # === Original-Vertices ===
    st.markdown("#### Original Vertices")
    
    # Falls keine Daten vorhanden sind
    filtered_full_data = tech_data.loc[current_indices] if not current_indices.empty else pd.DataFrame()
    
    if filtered_data.empty:
        st.dataframe(filtered_full_data, use_container_width=True)
    else:
        frames_to_concat = [filtered_full_data]  # Index bleibt erhalten
    
        # Füge Installed Capacity-Spalten hinzu, falls VALUE_-Modus
        if MAA_PREFIX == "VALUE_":
            installed_cols = [col for col in vertex_df.columns if col.startswith(INSTALLED_CAPACITY_PREFIX)]
            installed_part = vertex_df.loc[filtered_full_data.index, installed_cols]
            frames_to_concat.append(installed_part)  # KEIN reset_index
    
        # Füge zusätzliche Metriken hinzu
        if additional_cols:
            additional_metrics_part = vertex_df.loc[filtered_full_data.index, additional_cols]
            frames_to_concat.append(additional_metrics_part)  # KEIN reset_index
    
        # Alles korrekt entlang der Indizes zusammenführen
        full_with_all = pd.concat(frames_to_concat, axis=1)
    
        # Tabelle anzeigen
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
    
            # Check auf doppelte Spaltennamen vor dem concat
            all_columns = pd.concat(frames_to_concat, axis=1).columns
            duplicated_cols = all_columns[all_columns.duplicated()].tolist()
    
            if duplicated_cols:
                st.error(f"❌ Duplicate column names found: {duplicated_cols}")
                # Optional: automatisch duplikate entfernen oder umbenennen
                convex_with_all = pd.concat(frames_to_concat, axis=1)
                convex_with_all = convex_with_all.loc[:, ~convex_with_all.columns.duplicated()]
                st.info("ℹ️ Duplicate columns were automatically removed.")
            else:
                convex_with_all = pd.concat(frames_to_concat, axis=1)
    
            st.markdown("✅ Final combined DataFrame")
            st.dataframe(convex_with_all, use_container_width=True)

with tab2:
    st.markdown("## Overview")

    # =====================
    # SECTION: Overview
    # =====================
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("""
        This tool consists of a structured interface with **three main tabs**:
        
        - **Decision Tool**: Core interactive analysis and filtering of read in excel file  
        - **Explanation**: This overview section explains all components  
        - **Download Results**: Export diagrams and tables as zip file
        
        On the **left side**, a collapsible **sidebar** provides configuration options for layout, plot controls, additional metrics, and convex combination settings.
        """)
    with col2:
        st.markdown("""
        <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 1000px; margin: auto;">
            <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/Bild1.png" width="100%">
            <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
        </div>
        """, unsafe_allow_html=True)
    st.divider()
    # =====================
    # SECTION: Decision Tool
    # =====================
    st.markdown("## Decision Tool – Core Functionality")
    
    with st.expander("🎚️ Slider Explanation – Adjust parameters dynamically"):
        col1, col2 = st.columns([1, 1.5])  # Breitenverhältnis: 2/3 Text, 1/3 Bild
        with col1:
            
            st.markdown("""
The filtering system allows users to narrow down the dataset by selecting variables and adjusting value ranges using sliders. To start, a user chooses a technology or metric. For the selected variable, a slider becomes available, letting the user limit its value range.

Filtering is applied sequentially, meaning that each newly added constraint is only applied to the already filtered subset. This ensures consistent and logically ordered restriction of the dataset. For example, limiting one technology first affects all subsequent selections.

Only those vertices that fulfill all active constraints remain valid. These filtered vertices are automatically reflected in the time series plots, additional metric charts, and data tables. This provides a direct visual and tabular update based on the current selection.

At any point, the user can reset all applied filters using the Reset-button. This clears all sliders and restores the original unfiltered dataset, allowing for a fresh start.
    """)
            
        with col2:
            st.markdown("""
                <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 650px; margin: auto;">
                    <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/Slider.png" width="100%">
                    <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
                </div>
                """, unsafe_allow_html=True)
    
    with st.expander("📈 Diagram Explanation – Visual output of changes"):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
        The time series plots display how the currently valid data (vertices) evolve over time for each selected variable. Blue lines represent individual valid vertices, reflecting the outcome of all active filters.
        
        The shaded blue area indicates the full remaining range (min to max) of all valid data points at each point in time. This helps users understand the spread and variability of the current solution space.
        
        Additional plot settings can be customized via the sidebar:
        - **Plot layout** (number of columns)
        - **Maximum number of vertices displayed**: limits the number of individual lines in the plot for readability
        - **Show original range** (in red): displays the full data range before filtering
        - **Show convex combinations**: overlays the convex results in red, if generated
        
        These options help to tailor the visual analysis based on user needs and enable comparison between filtered and unfiltered data distributions.
            """)
        with col2:
            st.markdown("""
                <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 650px; margin: auto;">
                    <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/Installed_cap.png" width="100%">
                    <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
                </div>
                """, unsafe_allow_html=True)
    
    with st.expander("📊 Additional Metrics – Extended insights"):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
                This section visualizes additional numerical metrics that are not part of the main time series data. Users can select which of these metrics to display from a list of available options.
                
                In the sidebar, the following display settings can be adjusted:
                - **Plot type**: choose between a **violin plot** (distribution view) or a **scatter plot** (individual points)
                - **Show convex combinations**: toggle whether convex results are included in the visualizations
                
                    """)
        with col2:
            st.markdown("""
                <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 650px; margin: auto;">
                    <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/add_metrics.png" width="100%">
                    <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
                </div>
                """, unsafe_allow_html=True)
    
    with st.expander("🧾 Tables – Detailed data views"):
        st.markdown("""
            In the tables section, the currently valid data is presented in structured form. The output is divided into two parts:
            
            - **Original vertices**: These are the filtered data points based on all active slider constraints.
            - **Convex combinations**: If generated, these show weighted combinations of the original vertices.
            
            This separation ensures transparency between original solutions and aggregated results. All table data reflects the current filtering state and can be exported for further analysis.
                """)
    st.divider()
    # ---------------------
    # Sidebar as sub-section
    # ---------------------
    st.markdown("### Sidebar Options")
    
    with st.expander(" 🧱 Layout Options – Control layout structure"):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
            The layout can be switched between **full-width** and **two-column mode**.  
            If two-column mode is selected, the **relative width of both columns** can be adjusted using a slider.  
            Additionally, users can define how many plots are displayed **side by side** using a column count setting.
            """)
        with col2:
            st.markdown("""
            <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 200px; margin: auto;">
                <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/Layout.png" width="100%">
                <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander(" 📊 Plot Options – Adjust chart types"):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
            In this section, several visualization settings can be configured:
    
            - **Number of vertices to display** in plots can be set.  
              _For better performance, it is recommended to show fewer vertices, especially with large datasets._
            - Toggle the **display of convex combinations** in the plots.
            - Enable or disable the **original (unfiltered) data range**, shown as a red shaded area for reference.
            - Toggle the **density plots**, which show the distribution of values over time using contour shading.
            """)
        with col2:
            st.markdown("""
            <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 200px; margin: auto;">
                <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/Plot.png" width="100%">
                <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander(" 📌 Additional Metrics – Toggle metrics visibility"):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
            This section visualizes additional numerical metrics that are not part of the main time series data.  
            Users can select which of these metrics to display from a list of available options.
    
            In the sidebar, the following display settings can be adjusted:
            - **Plot type**: choose between a **violin plot** (distribution view) or a **scatter plot** (individual points)
            - **Show convex combinations**: toggle whether convex results are included in the visualizations
            """)
        with col2:
            st.markdown("""
            <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 200px; margin: auto;">
                <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/add_metrics2.png" width="100%">
                <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
            </div>
            """, unsafe_allow_html=True)
    
    with st.expander(" 🧮 Convex Combination Settings – Weighted model blending"):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("""
            Here you can configure the generation of convex combinations:
    
            - Define the **total number of combinations** to be generated.
            - Set how many **vertices should be combined** per combination.
            - Adjust the **α parameter** of the Dirichlet distribution, which controls the spread of weights.
            - Specify how many combinations should be created **per batch** before a new set of vertices is sampled,  
              until the total number is reached.
            """)
        with col2:
            st.markdown("""
            <div style="border: 3px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9; width: 200px; margin: auto;">
                <img src="https://raw.githubusercontent.com/PhilippKleu/Dashboard/develope/Konvex.png" width="100%">
                <p style="text-align: center; font-style: italic; margin-top: 5px;"></p>
            </div>
            """, unsafe_allow_html=True)
    
    
with tab3:
    # === Export Everything in One ZIP (Plots + Tables) ===
    st.subheader("📦 Export All Results (Plots + Tables) as ZIP")
    
    if st.button("🗜️ Generate Full ZIP Export"):
        has_plots = "stored_figures" in st.session_state and st.session_state["stored_figures"]
        has_data = not filtered_data.empty or (
            st.session_state.get("show_convex") and not st.session_state["convex_combinations"].empty
        )
    
        if not has_plots and not has_data:
            st.info("⚠️ No plots or tables available for export.")
        else:
            zip_buffer = BytesIO()
            with ZipFile(zip_buffer, "w") as zip_file:
                # === Add plots as PDFs ===
                if has_plots:
                    for name, fig in st.session_state["stored_figures"]:
                        pdf_bytes = BytesIO()
                        fig.savefig(pdf_bytes, format="pdf", bbox_inches="tight")
                        pdf_bytes.seek(0)
                        filename = f"plots/{name.replace(' ', '_')}.pdf"
                        zip_file.writestr(filename, pdf_bytes.read())
    
                # === Add Excel with tables ===
                if has_data:
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        if not filtered_data.empty:
                            frames_to_concat = [tech_data.loc[current_indices].reset_index(drop=True)]
                            if MAA_PREFIX == "VALUE_":
                                installed_cols = [col for col in vertex_df.columns if col.startswith(INSTALLED_CAPACITY_PREFIX)]
                                installed_part = vertex_df.loc[current_indices, installed_cols].reset_index(drop=True)
                                frames_to_concat.append(installed_part)
                            if additional_cols:
                                additional_part = vertex_df.loc[current_indices, additional_cols].reset_index(drop=True)
                                frames_to_concat.append(additional_part)
                            full_original_table = pd.concat(frames_to_concat, axis=1)
                            full_original_table.to_excel(writer, index=False, sheet_name="Filtered Vertices")
    
                        if (
                            st.session_state.get("show_convex") and 
                            not st.session_state["convex_combinations"].empty
                        ):
                            convex_data = st.session_state["convex_combinations"].reset_index(drop=True)
                            frames_convex = [convex_data]
                            if additional_cols and not st.session_state.get("convex_additional", pd.DataFrame()).empty:
                                convex_add = st.session_state["convex_additional"].loc[convex_data.index, additional_cols].reset_index(drop=True)
                                frames_convex.append(convex_add)
                            full_convex_table = pd.concat(frames_convex, axis=1)
                            full_convex_table.to_excel(writer, index=False, sheet_name="Convex Combinations")
    
                    excel_buffer.seek(0)
                    zip_file.writestr("tables/filtered_results.xlsx", excel_buffer.read())
    
            zip_buffer.seek(0)
            st.download_button(
                label="⬇️ Download ZIP (Plots + Tables)",
                data=zip_buffer,
                file_name="all_results_export.zip",
                mime="application/zip"
            )
