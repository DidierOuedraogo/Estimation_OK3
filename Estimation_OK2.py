import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import zipfile
import math
import base64
from stqdm import stqdm
import seaborn as sns
from PIL import Image
from io import BytesIO
import ezdxf
import trimesh
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import datetime
import tempfile
import os
import traceback
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="MineEstim - Krigeage Ordinaire",
    page_icon="⛏️",
    layout="wide"
)

# Fonction pour capturer et afficher les erreurs
def show_detailed_error(error_title, exception):
    st.error(error_title)
    st.write("**Détails de l'erreur:**")
    st.code(traceback.format_exc())
    st.write("**Type d'erreur:** ", type(exception).__name__)
    st.write("**Message d'erreur:** ", str(exception))

# Fonction pour tenter le chargement d'un fichier CSV avec différents paramètres
def attempt_csv_loading(file, encodings=['utf-8', 'latin1', 'iso-8859-1', 'cp1252'], 
                        separators=[',', ';', '\t', '|'], 
                        decimal_points=['.', ',']):
    best_df = None
    best_params = None
    best_score = -1
    
    def evaluate_quality(df):
        if df is None or df.empty:
            return -1
        col_count = len(df.columns)
        non_na_percent = df.notna().mean().mean() * 100
        numeric_cols = sum(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
        score = col_count * 10 + non_na_percent + numeric_cols * 5
        return score
    
    for encoding in encodings:
        for sep in separators:
            for dec in decimal_points:
                try:
                    file.seek(0)
                    df = pd.read_csv(
                        file, 
                        sep=sep, 
                        decimal=dec, 
                        encoding=encoding,
                        low_memory=False,
                        on_bad_lines='warn',
                        engine='python',
                        nrows=1000
                    )
                    score = evaluate_quality(df)
                    if score > best_score:
                        file.seek(0)
                        complete_df = pd.read_csv(
                            file, 
                            sep=sep, 
                            decimal=dec, 
                            encoding=encoding,
                            low_memory=False,
                            on_bad_lines='warn'
                        )
                        best_df = complete_df
                        best_params = {'encoding': encoding, 'separator': sep, 'decimal': dec}
                        best_score = score
                except Exception:
                    continue
    
    return best_df, best_params

# Fonction pour nettoyer et préparer les données
def clean_and_prepare_data(df, col_x, col_y, col_z, col_value, col_domain=None, domain_filter=None):
    initial_rows = len(df)
    stats = {
        'Étape': ['Données initiales'],
        'Nombre de lignes': [initial_rows],
        'Lignes retirées': [0],
        'Raison': ['']
    }
    
    # Conversion des colonnes clés en numérique
    for col in [col_x, col_y, col_z, col_value]:
        if col in df.columns:
            non_numeric_before = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            non_numeric_after = df[col].isna().sum()
            rows_affected = non_numeric_after - non_numeric_before
            
            if rows_affected > 0:
                stats['Étape'].append(f'Conversion numérique {col}')
                stats['Nombre de lignes'].append(len(df))
                stats['Lignes retirées'].append(rows_affected)
                stats['Raison'].append(f'Valeurs non numériques dans {col}')
    
    # Filtrage des NA dans les colonnes essentielles
    before_na_filter = len(df)
    df = df.dropna(subset=[col_x, col_y, col_z, col_value])
    rows_affected = before_na_filter - len(df)
    
    if rows_affected > 0:
        stats['Étape'].append('Filtrage valeurs manquantes')
        stats['Nombre de lignes'].append(len(df))
        stats['Lignes retirées'].append(rows_affected)
        stats['Raison'].append('Valeurs manquantes dans X, Y, Z ou Teneur')
    
    # Filtrage par domaine si spécifié
    if col_domain and col_domain != '-- Aucun --' and domain_filter:
        domain_filter_type, domain_filter_value = domain_filter
        before_domain_filter = len(df)
        
        if domain_filter_type == "=":
            df = df[df[col_domain] == domain_filter_value]
        elif domain_filter_type == "!=":
            df = df[df[col_domain] != domain_filter_value]
        elif domain_filter_type == "IN":
            df = df[df[col_domain].isin(domain_filter_value)]
        elif domain_filter_type == "NOT IN":
            df = df[~df[col_domain].isin(domain_filter_value)]
        
        rows_affected = before_domain_filter - len(df)
        
        if rows_affected > 0:
            stats['Étape'].append(f'Filtrage domaine {domain_filter_type}')
            stats['Nombre de lignes'].append(len(df))
            stats['Lignes retirées'].append(rows_affected)
            stats['Raison'].append(f'Ne correspond pas au filtre {col_domain} {domain_filter_type} {domain_filter_value}')
    
    stats_df = pd.DataFrame(stats)
    return df, stats_df

# Fonction pour créer une représentation visuelle des étapes de nettoyage
def plot_data_cleaning_steps(stats_df):
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=stats_df['Étape'],
        y=stats_df['Nombre de lignes'],
        name='Lignes restantes',
        marker_color='rgb(55, 83, 109)'
    ))
    
    fig.add_trace(go.Bar(
        x=stats_df['Étape'],
        y=stats_df['Lignes retirées'],
        name='Lignes retirées',
        marker_color='rgb(219, 64, 82)',
        text=stats_df['Raison'],
        textposition='auto'
    ))
    
    fig.update_layout(
        title_text='Impact des étapes de nettoyage des données',
        xaxis_title='Étape de traitement',
        yaxis_title='Nombre de lignes',
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )
    
    return fig

# Fonction pour convertir un DataFrame en composites
def df_to_composites(df, col_x, col_y, col_z, col_value, col_domain=None, density_column=None):
    composites = []
    
    for idx, row in df.iterrows():
        composite = {
            'X': float(row[col_x]),
            'Y': float(row[col_y]),
            'Z': float(row[col_z]),
            'VALUE': float(row[col_value])
        }
        
        if col_domain and col_domain != '-- Aucun --' and col_domain in row:
            composite['DOMAIN'] = row[col_domain]
        
        if density_column and density_column in row and pd.notna(row[density_column]):
            try:
                density_val = float(row[density_column])
                if density_val > 0:
                    composite['DENSITY'] = density_val
            except (ValueError, TypeError):
                pass
        
        composites.append(composite)
    
    return composites

# Logo simple pour l'industrie minière
def create_mining_logo():
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    circle = plt.Circle((5, 5), 2, fill=True, color='gold')
    ax.add_patch(circle)
    ax.plot([2, 4], [3, 7], 'k-', linewidth=3)
    ax.plot([3, 6], [8, 6], 'k-', linewidth=3)
    
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    return buf

# Fonction calculate_stats pour renvoyer les statistiques des valeurs
def calculate_stats(values):
    if not values or len(values) == 0:
        return {
            'count': 0,
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            'stddev': 0,
            'variance': 0,
            'cv': 0
        }
    
    values = np.array(values)
    return {
        'count': len(values),
        'min': np.min(values),
        'max': np.max(values),
        'mean': np.mean(values),
        'median': np.median(values),
        'stddev': np.std(values),
        'variance': np.var(values),
        'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
    }

# Fonction pour générer un rapport PDF d'estimation
def generate_estimation_report(estimated_blocks, composites_data, kriging_params, search_params, block_sizes, 
                              variogram_model=None, tonnage_data=None, plot_info=None, density_method="constant", 
                              density_value=2.7, density_column=None, project_name="Projet Minier"):
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading1_style = styles['Heading1']
        heading2_style = styles['Heading2']
        normal_style = styles['Normal']
        
        elements = []
        
        # Titre et date
        elements.append(Paragraph(f"Rapport d'Estimation - {project_name}", title_style))
        elements.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%d/%m/%Y')}", normal_style))
        elements.append(Paragraph(f"Auteur: Didier Ouedraogo, P.Geo", normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Méthodologie
        elements.append(Paragraph("1. Méthodologie d'Estimation", heading1_style))
        elements.append(Paragraph("Cette estimation a été réalisée par la méthode du krigeage ordinaire, "
                                 "qui est une méthode géostatistique optimale d'interpolation spatiale prenant en compte "
                                 "la structure spatiale des variables régionalisées.", normal_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Variogramme
        elements.append(Paragraph("1.1 Modèle variographique", heading2_style))
        
        if variogram_model:
            variogram_text = (f"Le modèle de variogramme utilisé est de type {variogram_model['type']} "
                             f"avec un effet pépite de {variogram_model['nugget']:.3f}, "
                             f"un palier de {variogram_model['sill']:.3f} et une portée de {variogram_model['range']:.1f} mètres.")
        else:
            variogram_text = ("Aucun modèle de variogramme spécifié. Les valeurs par défaut ont été utilisées pour l'estimation.")
        
        elements.append(Paragraph(variogram_text, normal_style))
        elements.append(Spacer(1, 0.1*inch))
        
        # Paramètres
        elements.append(Paragraph("2. Paramètres d'Estimation", heading1_style))
        
        data = [
            ["Paramètre", "Valeur"],
            ["Méthode d'estimation", "Krigeage ordinaire"],
            ["Type de variogramme", variogram_model['type'] if variogram_model else "-"],
            ["Effet pépite", str(variogram_model['nugget']) if variogram_model else "-"],
            ["Palier", str(variogram_model['sill']) if variogram_model else "-"],
            ["Portée", str(variogram_model['range']) + " m" if variogram_model else "-"],
            ["Anisotropie X", str(kriging_params['anisotropy']['x'])],
            ["Anisotropie Y", str(kriging_params['anisotropy']['y'])],
            ["Anisotropie Z", str(kriging_params['anisotropy']['z'])],
            ["Rayon de recherche X", str(search_params['x']) + " m"],
            ["Rayon de recherche Y", str(search_params['y']) + " m"],
            ["Rayon de recherche Z", str(search_params['z']) + " m"],
            ["Min. échantillons", str(search_params['min_samples'])],
            ["Max. échantillons", str(search_params['max_samples'])],
            ["Taille des blocs", f"{block_sizes['x']} × {block_sizes['y']} × {block_sizes['z']} m"]
        ]
        
        if density_method == "constant":
            data.append(["Densité", f"{density_value} t/m³ (constante)"])
        else:
            data.append(["Densité", f"Variable (colonne {density_column})"])
        
        t = Table(data, colWidths=[2.5*inch, 2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.2*inch))
        
        # Statistiques
        elements.append(Paragraph("3. Statistiques", heading1_style))
        
        # Statistiques des composites
        elements.append(Paragraph("3.1 Statistiques des Composites", heading2_style))
        composite_values = [comp['VALUE'] for comp in composites_data if 'VALUE' in comp]
        composite_stats = calculate_stats(composite_values)
        
        comp_data = [
            ["Statistique", "Valeur"],
            ["Nombre d'échantillons", str(composite_stats['count'])],
            ["Minimum", f"{composite_stats['min']:.3f}"],
            ["Maximum", f"{composite_stats['max']:.3f}"],
            ["Moyenne", f"{composite_stats['mean']:.3f}"],
            ["Médiane", f"{composite_stats['median']:.3f}"],
            ["Écart-type", f"{composite_stats['stddev']:.3f}"],
            ["CV", f"{composite_stats['cv']:.3f}"]
        ]
        
        comp_table = Table(comp_data, colWidths=[2.5*inch, 2*inch])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(comp_table)
        
        # Histogramme des composites si assez de données
        if composite_stats['count'] > 1:
            fig_comp, ax_comp = plt.subplots(figsize=(6, 4))
            n_bins = max(5, int(1 + 3.322 * math.log10(len(composite_values))))
            sns.histplot(composite_values, bins=n_bins, kde=True, color="darkblue", ax=ax_comp)
            ax_comp.set_title("Distribution des teneurs des composites")
            ax_comp.set_xlabel("Teneur")
            ax_comp.set_ylabel("Fréquence")
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                fig_comp.savefig(tmp_file.name, format='png', dpi=150, bbox_inches='tight')
                comp_hist_path = tmp_file.name
            
            elements.append(Spacer(1, 0.1*inch))
            comp_img = ReportLabImage(comp_hist_path, width=4*inch, height=3*inch)
            elements.append(comp_img)
            elements.append(Spacer(1, 0.2*inch))
        
        # Statistiques des blocs
        if estimated_blocks and len(estimated_blocks) > 0:
            elements.append(Paragraph("3.2 Statistiques du Modèle de Blocs", heading2_style))
            block_values = [block.get('value', 0) for block in estimated_blocks]
            block_stats = calculate_stats(block_values)
            
            block_data = [
                ["Statistique", "Valeur"],
                ["Nombre de blocs", str(block_stats['count'])],
                ["Minimum", f"{block_stats['min']:.3f}"],
                ["Maximum", f"{block_stats['max']:.3f}"],
                ["Moyenne", f"{block_stats['mean']:.3f}"],
                ["Médiane", f"{block_stats['median']:.3f}"],
                ["Écart-type", f"{block_stats['stddev']:.3f}"],
                ["CV", f"{block_stats['cv']:.3f}"]
            ]
            
            block_table = Table(block_data, colWidths=[2.5*inch, 2*inch])
            block_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(block_table)
            
            # Erreur d'estimation
            elements.append(Paragraph("3.3 Variance d'estimation", heading2_style))
            
            block_variances = [block.get('estimation_variance', 0) for block in estimated_blocks if 'estimation_variance' in block]
            
            if block_variances:
                var_stats = calculate_stats(block_variances)
                
                var_data = [
                    ["Statistique", "Valeur"],
                    ["Minimum", f"{var_stats['min']:.4f}"],
                    ["Maximum", f"{var_stats['max']:.4f}"],
                    ["Moyenne", f"{var_stats['mean']:.4f}"],
                    ["Médiane", f"{var_stats['median']:.4f}"]
                ]
                
                var_table = Table(var_data, colWidths=[2.5*inch, 2*inch])
                var_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(var_table)
                
                elements.append(Spacer(1, 0.1*inch))
                elements.append(Paragraph("La variance d'estimation est une mesure de l'incertitude associée à l'estimation par krigeage. "
                                          "Une variance faible indique une meilleure confiance dans l'estimation.", normal_style))
            else:
                elements.append(Paragraph("Aucune information sur la variance d'estimation n'est disponible.", normal_style))
            
            # Résumé global
            elements.append(Paragraph("3.4 Résumé Global", heading2_style))
            
            if density_method == "constant":
                avg_density = density_value
            else:
                block_volumes = [block.get('size_x', 0) * block.get('size_y', 0) * block.get('size_z', 0) for block in estimated_blocks]
                avg_density = sum(block.get('density', density_value) * vol for block, vol in zip(estimated_blocks, block_volumes)) / (sum(block_volumes) or 1)
            
            if 'size_x' in estimated_blocks[0] and 'size_y' in estimated_blocks[0] and 'size_z' in estimated_blocks[0]:
                block_volume = estimated_blocks[0]['size_x'] * estimated_blocks[0]['size_y'] * estimated_blocks[0]['size_z']
                total_volume = len(estimated_blocks) * block_volume
                total_tonnage = total_volume * avg_density
            else:
                block_volume = 0
                total_volume = 0
                total_tonnage = 0
            
            summary_data = [
                ["Paramètre", "Valeur"],
                ["Nombre de blocs", f"{len(estimated_blocks):,}"],
                ["Teneur moyenne", f"{block_stats['mean']:.3f}"],
                ["Densité moyenne", f"{avg_density:.2f} t/m³"],
                ["Volume total (m³)", f"{total_volume:,.0f}"],
                ["Tonnage total (t)", f"{total_tonnage:,.0f}"]
            ]
            
            summary_table = Table(summary_data, colWidths=[2.5*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(summary_table)
            elements.append(Spacer(1, 0.2*inch))
        
        # Analyse Tonnage-Teneur
        if tonnage_data and plot_info and 'cutoffs' in tonnage_data and 'tonnages' in tonnage_data and 'grades' in tonnage_data:
            elements.append(Paragraph("4. Analyse Tonnage-Teneur", heading1_style))
            
            if plot_info.get('method') != 'between':
                fig_tonnage, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
                
                ax1.plot(tonnage_data['cutoffs'], tonnage_data['tonnages'], 'b-', linewidth=2)
                ax1.set_xlabel('Teneur de coupure')
                ax1.set_ylabel('Tonnage (t)')
                ax1.set_title('Courbe Tonnage-Teneur')
                ax1.grid(True)
                
                ax2.plot(tonnage_data['cutoffs'], tonnage_data['grades'], 'g-', linewidth=2)
                ax2.set_xlabel('Teneur de coupure')
                ax2.set_ylabel('Teneur moyenne')
                ax2.set_title('Teneur moyenne en fonction de la coupure')
                ax2.grid(True)
                
                plt.tight_layout()
                
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    fig_tonnage.savefig(tmp_file.name, format='png', dpi=150, bbox_inches='tight')
                    tonnage_graph_path = tmp_file.name
                
                tonnage_img = ReportLabImage(tonnage_graph_path, width=5*inch, height=6*inch)
                elements.append(tonnage_img)
                elements.append(Spacer(1, 0.2*inch))
                
                elements.append(Paragraph("4.1 Résultats détaillés", heading2_style))
                
                cutoffs_subset = tonnage_data['cutoffs'][::3]
                tonnages_subset = tonnage_data['tonnages'][::3]
                grades_subset = tonnage_data['grades'][::3]
                metals_subset = tonnage_data['metals'][::3] if 'metals' in tonnage_data else [0] * len(cutoffs_subset)
                
                tonnage_table_data = [["Coupure", "Tonnage (t)", "Teneur moyenne", "Métal contenu"]]
                for i in range(len(cutoffs_subset)):
                    tonnage_table_data.append([
                        cutoffs_subset[i],
                        f"{tonnages_subset[i]:,.0f}",
                        f"{grades_subset[i]:.3f}",
                        f"{metals_subset[i]:,.0f}"
                    ])
                
                tonnage_table = Table(tonnage_table_data, colWidths=[1*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                tonnage_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(tonnage_table)
            else:
                min_grade = plot_info.get('min_grade', 0)
                max_grade = plot_info.get('max_grade', 0)
                elements.append(Paragraph(f"Méthode de coupure: Entre {min_grade:.2f} et {max_grade:.2f}", normal_style))
                
                between_data = [
                    ["Paramètre", "Valeur"],
                    ["Tonnage (t)", f"{tonnage_data['tonnages'][0]:,.0f}" if len(tonnage_data['tonnages']) > 0 else "0"],
                    ["Teneur moyenne", f"{tonnage_data['grades'][0]:.3f}" if len(tonnage_data['grades']) > 0 else "0"],
                    ["Métal contenu", f"{tonnage_data['metals'][0]:,.0f}" if 'metals' in tonnage_data and len(tonnage_data['metals']) > 0 else "0"]
                ]
                
                between_table = Table(between_data, colWidths=[2.5*inch, 2*inch])
                between_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(between_table)
        
        # Conclusion
        elements.append(Paragraph("5. Conclusion", heading1_style))
        elements.append(Paragraph("Ce rapport présente les résultats d'une estimation de ressources minérales "
                                 "par la méthode du krigeage ordinaire. Cette méthode géostatistique fournit "
                                 "non seulement une estimation des teneurs mais aussi une évaluation de "
                                 "l'incertitude associée à l'estimation.", normal_style))
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph("Le krigeage est une méthode d'interpolation optimale (BLUE - Best Linear Unbiased Estimator) "
                                 "qui prend en compte la structure spatiale de la minéralisation à travers "
                                 "le variogramme, ce qui permet de mieux modéliser la continuité du gisement.", normal_style))
        
        doc.build(elements)
        
        temp_files = [var for var in locals() if var.endswith('_path')]
        for file_path_var in temp_files:
            if os.path.exists(locals()[file_path_var]):
                os.unlink(locals()[file_path_var])
        
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Erreur lors de la génération du rapport PDF: {str(e)}")
        return None

# Fonction pour traiter le fichier DXF
def process_dxf_file(dxf_file):
    try:
        dxf_content = dxf_file.read()
        file_buffer = io.BytesIO(dxf_content)
        
        doc = ezdxf.readfile(file_buffer)
        msp = doc.modelspace()
        
        entities = []
        
        for entity in msp:
            if entity.dxftype() == 'POLYLINE' or entity.dxftype() == 'LWPOLYLINE':
                if hasattr(entity, 'closed') and entity.closed:
                    vertices = []
                    for vertex in entity.vertices():
                        vertices.append((vertex.dxf.location.x, vertex.dxf.location.y, vertex.dxf.location.z))
                    if len(vertices) >= 3:
                        entities.append({
                            'type': 'polyline',
                            'vertices': vertices
                        })
            elif entity.dxftype() == '3DFACE':
                vertices = [
                    (entity.dxf.vtx0.x, entity.dxf.vtx0.y, entity.dxf.vtx0.z),
                    (entity.dxf.vtx1.x, entity.dxf.vtx1.y, entity.dxf.vtx1.z),
                    (entity.dxf.vtx2.x, entity.dxf.vtx2.y, entity.dxf.vtx2.z),
                    (entity.dxf.vtx3.x, entity.dxf.vtx3.y, entity.dxf.vtx3.z)
                ]
                entities.append({
                    'type': '3dface',
                    'vertices': vertices
                })
            elif entity.dxftype() == 'MESH':
                vertices = []
                for vertex in entity.vertices():
                    vertices.append((vertex.x, vertex.y, vertex.z))
                if len(vertices) >= 3:
                    entities.append({
                        'type': 'mesh',
                        'vertices': vertices
                    })
        
        # Créer un maillage 3D à partir des entités
        mesh_vertices = []
        mesh_faces = []
        for entity in entities:
            vertex_offset = len(mesh_vertices)
            mesh_vertices.extend(entity['vertices'])
            
            if entity['type'] == 'polyline':
                for i in range(1, len(entity['vertices']) - 1):
                    mesh_faces.append([vertex_offset, vertex_offset + i, vertex_offset + i + 1])
            elif entity['type'] == '3dface':
                mesh_faces.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
                mesh_faces.append([vertex_offset, vertex_offset + 2, vertex_offset + 3])
            elif entity['type'] == 'mesh':
                for i in range(0, len(entity['vertices']), 3):
                    if i + 2 < len(entity['vertices']):
                        mesh_faces.append([vertex_offset + i, vertex_offset + i + 1, vertex_offset + i + 2])
        
        if mesh_vertices and mesh_faces:
            mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
            
            bounds = mesh.bounds
            min_bounds = {
                'x': bounds[0][0],
                'y': bounds[0][1],
                'z': bounds[0][2]
            }
            max_bounds = {
                'x': bounds[1][0],
                'y': bounds[1][1],
                'z': bounds[1][2]
            }
            
            return {
                'mesh': mesh,
                'bounds': {
                    'min': min_bounds,
                    'max': max_bounds
                },
                'vertices': mesh_vertices,
                'faces': mesh_faces
            }
        else:
            return None
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier DXF: {str(e)}")
        return None

# Fonctions pour le variogramme et le krigeage
def euclidean_distance(p1, p2, anisotropy=None):
    if anisotropy is None:
        anisotropy = {'x': 1, 'y': 1, 'z': 1}
    
    dx = (p2['x'] - p1['x']) / anisotropy['x']
    dy = (p2['y'] - p1['y']) / anisotropy['y']
    dz = (p2['z'] - p1['z']) / anisotropy['z']
    
    return math.sqrt(dx**2 + dy**2 + dz**2)

def spherical_variogram(h, c0, c, a):
    if h == 0:
        return 0
    elif h >= a:
        return c0 + c
    else:
        return c0 + c * (1.5 * (h/a) - 0.5 * (h/a)**3)

def exponential_variogram(h, c0, c, a):
    if h == 0:
        return 0
    else:
        return c0 + c * (1 - math.exp(-3 * h / a))

def gaussian_variogram(h, c0, c, a):
    if h == 0:
        return 0
    else:
        return c0 + c * (1 - math.exp(-3 * h**2 / a**2))

def ordinary_kriging(point, samples, variogram_model, anisotropy=None):
    n = len(samples)
    if n == 0:
        return 0, 1.0  # Valeur par défaut, variance maximale
    
    # Si un échantillon est exactement au point à estimer, retourner sa valeur
    for sample in samples:
        if sample['x'] == point['x'] and sample['y'] == point['y'] and sample['z'] == point['z']:
            return sample['value'], 0.0  # Variance nulle pour un point exact
    
    # Matrice des covariances entre échantillons
    K = np.zeros((n+1, n+1))
    
    # Remplir la matrice K
    for i in range(n):
        for j in range(n):
            # Calculer la distance entre les échantillons i et j
            h = euclidean_distance({'x': samples[i]['x'], 'y': samples[i]['y'], 'z': samples[i]['z']}, 
                                   {'x': samples[j]['x'], 'y': samples[j]['y'], 'z': samples[j]['z']}, 
                                   anisotropy)
            
            # Calculer la valeur du variogramme pour cette distance
            if variogram_model['type'] == 'spherical':
                gamma = spherical_variogram(h, variogram_model['nugget'], variogram_model['sill'], variogram_model['range'])
            elif variogram_model['type'] == 'exponential':
                gamma = exponential_variogram(h, variogram_model['nugget'], variogram_model['sill'], variogram_model['range'])
            elif variogram_model['type'] == 'gaussian':
                gamma = gaussian_variogram(h, variogram_model['nugget'], variogram_model['sill'], variogram_model['range'])
            
            # La covariance est le palier total moins le variogramme
            K[i, j] = variogram_model['total_sill'] - gamma
    
    # Ajouter les contraintes de la somme des poids = 1 (krigeage ordinaire)
    for i in range(n):
        K[i, n] = 1.0
        K[n, i] = 1.0
    K[n, n] = 0.0
    
    # Vecteur des covariances entre les échantillons et le point à estimer
    k = np.zeros(n+1)
    
    for i in range(n):
        # Calculer la distance entre l'échantillon i et le point à estimer
        h = euclidean_distance({'x': samples[i]['x'], 'y': samples[i]['y'], 'z': samples[i]['z']}, 
                               point, anisotropy)
        
        # Calculer la valeur du variogramme pour cette distance
        if variogram_model['type'] == 'spherical':
            gamma = spherical_variogram(h, variogram_model['nugget'], variogram_model['sill'], variogram_model['range'])
        elif variogram_model['type'] == 'exponential':
            gamma = exponential_variogram(h, variogram_model['nugget'], variogram_model['sill'], variogram_model['range'])
        elif variogram_model['type'] == 'gaussian':
            gamma = gaussian_variogram(h, variogram_model['nugget'], variogram_model['sill'], variogram_model['range'])
        
        # La covariance est le palier total moins le variogramme
        k[i] = variogram_model['total_sill'] - gamma
    
    # La dernière composante est 1 pour satisfaire la contrainte du krigeage ordinaire
    k[n] = 1.0
    
    # Résoudre le système pour trouver les poids
    try:
        weights = np.linalg.solve(K, k)
    except np.linalg.LinAlgError:
        # En cas de matrice singulière, ajouter un petit epsilon à la diagonale
        for i in range(n):
            K[i, i] += 1e-6
        try:
            weights = np.linalg.solve(K, k)
        except np.linalg.LinAlgError:
            # Si toujours singulière, utiliser une pseudo-inverse
            weights = np.linalg.lstsq(K, k, rcond=None)[0]
    
    # Extraire les poids du krigeage (sans le multiplicateur de Lagrange)
    lambda_weights = weights[:n]
    
    # Calculer l'estimation par krigeage
    estimate = sum(lambda_weights[i] * samples[i]['value'] for i in range(n))
    
    # Calculer la variance d'estimation (variance du krigeage)
    # La variance est la covariance au point moins la somme des produits poids * covariance
    kriging_variance = variogram_model['total_sill'] - np.dot(lambda_weights, k[:n]) - weights[n]
    
    return estimate, kriging_variance

def is_point_inside_mesh(point, mesh):
    if mesh is None:
        return True
    
    try:
        point_array = np.array([point['x'], point['y'], point['z']])
        return bool(mesh.contains([point_array])[0])
    except Exception as e:
        st.warning(f"Erreur lors de la vérification de l'enveloppe: {str(e)}")
        return True  # Par défaut, inclure le point en cas d'erreur

def create_block_model(composites, block_sizes, envelope_data=None, use_envelope=True):
    # Vérifier si les composites existent
    if not composites or len(composites) == 0:
        st.error("Aucun échantillon valide pour créer le modèle de blocs.")
        return [], {'min': {'x': 0, 'y': 0, 'z': 0}, 'max': {'x': 0, 'y': 0, 'z': 0}}
    
    # Déterminer les limites du modèle
    if use_envelope and envelope_data:
        min_bounds = envelope_data['bounds']['min']
        max_bounds = envelope_data['bounds']['max']
    else:
        x_values = [comp['X'] for comp in composites if 'X' in comp]
        y_values = [comp['Y'] for comp in composites if 'Y' in comp]
        z_values = [comp['Z'] for comp in composites if 'Z' in comp]
        
        if not x_values or not y_values or not z_values:
            st.error("Données insuffisantes pour déterminer les limites du modèle.")
            return [], {'min': {'x': 0, 'y': 0, 'z': 0}, 'max': {'x': 0, 'y': 0, 'z': 0}}
        
        min_bounds = {
            'x': math.floor(min(x_values) / block_sizes['x']) * block_sizes['x'],
            'y': math.floor(min(y_values) / block_sizes['y']) * block_sizes['y'],
            'z': math.floor(min(z_values) / block_sizes['z']) * block_sizes['z']
        }
        
        max_bounds = {
            'x': math.ceil(max(x_values) / block_sizes['x']) * block_sizes['x'],
            'y': math.ceil(max(y_values) / block_sizes['y']) * block_sizes['y'],
            'z': math.ceil(max(z_values) / block_sizes['z']) * block_sizes['z']
        }
    
    # Créer les blocs
    blocks = []
    
    x_range = np.arange(min_bounds['x'] + block_sizes['x']/2, max_bounds['x'] + block_sizes['x']/2, block_sizes['x'])
    y_range = np.arange(min_bounds['y'] + block_sizes['y']/2, max_bounds['y'] + block_sizes['y']/2, block_sizes['y'])
    z_range = np.arange(min_bounds['z'] + block_sizes['z']/2, max_bounds['z'] + block_sizes['z']/2, block_sizes['z'])
    
    mesh = envelope_data['mesh'] if envelope_data and use_envelope else None
    
    with st.spinner('Création du modèle de blocs...'):
        progress_bar = st.progress(0)
        total_blocks = len(x_range) * len(y_range) * len(z_range)
        block_count = 0
        
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    block = {
                        'x': x,
                        'y': y,
                        'z': z,
                        'size_x': block_sizes['x'],
                        'size_y': block_sizes['y'],
                        'size_z': block_sizes['z']
                    }
                    
                    # Vérifier si le bloc est dans l'enveloppe
                    if not use_envelope or is_point_inside_mesh(block, mesh):
                        blocks.append(block)
                    
                    block_count += 1
                    if block_count % 100 == 0 or block_count == total_blocks:
                        progress_bar.progress(min(block_count / total_blocks, 1.0))
        
        progress_bar.progress(1.0)
    
    return blocks, {'min': min_bounds, 'max': max_bounds}

def estimate_block_model_kriging(empty_blocks, composites, kriging_params, search_params, variogram_model, density_method="constant", density_value=2.7):
    estimated_blocks = []
    
    # Vérifier si les entrées sont valides
    if not empty_blocks or len(empty_blocks) == 0:
        st.error("Aucun bloc à estimer.")
        return []
    
    if not composites or len(composites) == 0:
        st.error("Aucun échantillon disponible pour l'estimation.")
        return []
    
    # S'assurer que le modèle de variogramme a la bonne structure
    if not variogram_model:
        st.error("Aucun modèle de variogramme spécifié.")
        return []
    
    if 'total_sill' not in variogram_model:
        variogram_model['total_sill'] = variogram_model['nugget'] + variogram_model['sill']
    
    with st.spinner('Estimation par krigeage ordinaire...'):
        progress_bar = st.progress(0)
        
        for idx, block in enumerate(stqdm(empty_blocks)):
            progress = idx / len(empty_blocks)
            progress_bar.progress(progress)
            
            # Chercher les échantillons pour le krigeage
            samples = []
            density_samples = []
            
            for composite in composites:
                if 'X' not in composite or 'Y' not in composite or 'Z' not in composite or 'VALUE' not in composite:
                    continue
                
                # Appliquer l'anisotropie
                dx = (composite['X'] - block['x']) / kriging_params['anisotropy']['x']
                dy = (composite['Y'] - block['y']) / kriging_params['anisotropy']['y']
                dz = (composite['Z'] - block['z']) / kriging_params['anisotropy']['z']
                
                distance = math.sqrt(dx**2 + dy**2 + dz**2)
                
                if distance <= max(search_params['x'], search_params['y'], search_params['z']):
                    samples.append({
                        'x': composite['X'],
                        'y': composite['Y'],
                        'z': composite['Z'],
                        'value': composite['VALUE'],
                        'distance': distance
                    })
                    
                    # Si la densité est variable, ajouter les échantillons de densité
                    if density_method == "variable" and 'DENSITY' in composite:
                        density_samples.append({
                            'x': composite['X'],
                            'y': composite['Y'],
                            'z': composite['Z'],
                            'value': composite['DENSITY'],
                            'distance': distance
                        })
            
            samples.sort(key=lambda x: x['distance'])
            
            if len(samples) >= search_params['min_samples']:
                used_samples = samples[:min(len(samples), search_params['max_samples'])]
                
                # Estimation par krigeage
                estimate, variance = ordinary_kriging(
                    block, 
                    used_samples, 
                    variogram_model,
                    kriging_params['anisotropy']
                )
                
                estimated_block = block.copy()
                estimated_block['value'] = estimate
                estimated_block['estimation_variance'] = variance
                
                # Estimer la densité si nécessaire
                if density_method == "variable" and density_samples:
                    density_samples.sort(key=lambda x: x['distance'])
                    used_density_samples = density_samples[:min(len(density_samples), search_params['max_samples'])]
                    
                    density_estimate, _ = ordinary_kriging(
                        block, 
                        used_density_samples, 
                        variogram_model,
                        kriging_params['anisotropy']
                    )
                    
                    estimated_block['density'] = density_estimate
                else:
                    estimated_block['density'] = density_value
                
                estimated_blocks.append(estimated_block)
        
        progress_bar.progress(1.0)
    
    return estimated_blocks

def calculate_tonnage_grade(blocks, density_method="constant", density_value=2.7, method="above", cutoff_value=None, cutoff_min=None, cutoff_max=None):
    if not blocks:
        return {
            'cutoffs': [],
            'tonnages': [],
            'grades': [],
            'metals': []
        }, {
            'method': method,
            'min_grade': 0,
            'max_grade': 0
        }
    
    # Extraire les valeurs
    values = [block.get('value', 0) for block in blocks]
    
    if not values:
        return {
            'cutoffs': [],
            'tonnages': [],
            'grades': [],
            'metals': []
        }, {
            'method': method,
            'min_grade': 0,
            'max_grade': 0
        }
    
    min_grade = min(values)
    max_grade = max(values)
    
    # Générer les coupures
    step = (max_grade - min_grade) / 20 if max_grade > min_grade else 0.1
    cutoffs = np.arange(min_grade, max_grade + step, max(step, 0.0001))
    
    tonnages = []
    grades = []
    metals = []
    cutoff_labels = []
    
    for cutoff in cutoffs:
        cutoff_labels.append(f"{cutoff:.2f}")
        
        if method == 'above':
            filtered_blocks = [block for block in blocks if block.get('value', 0) >= cutoff]
        elif method == 'below':
            filtered_blocks = [block for block in blocks if block.get('value', 0) <= cutoff]
        elif method == 'between':
            filtered_blocks = [block for block in blocks if cutoff_min <= block.get('value', 0) <= cutoff_max]
            
            # Pour la méthode between, on n'a besoin que d'un seul résultat
            if cutoff > min_grade:
                continue
        
        if not filtered_blocks:
            tonnages.append(0)
            grades.append(0)
            metals.append(0)
            continue
        
        total_tonnage = 0
        total_metal = 0
        
        for block in filtered_blocks:
            if 'size_x' in block and 'size_y' in block and 'size_z' in block:
                block_volume = block['size_x'] * block['size_y'] * block['size_z']
                block_density = block.get('density', density_value) if density_method == "variable" else density_value
                block_tonnage = block_volume * block_density
                
                total_tonnage += block_tonnage
                total_metal += block_tonnage * block.get('value', 0)
        
        if total_tonnage > 0:
            avg_grade = total_metal / total_tonnage
        else:
            avg_grade = 0
        
        tonnages.append(total_tonnage)
        grades.append(avg_grade)
        metals.append(total_metal)
    
    return {
        'cutoffs': cutoff_labels,
        'tonnages': tonnages,
        'grades': grades,
        'metals': metals
    }, {
        'method': method,
        'min_grade': min_grade,
        'max_grade': max_grade
    }

# Fonctions de visualisation
def plot_3d_model_with_cubes(blocks, composites, envelope_data=None, block_scale=0.9, color_by='value'):
    fig = go.Figure()
    
    # Ajouter les composites
    if composites:
        x = [comp.get('X', 0) for comp in composites if 'X' in comp and 'Y' in comp and 'Z' in comp and 'VALUE' in comp]
        y = [comp.get('Y', 0) for comp in composites if 'X' in comp and 'Y' in comp and 'Z' in comp and 'VALUE' in comp]
        z = [comp.get('Z', 0) for comp in composites if 'X' in comp and 'Y' in comp and 'Z' in comp and 'VALUE' in comp]
        values = [comp.get('VALUE', 0) for comp in composites if 'X' in comp and 'Y' in comp and 'Z' in comp and 'VALUE' in comp]
        
        if x and y and z and values:
            composite_scatter = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=3,
                    color=values,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Teneur")
                ),
                text=[f"Teneur: {v:.3f}" for v in values],
                name='Composites'
            )
            fig.add_trace(composite_scatter)
    
    # Ajouter les blocs en tant que cubes
    if blocks:
        # Vérifier les clés nécessaires dans les blocs
        valid_blocks = [block for block in blocks 
                      if 'x' in block and 'y' in block and 'z' in block 
                      and 'size_x' in block and 'size_y' in block and 'size_z' in block 
                      and color_by in block]
        
        if not valid_blocks:
            st.warning("Aucun bloc valide à afficher.")
            return fig
        
        # Limiter le nombre de blocs pour éviter de surcharger la visualisation
        max_display_blocks = 2000
        if len(valid_blocks) > max_display_blocks:
            st.warning(f"Le modèle contient {len(valid_blocks)} blocs. Pour une meilleure performance, seuls {max_display_blocks} blocs sont affichés.")
            valid_blocks = valid_blocks[:max_display_blocks]
        
        # Créer des cubes pour chaque bloc (en utilisant Mesh3d)
        x_vals = []
        y_vals = []
        z_vals = []
        i_vals = []
        j_vals = []
        k_vals = []
        intensity = []
        
        for idx, block in enumerate(valid_blocks):
            # Créer les 8 sommets d'un cube
            x_size = block['size_x'] * block_scale / 2
            y_size = block['size_y'] * block_scale / 2
            z_size = block['size_z'] * block_scale / 2
            
            x0, y0, z0 = block['x'] - x_size, block['y'] - y_size, block['z'] - z_size
            x1, y1, z1 = block['x'] + x_size, block['y'] + y_size, block['z'] + z_size
            
            # Ajouter les sommets
            vertices = [
                (x0, y0, z0),  # 0
                (x1, y0, z0),  # 1
                (x1, y1, z0),  # 2
                (x0, y1, z0),  # 3
                (x0, y0, z1),  # 4
                (x1, y0, z1),  # 5
                (x1, y1, z1),  # 6
                (x0, y1, z1)   # 7
            ]
            
            # Ajouter les faces du cube (triangles)
            faces = [
                (0, 1, 2), (0, 2, 3),  # bottom
                (4, 5, 6), (4, 6, 7),  # top
                (0, 1, 5), (0, 5, 4),  # front
                (2, 3, 7), (2, 7, 6),  # back
                (0, 3, 7), (0, 7, 4),  # left
                (1, 2, 6), (1, 6, 5)   # right
            ]
            
            for v in vertices:
                x_vals.append(v[0])
                y_vals.append(v[1])
                z_vals.append(v[2])
                intensity.append(block[color_by])
            
            offset = idx * 8  # 8 sommets par cube
            for f in faces:
                i_vals.append(offset + f[0])
                j_vals.append(offset + f[1])
                k_vals.append(offset + f[2])
        
        if x_vals and y_vals and z_vals and i_vals and j_vals and k_vals and intensity:
            # Utiliser une échelle de couleur appropriée au type de valeur
            if color_by == 'value':
                colorscale = 'Viridis'
                colorbar_title = "Teneur"
                block_name = 'Blocs estimés'
            elif color_by == 'estimation_variance':
                colorscale = 'Reds'
                colorbar_title = "Variance"
                block_name = 'Variance d\'estimation'
            else:
                colorscale = 'Viridis'
                colorbar_title = color_by
                block_name = 'Blocs'
            
            block_mesh = go.Mesh3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                i=i_vals,
                j=j_vals,
                k=k_vals,
                intensity=intensity,
                colorscale=colorscale,
                opacity=0.7,
                name=block_name,
                colorbar=dict(title=colorbar_title)
            )
            fig.add_trace(block_mesh)
    
    # Ajouter l'enveloppe DXF
    if envelope_data and 'vertices' in envelope_data and 'faces' in envelope_data:
        vertices = envelope_data['vertices']
        faces = envelope_data['faces']
        
        if vertices and faces:
            i, j, k = [], [], []
            for face in faces:
                if len(face) >= 3:
                    i.append(face[0])
                    j.append(face[1])
                    k.append(face[2])
            
            if i and j and k:
                wireframe = go.Mesh3d(
                    x=[v[0] for v in vertices],
                    y=[v[1] for v in vertices],
                    z=[v[2] for v in vertices],
                    i=i, j=j, k=k,
                    opacity=0.3,
                    color='green',
                    name='Enveloppe'
                )
                fig.add_trace(wireframe)
    
    # Mise en page
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(x=0, y=0.9)
    )
    
    return fig

def plot_histogram(values, title, color='steelblue'):
    if not values or len(values) <= 1:
        # Créer un graphique vide avec un message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Données insuffisantes pour l'histogramme", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculer le nombre de bins (au moins 5)
    n_bins = max(5, int(1 + 3.322 * math.log10(len(values))))
    
    sns.histplot(values, bins=n_bins, kde=True, color=color, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Valeur')
    ax.set_ylabel('Fréquence')
    
    return fig

def plot_tonnage_grade(tonnage_data, plot_info=None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Vérifier si les données sont valides
    if (not tonnage_data or 'cutoffs' not in tonnage_data or 'tonnages' not in tonnage_data or 'grades' not in tonnage_data or
        len(tonnage_data['cutoffs']) == 0 or len(tonnage_data['tonnages']) == 0 or len(tonnage_data['grades']) == 0):
        # Ajouter un texte indiquant l'absence de données
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Données insuffisantes pour le graphique tonnage-teneur",
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    if plot_info and plot_info.get('method') == 'between':
        # Pour la méthode 'between', on utilise un graphique à barres
        fig.add_trace(
            go.Bar(
                x=['Résultat'],
                y=[tonnage_data['tonnages'][0]],
                name='Tonnage',
                marker_color='rgb(63, 81, 181)'
            )
        )
        
        fig.add_trace(
            go.Bar(
                x=['Résultat'],
                y=[tonnage_data['grades'][0]],
                name='Teneur moyenne',
                marker_color='rgb(0, 188, 212)'
            ),
            secondary_y=True
        )
    else:
        # Pour les méthodes 'above' et 'below', on utilise un graphique en ligne
        fig.add_trace(
            go.Scatter(
                x=tonnage_data['cutoffs'],
                y=tonnage_data['tonnages'],
                name='Tonnage',
                fill='tozeroy',
                mode='lines',
                line=dict(color='rgb(63, 81, 181)')
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=tonnage_data['cutoffs'],
                y=tonnage_data['grades'],
                name='Teneur moyenne',
                mode='lines',
                line=dict(color='rgb(0, 188, 212)')
            ),
            secondary_y=True
        )
    
    fig.update_layout(
        title_text='Courbe Tonnage-Teneur',
        xaxis_title='Teneur de coupure',
        legend=dict(x=0, y=1.1, orientation='h')
    )
    
    fig.update_yaxes(title_text='Tonnage (t)', secondary_y=False)
    fig.update_yaxes(title_text='Teneur moyenne', secondary_y=True)
    
    return fig

def plot_metal_content(tonnage_data, plot_info=None):
    fig = go.Figure()
    
    # Vérifier si les données sont valides
    if (not tonnage_data or 'cutoffs' not in tonnage_data or 'metals' not in tonnage_data or
        len(tonnage_data['cutoffs']) == 0 or len(tonnage_data['metals']) == 0):
        # Ajouter un texte indiquant l'absence de données
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Données insuffisantes pour le graphique de métal contenu",
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    if plot_info and plot_info.get('method') == 'between':
        # Pour la méthode 'between', on utilise un graphique à barres
        if len(tonnage_data['metals']) > 0:
            fig.add_trace(
                go.Bar(
                    x=['Résultat'],
                    y=[tonnage_data['metals'][0]],
                    name='Métal contenu',
                    marker_color='rgb(76, 175, 80)'
                )
            )
    else:
        # Pour les méthodes 'above' et 'below', on utilise un graphique en ligne
        fig.add_trace(
            go.Scatter(
                x=tonnage_data['cutoffs'],
                y=tonnage_data['metals'],
                name='Métal contenu',
                fill='tozeroy',
                mode='lines',
                line=dict(color='rgb(76, 175, 80)')
            )
        )
    
    fig.update_layout(
        title_text='Métal contenu',
        xaxis_title='Teneur de coupure',
        yaxis_title='Métal contenu'
    )
    
    return fig

# Interface utilisateur Streamlit
# Logo en haut de la page
logo = create_mining_logo()
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image(logo, width=150)
    st.title("MineEstim - Estimation par krigeage ordinaire")
    st.caption("Développé par Didier Ouedraogo, P.Geo")

# Sidebar - Chargement des données et paramètres
with st.sidebar:
    st.header("Données")
    
    # Nom du projet
    project_name = st.text_input("Nom du projet", "Projet Minier")
    
    uploaded_file = st.file_uploader("Fichier CSV des composites", type=["csv"])
    
    if uploaded_file:
        try:
            # Section pour la conversion CSV
            st.write("### Options de lecture CSV")
            
            # Chargement automatique avec détection des paramètres
            with st.spinner("Analyse du fichier CSV en cours..."):
                if 'csv_params' not in st.session_state:
                    best_df, best_params = attempt_csv_loading(uploaded_file)
                    st.session_state.csv_params = best_params
                    df = best_df
                    
                    if df is None:
                        st.error("Impossible de charger le fichier CSV. Veuillez vérifier le format.")
                        st.stop()
                else:
                    # Utiliser les paramètres déjà détectés
                    params = st.session_state.csv_params
                    uploaded_file.seek(0)
                    df = pd.read_csv(
                        uploaded_file, 
                        sep=params['separator'], 
                        decimal=params['decimal'], 
                        encoding=params['encoding'],
                        low_memory=False,
                        on_bad_lines='warn'
                    )
            
            # Afficher les paramètres détectés automatiquement
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_encoding = st.selectbox(
                    "Encodage du fichier", 
                    options=['utf-8', 'latin1', 'iso-8859-1', 'cp1252'],
                    index=['utf-8', 'latin1', 'iso-8859-1', 'cp1252'].index(st.session_state.csv_params['encoding'])
                )
            
            with col2:
                separator_options = [',', ';', '\t', '|']
                separator_names = ["Virgule (,)", "Point-virgule (;)", "Tabulation", "Pipe (|)"]
                selected_separator = st.selectbox(
                    "Séparateur", 
                    options=separator_options, 
                    index=separator_options.index(st.session_state.csv_params['separator']),
                    format_func=lambda x: separator_names[separator_options.index(x)]
                )
            
            with col3:
                decimal_options = ['.', ',']
                selected_decimal = st.selectbox(
                    "Séparateur décimal", 
                    options=decimal_options,
                    index=decimal_options.index(st.session_state.csv_params['decimal']),
                    format_func=lambda x: "Point (.)" if x == '.' else "Virgule (,)"
                )
            
            # Recharger avec les paramètres sélectionnés si différents
            if (selected_encoding != st.session_state.csv_params['encoding'] or
                selected_separator != st.session_state.csv_params['separator'] or
                selected_decimal != st.session_state.csv_params['decimal']):
                
                uploaded_file.seek(0)
                df = pd.read_csv(
                    uploaded_file, 
                    sep=selected_separator, 
                    decimal=selected_decimal, 
                    encoding=selected_encoding,
                    low_memory=False,
                    on_bad_lines='warn'
                )
                
                # Mettre à jour les paramètres stockés
                st.session_state.csv_params = {
                    'encoding': selected_encoding,
                    'separator': selected_separator, 
                    'decimal': selected_decimal
                }
            
            st.success(f"{len(df)} lignes chargées")
            
            # Afficher un aperçu des données
            with st.expander("Aperçu des données chargées", expanded=True):
                st.write("Aperçu des premières lignes :", df.head())
                st.write("Types de données :", df.dtypes)
                st.write("Nombre de valeurs non-nulles :", df.count())
            
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {str(e)}")
            st.stop()
        
        # Vérifier si le DataFrame est valide
        if df is None or df.empty:
            st.error("Le fichier CSV ne contient pas de données valides.")
            st.stop()
        
        # Mappage des colonnes
        st.subheader("Mappage des colonnes")
        
        try:
            # Sélection des colonnes
            col_x = st.selectbox("Colonne X", options=df.columns, 
                                index=df.columns.get_loc('X') if 'X' in df.columns else 0)
            col_y = st.selectbox("Colonne Y", options=df.columns, 
                                index=df.columns.get_loc('Y') if 'Y' in df.columns else 0)
            col_z = st.selectbox("Colonne Z", options=df.columns, 
                                index=df.columns.get_loc('Z') if 'Z' in df.columns else 0)
            
            # Colonne de teneur
            value_col_index = (df.columns.get_loc('VALUE') if 'VALUE' in df.columns else 0)
            col_value = st.selectbox("Colonne Teneur", options=df.columns, index=value_col_index)
        except Exception as e:
            st.error(f"Erreur lors du mappage des colonnes: {str(e)}")
            st.stop()
        
        # Option pour la densité
        density_options = ["Constante", "Variable (colonne)"]
        density_method = st.radio("Méthode de densité", options=density_options)
        
        if density_method == "Constante":
            density_value = st.number_input("Densité (t/m³)", min_value=0.1, value=2.7, step=0.1)
            density_column = None
        else:
            density_column = st.selectbox("Colonne Densité", options=df.columns, 
                                        index=df.columns.get_loc('DENSITY') if 'DENSITY' in df.columns else 0)
            if density_column in df.columns:
                st.info(f"Densité moyenne des échantillons: {df[density_column].mean():.2f} t/m³")
        
        # Filtres optionnels
        st.subheader("Filtre (facultatif)")
        
        domain_options = ['-- Aucun --'] + list(df.columns)
        domain_index = domain_options.index('DOMAIN') if 'DOMAIN' in domain_options else 0
        col_domain = st.selectbox("Colonne de domaine", options=domain_options, index=domain_index)
        
        # Si un domaine est sélectionné
        domain_filter_value = None
        if col_domain != '-- Aucun --':
            domain_filter_type = st.selectbox("Type de filtre", options=["=", "!=", "IN", "NOT IN"])
            
            if domain_filter_type in ["=", "!="]:
                unique_values = df[col_domain].dropna().unique()
                if len(unique_values) > 0:
                    # Utiliser une liste déroulante pour sélectionner une valeur unique
                    domain_filter_value = st.selectbox("Valeur", options=unique_values)
                else:
                    domain_filter_value = st.text_input("Valeur")
            else:
                domain_values = df[col_domain].dropna().unique()
                domain_filter_value = st.multiselect("Valeurs", options=domain_values)
        
        # Enveloppe DXF
        st.subheader("Enveloppe (facultatif)")
        
        envelope_method = st.radio("Méthode d'enveloppe", ["Manuelle", "DXF"])
        
        if envelope_method == "DXF":
            uploaded_dxf = st.file_uploader("Fichier DXF de l'enveloppe", type=["dxf"])
            
            if uploaded_dxf:
                envelope_data = process_dxf_file(uploaded_dxf)
                if envelope_data:
                    st.success(f"Enveloppe DXF chargée avec succès")
                    st.session_state.envelope_data = envelope_data
                else:
                    st.error("Impossible de traiter le fichier DXF. Assurez-vous qu'il contient des entités 3D fermées.")
                    st.session_state.envelope_data = None
        else:  # Enveloppe manuelle
            try:
                col1, col2 = st.columns(2)
                
                # Valeurs par défaut pour les limites min/max
                default_min_x = float(df[col_x].min()) if pd.api.types.is_numeric_dtype(df[col_x]) else 0
                default_min_y = float(df[col_y].min()) if pd.api.types.is_numeric_dtype(df[col_y]) else 0
                default_min_z = float(df[col_z].min()) if pd.api.types.is_numeric_dtype(df[col_z]) else 0
                default_max_x = float(df[col_x].max()) if pd.api.types.is_numeric_dtype(df[col_x]) else 100
                default_max_y = float(df[col_y].max()) if pd.api.types.is_numeric_dtype(df[col_y]) else 100
                default_max_z = float(df[col_z].max()) if pd.api.types.is_numeric_dtype(df[col_z]) else 100
                
                with col1:
                    st.markdown("Minimum")
                    min_x = st.number_input("Min X", value=default_min_x, format="%.2f")
                    min_y = st.number_input("Min Y", value=default_min_y, format="%.2f")
                    min_z = st.number_input("Min Z", value=default_min_z, format="%.2f")
                
                with col2:
                    st.markdown("Maximum")
                    max_x = st.number_input("Max X", value=default_max_x, format="%.2f")
                    max_y = st.number_input("Max Y", value=default_max_y, format="%.2f")
                    max_z = st.number_input("Max Z", value=default_max_z, format="%.2f")
                
                envelope_bounds = {
                    'min': {'x': min_x, 'y': min_y, 'z': min_z},
                    'max': {'x': max_x, 'y': max_y, 'z': max_z}
                }
                
                # Créer une enveloppe simple à partir des limites manuelles
                if envelope_method == "Manuelle":
                    # Créer un cube simple à partir des limites
                    vertices = [
                        (min_x, min_y, min_z),
                        (max_x, min_y, min_z),
                        (max_x, max_y, min_z),
                        (min_x, max_y, min_z),
                        (min_x, min_y, max_z),
                        (max_x, min_y, max_z),
                        (max_x, max_y, max_z),
                        (min_x, max_y, max_z)
                    ]
                    
                    # Définir les faces du cube
                    faces = [
                        [0, 1, 2], [0, 2, 3],  # bottom
                        [4, 5, 6], [4, 6, 7],  # top
                        [0, 1, 5], [0, 5, 4],  # front
                        [2, 3, 7], [2, 7, 6],  # back
                        [0, 3, 7], [0, 7, 4],  # left
                        [1, 2, 6], [1, 6, 5]   # right
                    ]
                    
                    # Créer un maillage trimesh
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                    
                    # Stocker dans la session
                    st.session_state.envelope_data = {
                        'mesh': mesh,
                        'bounds': envelope_bounds,
                        'vertices': vertices,
                        'faces': faces
                    }
            except Exception as e:
                st.error(f"Erreur lors de la création de l'enveloppe manuelle: {str(e)}")
                st.session_state.envelope_data = None
        
        use_envelope = st.checkbox("Restreindre l'estimation à l'enveloppe", value=True)
        st.session_state.use_envelope = use_envelope
    
    # Paramètres du krigeage
    st.header("Paramètres du krigeage")
    
    # Type de variogramme
    variogram_type = st.selectbox(
        "Type de variogramme", 
        options=["spherical", "exponential", "gaussian"],
        format_func=lambda x: "Sphérique" if x == "spherical" else "Exponentiel" if x == "exponential" else "Gaussien"
    )
    
    # Paramètres du variogramme
    col1, col2 = st.columns(2)
    
    with col1:
        nugget = st.number_input("Effet pépite", min_value=0.0, value=0.0, step=0.01)
        sill = st.number_input("Palier", min_value=0.01, value=1.0, step=0.1)
    
    with col2:
        range_val = st.number_input("Portée (m)", min_value=1.0, value=100.0, step=10.0)
    
    # Anisotropie
    st.subheader("Anisotropie (ratio des distances)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        anisotropy_x = st.number_input("X", min_value=0.1, value=1.0, step=0.1)
    
    with col2:
        anisotropy_y = st.number_input("Y", min_value=0.1, value=1.0, step=0.1)
    
    with col3:
        anisotropy_z = st.number_input("Z", min_value=0.1, value=0.5, step=0.1)
    
    # Paramètres du modèle de blocs
    st.header("Paramètres du modèle")
    
    st.subheader("Taille des blocs (m)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        block_size_x = st.number_input("X", min_value=1, value=10, step=1)
    
    with col2:
        block_size_y = st.number_input("Y", min_value=1, value=10, step=1)
    
    with col3:
        block_size_z = st.number_input("Z", min_value=1, value=5, step=1)
    
    st.subheader("Rayon de recherche (m)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_radius_x = st.number_input("X ", min_value=1, value=50, step=1)
    
    with col2:
        search_radius_y = st.number_input("Y ", min_value=1, value=50, step=1)
    
    with col3:
        search_radius_z = st.number_input("Z ", min_value=1, value=25, step=1)
    
    min_samples = st.number_input("Nombre min d'échantillons", min_value=1, value=2, step=1)
    max_samples = st.number_input("Nombre max d'échantillons", min_value=1, value=12, step=1)

# Traitement des données
if uploaded_file:
    # Diagnostic des données
    st.subheader("Traitement et validation des données")
    
    try:
        # Préparer le filtrage par domaine
        domain_filter = None
        if col_domain != '-- Aucun --' and domain_filter_value is not None:
            domain_filter = (domain_filter_type, domain_filter_value)
        
        # Nettoyer et préparer les données
        cleaned_df, cleaning_stats = clean_and_prepare_data(
            df, col_x, col_y, col_z, col_value, col_domain, domain_filter
        )
        
        # Afficher les statistiques de nettoyage
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Graphique montrant l'impact des étapes de nettoyage
            fig = plot_data_cleaning_steps(cleaning_stats)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("### Détails du nettoyage")
            st.dataframe(cleaning_stats, use_container_width=True, hide_index=True)
            
            final_count = cleaning_stats['Nombre de lignes'].iloc[-1]
            initial_count = cleaning_stats['Nombre de lignes'].iloc[0]
            percent_kept = (final_count / initial_count * 100) if initial_count > 0 else 0
            
            st.metric(
                "Échantillons valides", 
                f"{final_count}", 
                f"{percent_kept:.1f}% des données originales"
            )
        
        # Vérifier s'il y a des lignes valides après nettoyage
        if len(cleaned_df) == 0:
            st.error("Aucun échantillon valide après nettoyage. Vérifiez vos données et filtres.")
            st.stop()
        
        # Convertir en liste de composites
        composites_data = df_to_composites(
            cleaned_df, col_x, col_y, col_z, col_value, 
            col_domain, density_column if density_method == "Variable (colonne)" else None
        )
        
        # Afficher un aperçu des composites
        with st.expander("Aperçu des composites", expanded=False):
            st.write("### 5 premiers composites")
            composites_preview = pd.DataFrame(composites_data[:5])
            st.write(composites_preview)
            
            st.write("### Statistiques des composites")
            composite_values = [comp['VALUE'] for comp in composites_data]
            
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                st.write(f"Nombre total de composites : {len(composites_data)}")
                if len(composite_values) > 0:
                    st.write(f"Teneur minimale : {min(composite_values):.3f}")
                    st.write(f"Teneur maximale : {max(composite_values):.3f}")
                    st.write(f"Teneur moyenne : {sum(composite_values)/len(composite_values):.3f}")
            
            with col_stats2:
                # Afficher un histogramme miniature
                if len(composite_values) > 1:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    n_bins = min(20, max(5, int(1 + 3.322 * math.log10(len(composite_values)))))
                    ax.hist(composite_values, bins=n_bins, color='steelblue', alpha=0.7)
                    ax.set_title("Distribution des teneurs")
                    ax.set_xlabel("Teneur")
                    ax.set_ylabel("Fréquence")
                    st.pyplot(fig)
        
        # Validation finale
        if len(composites_data) < 2:
            st.warning("Un minimum de 2 composites est recommandé pour l'estimation. Les résultats pourraient ne pas être fiables.")
        
        # Succès!
        st.success(f"Traitement des données réussi : {len(composites_data)} composites valides prêts pour l'estimation!")
        
    except Exception as e:
        st.error(f"Erreur lors du traitement des données : {str(e)}")
        show_detailed_error("Erreur détaillée", e)
        st.stop()
    
    # Afficher les statistiques des composites
    composite_values = [comp['VALUE'] for comp in composites_data]
    composite_stats = calculate_stats(composite_values)
    
    # Créer le modèle de variogramme
    variogram_model = {
        'type': variogram_type,
        'nugget': nugget,
        'sill': sill,
        'range': range_val,
        'total_sill': nugget + sill
    }
    
    # Enregistrer le modèle de variogramme dans la session
    st.session_state.variogram_model = variogram_model
    
    # Onglets principaux
    tabs = st.tabs(["Modèle 3D", "Statistiques", "Tonnage-Teneur", "Rapport"])
    
    with tabs[0]:  # Modèle 3D
        st.subheader("Modèle de blocs 3D")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            create_model_button = st.button("Créer le modèle de blocs", type="primary")
            
            if "empty_blocks" in st.session_state and st.session_state.empty_blocks:
                estimate_button = st.button("Estimer par krigeage ordinaire", type="primary")
            
            # Options d'affichage
            st.subheader("Options d'affichage")
            show_composites = st.checkbox("Afficher les composites", value=True)
            show_blocks = st.checkbox("Afficher les blocs", value=True)
            show_envelope = st.checkbox("Afficher l'enveloppe", value=True if 'envelope_data' in st.session_state and st.session_state.envelope_data else False)
            
            # Taille des cubes
            block_scale = st.slider("Taille des blocs (échelle)", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
            
            # Option de coloration
            if "estimated_blocks" in st.session_state:
                color_by = st.radio(
                    "Colorer par", 
                    options=["value", "estimation_variance"],
                    format_func=lambda x: "Teneur" if x == "value" else "Variance d'estimation"
                )
            else:
                color_by = "value"
        
        with col1:
            try:
                if create_model_button:
                    # Créer le modèle de blocs vide
                    block_sizes = {'x': block_size_x, 'y': block_size_y, 'z': block_size_z}
                    envelope_data = st.session_state.envelope_data if 'envelope_data' in st.session_state else None
                    use_envelope = st.session_state.use_envelope if 'use_envelope' in st.session_state else False
                    
                    empty_blocks, model_bounds = create_block_model(
                        composites_data, 
                        block_sizes, 
                        envelope_data, 
                        use_envelope
                    )
                    
                    if not empty_blocks:
                        st.error("Impossible de créer le modèle de blocs. Vérifiez vos paramètres et données.")
                    else:
                        st.session_state.empty_blocks = empty_blocks
                        st.session_state.model_bounds = model_bounds
                        
                        st.success(f"Modèle créé avec {len(empty_blocks)} blocs")
                        
                        # Afficher le modèle 3D
                        envelope_data_to_show = envelope_data if show_envelope and envelope_data else None
                        fig = plot_3d_model_with_cubes(
                            [],
                            composites_data if show_composites else [],
                            envelope_data_to_show,
                            block_scale
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                elif "empty_blocks" in st.session_state and estimate_button:
                    # Paramètres pour le krigeage
                    kriging_params = {
                        'anisotropy': {'x': anisotropy_x, 'y': anisotropy_y, 'z': anisotropy_z}
                    }
                    
                    search_params = {
                        'x': search_radius_x,
                        'y': search_radius_y,
                        'z': search_radius_z,
                        'min_samples': min_samples,
                        'max_samples': max_samples
                    }
                    
                    # Détermine la méthode de densité
                    if density_method == "Variable (colonne)":
                        density_method_str = "variable"
                        density_value_num = None
                    else:
                        density_method_str = "constant"
                        density_value_num = density_value
                    
                    # Estimer le modèle
                    estimated_blocks = estimate_block_model_kriging(
                        st.session_state.empty_blocks, 
                        composites_data,
                        kriging_params,
                        search_params,
                        variogram_model,
                        density_method_str,
                        density_value_num
                    )
                    
                    if not estimated_blocks:
                        st.error("L'estimation n'a pas produit de blocs. Vérifiez vos paramètres.")
                    else:
                        st.session_state.estimated_blocks = estimated_blocks
                        
                        # Stocker les paramètres pour le rapport
                        st.session_state.kriging_params = kriging_params
                        st.session_state.search_params = search_params
                        st.session_state.block_sizes = {'x': block_size_x, 'y': block_size_y, 'z': block_size_z}
                        st.session_state.density_method = density_method_str
                        st.session_state.density_value = density_value_num if density_method_str == "constant" else None
                        st.session_state.density_column = density_column if density_method_str == "variable" else None
                        st.session_state.project_name = project_name
                        
                        st.success(f"Estimation terminée, {len(estimated_blocks)} blocs estimés")
                        
                        # Afficher le modèle estimé
                        envelope_data_to_show = st.session_state.envelope_data if show_envelope and 'envelope_data' in st.session_state and st.session_state.envelope_data else None
                        fig = plot_3d_model_with_cubes(
                            estimated_blocks if show_blocks else [],
                            composites_data if show_composites else [],
                            envelope_data_to_show,
                            block_scale,
                            color_by
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Section d'export
                        st.subheader("Exporter")
                        
                        # Export du modèle de blocs en CSV
                        if st.button("Exporter modèle de blocs (CSV)"):
                            # Créer un DataFrame pour l'export
                            export_df = pd.DataFrame(estimated_blocks)
                            
                            # Renommer les colonnes pour correspondre au format d'origine
                            export_df = export_df.rename(columns={
                                'x': 'X', 'y': 'Y', 'z': 'Z', 'value': 'VALUE',
                                'size_x': 'SIZE_X', 'size_y': 'SIZE_Y', 'size_z': 'SIZE_Z',
                                'density': 'DENSITY', 'estimation_variance': 'KRIG_VAR'
                            })
                            
                            # Ajouter des informations supplémentaires
                            export_df['VOLUME'] = export_df['SIZE_X'] * export_df['SIZE_Y'] * export_df['SIZE_Z']
                            export_df['TONNAGE'] = export_df['VOLUME'] * export_df['DENSITY']
                            export_df['METAL_CONTENT'] = export_df['VALUE'] * export_df['TONNAGE']
                            
                            # Créer le lien de téléchargement
                            csv = export_df.to_csv(index=False)
                            st.download_button(
                                label="Télécharger CSV",
                                data=csv,
                                file_name=f"{project_name.replace(' ', '_')}_modele_blocs_krigeage.csv",
                                mime="text/csv"
                            )
                
                elif "estimated_blocks" in st.session_state:
                    # Afficher le modèle estimé déjà calculé
                    envelope_data_to_show = st.session_state.envelope_data if show_envelope and 'envelope_data' in st.session_state and st.session_state.envelope_data else None
                    fig = plot_3d_model_with_cubes(
                        st.session_state.estimated_blocks if show_blocks else [],
                        composites_data if show_composites else [],
                        envelope_data_to_show,
                        block_scale,
                        color_by
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Section d'export
                    st.subheader("Exporter")
                    
                    # Export du modèle de blocs en CSV
                    if st.button("Exporter modèle de blocs (CSV)"):
                        # Créer un DataFrame pour l'export
                        export_df = pd.DataFrame(st.session_state.estimated_blocks)
                        
                        # Renommer les colonnes pour correspondre au format d'origine
                        export_df = export_df.rename(columns={
                            'x': 'X', 'y': 'Y', 'z': 'Z', 'value': 'VALUE',
                            'size_x': 'SIZE_X', 'size_y': 'SIZE_Y', 'size_z': 'SIZE_Z',
                            'density': 'DENSITY', 'estimation_variance': 'KRIG_VAR'
                        })
                        
                        # Ajouter des informations supplémentaires
                        export_df['VOLUME'] = export_df['SIZE_X'] * export_df['SIZE_Y'] * export_df['SIZE_Z']
                        export_df['TONNAGE'] = export_df['VOLUME'] * export_df['DENSITY']
                        export_df['METAL_CONTENT'] = export_df['VALUE'] * export_df['TONNAGE']
                        
                        # Créer le lien de téléchargement
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="Télécharger CSV",
                            data=csv,
                            file_name=f"{project_name.replace(' ', '_')}_modele_blocs_krigeage.csv",
                            mime="text/csv"
                        )
                
                else:
                    # Afficher seulement les composites si aucun modèle n'est créé
                    envelope_data_to_show = st.session_state.envelope_data if show_envelope and 'envelope_data' in st.session_state and st.session_state.envelope_data else None
                    fig = plot_3d_model_with_cubes(
                        [],
                        composites_data if show_composites else [],
                        envelope_data_to_show,
                        block_scale
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur dans l'onglet Modèle 3D: {str(e)}")
                show_detailed_error("Erreur détaillée", e)
    
    with tabs[1]:  # Statistiques
        st.subheader("Statistiques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Statistiques des composites")
            
            if composite_stats['count'] > 0:
                stats_df = pd.DataFrame({
                    'Paramètre': ['Nombre d\'échantillons', 'Minimum', 'Maximum', 'Moyenne', 'Médiane', 'Écart-type', 'CV'],
                    'Valeur': [
                        composite_stats['count'],
                        f"{composite_stats['min']:.3f}",
                        f"{composite_stats['max']:.3f}",
                        f"{composite_stats['mean']:.3f}",
                        f"{composite_stats['median']:.3f}",
                        f"{composite_stats['stddev']:.3f}",
                        f"{composite_stats['cv']:.3f}"
                    ]
                })
                
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
                
                st.markdown("### Histogramme des composites")
                fig = plot_histogram(composite_values, f"Distribution des teneurs des composites ({col_value})", "darkblue")
                st.pyplot(fig)
            else:
                st.warning("Aucune donnée valide pour calculer les statistiques des composites.")
            
            # Statistiques de densité si disponible
            if density_method == "Variable (colonne)" and density_column:
                st.markdown("### Statistiques de densité")
                density_values = [comp.get('DENSITY') for comp in composites_data if 'DENSITY' in comp]
                if density_values:
                    density_stats = calculate_stats(density_values)
                    
                    density_stats_df = pd.DataFrame({
                        'Paramètre': ['Nombre d\'échantillons', 'Minimum', 'Maximum', 'Moyenne', 'Médiane', 'Écart-type', 'CV'],
                        'Valeur': [
                            density_stats['count'],
                            f"{density_stats['min']:.3f}",
                            f"{density_stats['max']:.3f}",
                            f"{density_stats['mean']:.3f}",
                            f"{density_stats['median']:.3f}",
                            f"{density_stats['stddev']:.3f}",
                            f"{density_stats['cv']:.3f}"
                        ]
                    })
                    
                    st.dataframe(density_stats_df, hide_index=True, use_container_width=True)
            
            # Informations sur le variogramme
            st.markdown("### Paramètres du variogramme")
            variogram_df = pd.DataFrame({
                'Paramètre': ['Type', 'Effet pépite', 'Palier', 'Portée', 'Palier total'],
                'Valeur': [
                    variogram_model['type'],
                    f"{variogram_model['nugget']:.3f}",
                    f"{variogram_model['sill']:.3f}",
                    f"{variogram_model['range']:.1f} m",
                    f"{variogram_model['total_sill']:.3f}"
                ]
            })
            st.dataframe(variogram_df, hide_index=True, use_container_width=True)
        
        with col2:
            if "estimated_blocks" in st.session_state and st.session_state.estimated_blocks:
                block_values = [block.get('value', 0) for block in st.session_state.estimated_blocks]
                block_stats = calculate_stats(block_values)
                
                st.markdown("### Statistiques du modèle de blocs")
                
                stats_df = pd.DataFrame({
                    'Paramètre': ['Nombre de blocs', 'Minimum', 'Maximum', 'Moyenne', 'Médiane', 'Écart-type', 'CV'],
                    'Valeur': [
                        block_stats['count'],
                        f"{block_stats['min']:.3f}",
                        f"{block_stats['max']:.3f}",
                        f"{block_stats['mean']:.3f}",
                        f"{block_stats['median']:.3f}",
                        f"{block_stats['stddev']:.3f}",
                        f"{block_stats['cv']:.3f}"
                    ]
                })
                
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
                
                st.markdown("### Histogramme du modèle de blocs")
                fig = plot_histogram(block_values, f"Distribution des teneurs du modèle de blocs ({col_value})", "teal")
                st.pyplot(fig)
                
                # Statistiques de la variance d'estimation
                variance_values = [block.get('estimation_variance', 0) for block in st.session_state.estimated_blocks if 'estimation_variance' in block]
                if variance_values:
                    st.markdown("### Variance d'estimation")
                    variance_stats = calculate_stats(variance_values)
                    
                    variance_df = pd.DataFrame({
                        'Paramètre': ['Minimum', 'Maximum', 'Moyenne', 'Médiane', 'Écart-type'],
                        'Valeur': [
                            f"{variance_stats['min']:.4f}",
                            f"{variance_stats['max']:.4f}",
                            f"{variance_stats['mean']:.4f}",
                            f"{variance_stats['median']:.4f}",
                            f"{variance_stats['stddev']:.4f}"
                        ]
                    })
                    
                    st.dataframe(variance_df, hide_index=True, use_container_width=True)
                    
                    st.markdown("### Histogramme de la variance d'estimation")
                    fig = plot_histogram(variance_values, "Distribution de la variance d'estimation", "salmon")
                    st.pyplot(fig)
                
                # Résumé des statistiques globales
                st.markdown("### Résumé global")
                
                # Vérifier les clés nécessaires
                if (all(key in st.session_state.estimated_blocks[0] for key in ['size_x', 'size_y', 'size_z']) and 
                    block_stats['count'] > 0):
                    block_volume = st.session_state.estimated_blocks[0]['size_x'] * st.session_state.estimated_blocks[0]['size_y'] * st.session_state.estimated_blocks[0]['size_z']
                    total_volume = len(st.session_state.estimated_blocks) * block_volume
                    
                    # Calcul du tonnage avec densité variable ou constante
                    if density_method == "Variable (colonne)":
                        total_tonnage = sum(block.get('density', density_value) * block_volume for block in st.session_state.estimated_blocks)
                        avg_density = total_tonnage / total_volume if total_volume > 0 else density_value
                    else:
                        avg_density = density_value
                        total_tonnage = total_volume * avg_density
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Nombre de blocs", f"{len(st.session_state.estimated_blocks)}")
                    
                    with col2:
                        st.metric(f"Teneur moyenne {col_value}", f"{block_stats['mean']:.3f}")
                    
                    with col3:
                        st.metric("Volume total (m³)", f"{total_volume:,.0f}")
                    
                    with col4:
                        st.metric("Tonnage total (t)", f"{total_tonnage:,.0f}")
                else:
                    st.warning("Données insuffisantes pour calculer les métriques globales.")
            else:
                st.info("Veuillez d'abord créer et estimer le modèle de blocs pour afficher les statistiques.")
    
    with tabs[2]:  # Tonnage-Teneur
        st.subheader("Analyse Tonnage-Teneur")
        
        if "estimated_blocks" in st.session_state and st.session_state.estimated_blocks:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                cutoff_method = st.selectbox(
                    "Méthode de coupure",
                    options=["above", "below", "between"],
                    format_func=lambda x: "Teneur ≥ Coupure" if x == "above" else "Teneur ≤ Coupure" if x == "below" else "Entre deux teneurs"
                )
            
            cutoff_value = None
            cutoff_min = None
            cutoff_max = None
            
            if cutoff_method == "between":
                with col2:
                    cutoff_min = st.number_input("Teneur min", min_value=0.0, value=0.5, step=0.1)
                
                with col3:
                    cutoff_max = st.number_input("Teneur max", min_value=cutoff_min, value=1.0, step=0.1)
            else:
                with col2:
                    cutoff_value = st.number_input("Teneur de coupure", min_value=0.0, value=0.5, step=0.1)
            
            with col4:
                if st.button("Calculer", type="primary"):
                    try:
                        # Détermine la méthode de densité
                        density_method_str = st.session_state.density_method if 'density_method' in st.session_state else "constant"
                        density_value_num = st.session_state.density_value if 'density_value' in st.session_state else density_value
                        
                        # Calculer les données tonnage-teneur
                        tonnage_data, plot_info = calculate_tonnage_grade(
                            st.session_state.estimated_blocks,
                            density_method_str,
                            density_value_num,
                            cutoff_method,
                            cutoff_value,
                            cutoff_min,
                            cutoff_max
                        )
                        
                        st.session_state.tonnage_data = tonnage_data
                        st.session_state.plot_info = plot_info
                    except Exception as e:
                        st.error(f"Erreur lors du calcul tonnage-teneur: {str(e)}")
                        show_detailed_error("Erreur détaillée", e)
            
            if "tonnage_data" in st.session_state:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Graphique Tonnage-Teneur
                    fig = plot_tonnage_grade(st.session_state.tonnage_data, st.session_state.plot_info)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Graphique Métal contenu
                    fig = plot_metal_content(st.session_state.tonnage_data, st.session_state.plot_info)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Tableau des résultats
                st.subheader("Résultats détaillés")
                
                # Vérifier que les données existent
                if ('plot_info' in st.session_state and 'tonnage_data' in st.session_state and
                    'cutoffs' in st.session_state.tonnage_data and 'tonnages' in st.session_state.tonnage_data and 
                    'grades' in st.session_state.tonnage_data and 'metals' in st.session_state.tonnage_data):
                    
                    if st.session_state.plot_info.get('method') == 'between':
                        # Pour la méthode between, afficher un seul résultat
                        if len(st.session_state.tonnage_data['tonnages']) > 0:
                            result_df = pd.DataFrame({
                                'Coupure': [f"{cutoff_min:.2f} - {cutoff_max:.2f}"],
                                'Tonnage (t)': [st.session_state.tonnage_data['tonnages'][0]],
                                'Teneur moyenne': [st.session_state.tonnage_data['grades'][0]],
                                'Métal contenu': [st.session_state.tonnage_data['metals'][0]]
                            })
                            st.dataframe(result_df, hide_index=True, use_container_width=True)
                        else:
                            st.warning("Aucun résultat pour cette coupure.")
                    else:
                        # Pour les méthodes above et below, afficher la courbe complète
                        result_df = pd.DataFrame({
                            'Coupure': st.session_state.tonnage_data['cutoffs'],
                            'Tonnage (t)': st.session_state.tonnage_data['tonnages'],
                            'Teneur moyenne': st.session_state.tonnage_data['grades'],
                            'Métal contenu': st.session_state.tonnage_data['metals']
                        })
                        st.dataframe(result_df, hide_index=True, use_container_width=True)
                else:
                    st.warning("Données tonnage-teneur incomplètes.")
                
                # Export des résultats
                st.subheader("Exporter les résultats")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export Excel
                    if st.button("Exporter en Excel"):
                        try:
                            # Créer un buffer pour le fichier Excel
                            output = io.BytesIO()
                            
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                # Écrire les données tonnage-teneur
                                result_df.to_excel(writer, sheet_name='Tonnage-Teneur', index=False)
                                
                                # Ajouter une feuille pour les paramètres
                                density_info = f"Variable (colonne {density_column})" if density_method == "Variable (colonne)" else f"Constante ({density_value} t/m³)"
                                
                                param_df = pd.DataFrame({
                                    'Paramètre': [
                                        'Méthode de coupure', 
                                        'Méthode d\'estimation',
                                        'Type de variogramme',
                                        'Effet pépite',
                                        'Palier',
                                        'Portée',
                                        'Taille des blocs (m)',
                                        'Densité',
                                        'Date d\'exportation'
                                    ],
                                    'Valeur': [
                                        "Teneur ≥ Coupure" if cutoff_method == "above" else "Teneur ≤ Coupure" if cutoff_method == "below" else f"Entre {cutoff_min} et {cutoff_max}",
                                        'Krigeage ordinaire',
                                        variogram_type,
                                        nugget,
                                        sill,
                                        range_val,
                                        f"{block_size_x} × {block_size_y} × {block_size_z}",
                                        density_info,
                                        pd.Timestamp.now().strftime('%Y-%m-%d')
                                    ]
                                })
                                param_df.to_excel(writer, sheet_name='Paramètres', index=False)
                            
                            # Télécharger le fichier
                            output.seek(0)
                            st.download_button(
                                label="Télécharger Excel",
                                data=output,
                                file_name=f"{project_name.replace(' ', '_')}_tonnage_teneur_krigeage.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except Exception as e:
                            st.error(f"Erreur lors de l'export Excel: {str(e)}")
                            show_detailed_error("Erreur détaillée", e)
                
                with col2:
                    # Export graphiques PNG
                    if st.button("Exporter graphiques PNG"):
                        try:
                            # Créer un buffer ZIP pour les graphiques
                            zip_buffer = io.BytesIO()
                            
                            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                                # Ajouter le graphique Tonnage-Teneur
                                fig = plot_tonnage_grade(st.session_state.tonnage_data, st.session_state.plot_info)
                                fig_bytes = fig.to_image(format="png", scale=2)
                                zip_file.writestr("tonnage_teneur.png", fig_bytes)
                                
                                # Ajouter le graphique Métal contenu
                                fig = plot_metal_content(st.session_state.tonnage_data, st.session_state.plot_info)
                                fig_bytes = fig.to_image(format="png", scale=2)
                                zip_file.writestr("metal_contenu.png", fig_bytes)
                            
                            # Télécharger le fichier ZIP
                            zip_buffer.seek(0)
                            st.download_button(
                                label="Télécharger graphiques PNG",
                                data=zip_buffer,
                                file_name=f"{project_name.replace(' ', '_')}_graphiques_tonnage_teneur.zip",
                                mime="application/zip"
                            )
                        except Exception as e:
                            st.error(f"Erreur lors de l'export des graphiques: {str(e)}")
                            show_detailed_error("Erreur détaillée", e)
        else:
            st.info("Veuillez d'abord créer et estimer le modèle de blocs pour effectuer l'analyse Tonnage-Teneur.")
    
    with tabs[3]:  # Rapport
        st.subheader("Rapport d'estimation")
        
        if "estimated_blocks" in st.session_state and st.session_state.estimated_blocks:
            if st.button("Générer le rapport PDF", type="primary"):
                try:
                    # Récupérer les paramètres stockés
                    kriging_params = st.session_state.kriging_params if 'kriging_params' in st.session_state else {
                        'anisotropy': {'x': anisotropy_x, 'y': anisotropy_y, 'z': anisotropy_z}
                    }
                    
                    search_params = st.session_state.search_params if 'search_params' in st.session_state else {
                        'x': search_radius_x,
                        'y': search_radius_y,
                        'z': search_radius_z,
                        'min_samples': min_samples,
                        'max_samples': max_samples
                    }
                    
                    block_sizes = st.session_state.block_sizes if 'block_sizes' in st.session_state else {
                        'x': block_size_x, 'y': block_size_y, 'z': block_size_z
                    }
                    
                    density_method_str = st.session_state.density_method if 'density_method' in st.session_state else "constant"
                    density_value_num = st.session_state.density_value if 'density_value' in st.session_state else density_value
                    density_column_name = st.session_state.density_column if 'density_column' in st.session_state else density_column
                    
                    tonnage_data = st.session_state.tonnage_data if 'tonnage_data' in st.session_state else None
                    plot_info = st.session_state.plot_info if 'plot_info' in st.session_state else None
                    
                    # Générer le rapport PDF
                    pdf_buffer = generate_estimation_report(
                        st.session_state.estimated_blocks,
                        composites_data,
                        kriging_params,
                        search_params,
                        block_sizes,
                        variogram_model,
                        tonnage_data,
                        plot_info,
                        density_method_str,
                        density_value_num,
                        density_column_name,
                        project_name
                    )
                    
                    if pdf_buffer:
                        st.success("Rapport PDF généré avec succès!")
                        
                        # Bouton de téléchargement
                        st.download_button(
                            label="Télécharger le rapport PDF",
                            data=pdf_buffer,
                            file_name=f"{project_name.replace(' ', '_')}_rapport_krigeage.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.error("Échec de la génération du rapport PDF.")
                
                except Exception as e:
                    st.error(f"Erreur lors de la génération du rapport: {str(e)}")
                    show_detailed_error("Erreur détaillée", e)
        else:
            st.info("Veuillez d'abord créer et estimer le modèle de blocs pour générer un rapport.")
else:
    st.info("Veuillez charger un fichier CSV de composites pour commencer.")