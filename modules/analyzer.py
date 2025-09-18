import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
from IPython.display import HTML, display
import base64
from io import BytesIO
warnings.filterwarnings('ignore')

# Professional styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("whitegrid")
sns.set_palette("Set2")

class ProfessionalMLAnalyzer:
    """Professional-grade ML experiment analyzer with beautiful visualizations"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir
        self.results_df = None
        self.models_metadata = {}
        self.loaded_models = {}
        
        # Load all data
        self._load_results()
        self._load_models_metadata()
        
        # Clean and prepare data
        self._prepare_data()
    
    def _load_results(self):
        """Load results from CSV/JSON"""
        results_path = os.path.join(self.experiment_dir, "results", "results_summary.csv")
        
        if os.path.exists(results_path):
            self.results_df = pd.read_csv(results_path, index_col=0)
            # Remove error rows
            self.results_df = self.results_df.dropna(subset=['accuracy'])
            print(f"Loaded {len(self.results_df)} successful experiments")
        else:
            raise FileNotFoundError(f"Results file not found: {results_path}")
    
    def _load_models_metadata(self):
        """Load metadata for all models"""
        models_dir = os.path.join(self.experiment_dir, "models")
        
        if not os.path.exists(models_dir):
            return
        
        metadata_files = [f for f in os.listdir(models_dir) if f.endswith("_metadata.json")]
        
        for metadata_file in metadata_files:
            model_name = metadata_file.replace("_metadata.json", "")
            metadata_path = os.path.join(models_dir, metadata_file)
            
            with open(metadata_path, 'r') as f:
                self.models_metadata[model_name] = json.load(f)
    
    def _prepare_data(self):
        """Clean and prepare data for analysis"""
        if self.results_df is None:
            return
        
        # Extract components from model names for better analysis
        components = []
        for name in self.results_df.index:
            parts = name.split('_')
            
            # Extract scaler
            scaler = parts[0] if len(parts) > 0 else 'unknown'
            
            # Extract PCA
            pca = 'None'
            for part in parts:
                if 'pca' in part:
                    pca = part
                    break
            
            # Extract model (everything after PCA or scaler)
            model_parts = []
            skip_next = False
            for i, part in enumerate(parts):
                if skip_next:
                    skip_next = False
                    continue
                if 'pca' in part or part in ['standard', 'minmax', 'robust']:
                    if 'pca' not in part:  # It's a scaler
                        continue
                    else:  # It's PCA
                        continue
                model_parts.append(part)
            
            model = '_'.join(model_parts) if model_parts else 'unknown'
            
            components.append({
                'scaler': scaler,
                'pca': pca,
                'model': model,
                'name': name
            })
        
        self.components_df = pd.DataFrame(components)
        self.results_df = self.results_df.join(self.components_df.set_index('name'))
    
    def generate_html_report(self, top_k: int = 20, bottom_k: int = 10):
        """Generate beautiful HTML report"""
        
        # Get top and bottom models
        top_models = self.results_df.nlargest(top_k, 'accuracy')
        bottom_models = self.results_df.nsmallest(bottom_k, 'accuracy')
        
        # Generate plots and convert to base64
        plots_html = self._generate_all_plots(top_k, bottom_k)
        
        # Generate summary stats
        stats_html = self._generate_summary_stats()
        
        # Generate insights
        insights_html = self._generate_insights(top_models, bottom_models)
        
        # Combine everything
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Experiment Analysis Report</title>
        <style>
            body {{
                font-family: 'Times New Roman', serif;
                line-height: 1.6;
                color: #212529;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                background-color: #ffffff; /* trắng tinh, không xám */
            }}
            .header {{
                background: #f5f5f5;
                color: #212529;
                padding: 25px;
                border-bottom: 1px solid #ccc;
                margin-bottom: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2em;
                font-weight: 600;
                font-family: 'Georgia', serif;
            }}
            h2 {{
                font-family: 'Georgia', serif;
                font-size: 1.6em;
                color: #212529;
                border-bottom: 1px solid #aaa;
                padding-bottom: 6px;
                margin-top: 35px;
                margin-bottom: 18px;
            }}
            h3 {{
                font-family: 'Georgia', serif;
                font-size: 1.2em;
                color: #343a40;
                margin-top: 25px;
                margin-bottom: 12px;
            }}
            .section {{
                background: #fff;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 3px;
                margin-bottom: 25px;
            }}
            .metric-card {{
                background: #f8f9fa;
                color: #212529;
                padding: 12px;
                border-radius: 3px;
                text-align: center;
                margin: 6px;
                min-width: 120px;
                border: 1px solid #ccc;
            }}
            .metric-card h3 {{
                margin: 0;
                font-size: 1.6em;
                font-weight: 700;
                color: #212529;
            }}
            .metric-card p {{
                margin: 2px 0 0 0;
                font-size: 0.85em;
                color: #555;
            }}
            .model-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 0.9em;
            }}
            .model-table th, .model-table td {{
                padding: 6px 10px;
                text-align: left;
                border-bottom: 1px solid #ccc;
            }}
            .model-table th {{
                background-color: #f2f2f2;
                color: #212529;
                font-weight: 600;
            }}
            .insight-box {{
                background-color: #f8f9fa;
                border-left: 3px solid #333; /* neutral dark line */
                padding: 12px;
                margin: 18px 0;
                border-radius: 3px;
                font-size: 0.9em;
            }}
            .plot-container img {{
                max-width: 100%;
                border: 1px solid #ccc;
                border-radius: 3px;
                box-shadow: none; /* bỏ shadow màu mè */
            }}
        </style>

        </head>
        <body>
            <div class="header">
                <h1>ML Experiment Analysis Report</h1>
                <p>Comprehensive analysis of {len(self.results_df)} machine learning models</p>
                <p>Best Accuracy: {self.results_df['accuracy'].max():.4f} | Worst: {self.results_df['accuracy'].min():.4f}</p>
            </div>
            
            {stats_html}
            {insights_html}
            {plots_html}
        </body>
        </html>
        """
        
        # Save and display
        report_path = os.path.join(self.experiment_dir, "analysis_report.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        display(HTML(html_content))
        print(f"HTML report saved to: {report_path}")
        
        return html_content
    
    def _generate_summary_stats(self):
        """Generate summary statistics HTML"""
        metrics = ['accuracy', 'weighted_precision', 'weighted_recall', 'weighted_f1']
        
        cards_html = ""
        for metric in metrics:
            if metric in self.results_df.columns:
                best_val = self.results_df[metric].max()
                mean_val = self.results_df[metric].mean()
                cards_html += f"""
                <div class="metric-card">
                    <h3>{best_val:.3f}</h3>
                    <p>{metric.replace('_', ' ').title()}</p>
                    <p>Avg: {mean_val:.3f}</p>
                </div>
                """
        
        return f"""
        <div class="section">
            <h2>Performance Overview</h2>
            <div class="metrics-container">
                {cards_html}
            </div>
        </div>
        """
    
    def _generate_insights(self, top_models, bottom_models):
        """Generate insights HTML"""
        
        # Best models table
        best_table = self._create_model_table(top_models.head(10), "best-model", "Top 10 Models")
        
        # Worst models table  
        worst_table = self._create_model_table(bottom_models.head(10), "worst-model", "Bottom 10 Models")
        
        # Component analysis
        component_analysis = self._generate_component_analysis()
        
        return f"""
        <div class="section">
            <h2>Model Performance Analysis</h2>
            <div class="two-column">
                <div>{best_table}</div>
                <div>{worst_table}</div>
            </div>
        </div>
        
        {component_analysis}
        """
    
    def _create_model_table(self, models_df, css_class, title):
        """Create HTML table for models with LaTeX-style formatting"""
        rows_html = ""
        for i, (name, row) in enumerate(models_df.iterrows(), 1):
            # Truncate long names
            display_name = name[:45] + "..." if len(name) > 45 else name
            
            # Add special formatting for top ranks
            rank_class = ""
            if i == 1:
                rank_class = "rank-1"  # Bold for best
                css_class += " best-model"
            elif i == 2:
                rank_class = "rank-2"  # Underline for second best
                css_class += " second-best"
            
            rows_html += f"""
            <tr class="{css_class}">
                <td class="{rank_class}"><strong>{i}</strong></td>
                <td class="{rank_class}">{display_name}</td>
                <td class="{rank_class}">{row['accuracy']:.4f}</td>
                <td class="{rank_class}">{row['weighted_f1']:.4f}</td>
                <td class="{rank_class}">{row['weighted_precision']:.4f}</td>
                <td class="{rank_class}">{row['weighted_recall']:.4f}</td>
            </tr>
            """
        
        return f"""
        <h3>{title}</h3>
        <table class="model-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model Configuration</th>
                    <th>Accuracy</th>
                    <th>F1-Score</th>
                    <th>Precision</th>
                    <th>Recall</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """
    
    def _generate_component_analysis(self):
        """Generate component analysis HTML"""
        
        # Analyze each component type
        analyses = []
        
        # Scaler analysis
        if 'scaler' in self.results_df.columns:
            scaler_stats = self.results_df.groupby('scaler')['accuracy'].agg(['mean', 'std', 'count']).round(4)
            scaler_stats = scaler_stats.sort_values('mean', ascending=False)
            
            scaler_html = "<h4>Scaler Performance</h4><ul>"
            for scaler, row in scaler_stats.iterrows():
                scaler_html += f"<li><strong>{scaler}:</strong> {row['mean']:.4f} ± {row['std']:.4f} ({row['count']} models)</li>"
            scaler_html += "</ul>"
            analyses.append(scaler_html)
        
        # PCA analysis
        if 'pca' in self.results_df.columns:
            pca_stats = self.results_df.groupby('pca')['accuracy'].agg(['mean', 'std', 'count']).round(4)
            pca_stats = pca_stats.sort_values('mean', ascending=False)
            
            pca_html = "<h4>PCA Performance</h4><ul>"
            for pca, row in pca_stats.iterrows():
                pca_html += f"<li><strong>{pca}:</strong> {row['mean']:.4f} ± {row['std']:.4f} ({row['count']} models)</li>"
            pca_html += "</ul>"
            analyses.append(pca_html)
        
        # Model analysis
        if 'model' in self.results_df.columns:
            model_stats = self.results_df.groupby('model')['accuracy'].agg(['mean', 'std', 'count']).round(4)
            model_stats = model_stats.sort_values('mean', ascending=False)
            
            model_html = "<h4>Model Type Performance</h4><ul>"
            for model, row in model_stats.iterrows():
                model_html += f"<li><strong>{model}:</strong> {row['mean']:.4f}</li>"
            model_html += "</ul>"
            analyses.append(model_html)
        
        # Key insights
        insights = self._generate_key_insights()
        
        return f"""
        <div class="section">
            <h2>Component Analysis</h2>
            <div class="two-column">
                <div>
                    {''.join(analyses[:2])}
                </div>
                <div>
                    {''.join(analyses[2:])}
                    {insights}
                </div>
            </div>
        </div>
        """
    
    def _generate_key_insights(self):
        """Generate key insights"""
        insights = []
        
        # Performance spread
        acc_std = self.results_df['accuracy'].std()
        if acc_std < 0.02:
            insights.append("Models show similar performance - consider using simpler approaches")
        elif acc_std > 0.1:
            insights.append("High performance variance detected - some techniques much more effective")
        
        # Best components
        if 'scaler' in self.results_df.columns:
            best_scaler = self.results_df.groupby('scaler')['accuracy'].mean().idxmax()
            insights.append(f"Best scaler: {best_scaler}")
        
        if 'model' in self.results_df.columns:
            best_model = self.results_df.groupby('model')['accuracy'].mean().idxmax()
            insights.append(f"Best model type: {best_model}")
        
        insights_html = "<div class='insight-box'><h4>Key Insights</h4><ul>"
        for insight in insights:
            insights_html += f"<li>{insight}</li>"
        insights_html += "</ul></div>"
        
        return insights_html
    
    def _generate_all_plots(self, top_k, bottom_k):
        """Generate all plots and return HTML"""
        plots_html = "<div class='section'><h2>Visual Analysis</h2>"
        
        # 1. Performance distribution
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Clean histogram
        ax.hist(self.results_df['accuracy'], bins=25, alpha=0.8, color='#3498db', edgecolor='#2c3e50', linewidth=1)
        ax.axvline(self.results_df['accuracy'].mean(), color='#e74c3c', linestyle='--', linewidth=2, 
                  label=f'Mean: {self.results_df["accuracy"].mean():.4f}')
        ax.axvline(self.results_df['accuracy'].max(), color='#27ae60', linestyle='--', linewidth=2,
                  label=f'Best: {self.results_df["accuracy"].max():.4f}')
        
        ax.set_xlabel('Accuracy Score', fontsize=12, fontweight='600')
        ax.set_ylabel('Number of Models', fontsize=12, fontweight='600')
        ax.set_title('Distribution of Model Performance', fontsize=14, fontweight='600', pad=20)
        ax.legend(frameon=True, fancybox=False, shadow=False)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        img_str = self._fig_to_base64(fig)
        plt.close()
        
        plots_html += f"""
        <div class="plot-container">
            <h3>Performance Distribution</h3>
            <img src="data:image/png;base64,{img_str}" alt="Performance Distribution">
        </div>
        """
        
        # 2. Top vs Bottom comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Top models
        top_models = self.results_df.nlargest(15, 'accuracy')
        colors_top = sns.color_palette("Blues_r", n_colors=len(top_models))
        
        bars1 = ax1.barh(range(len(top_models)), top_models['accuracy'], color=colors_top)
        ax1.set_yticks(range(len(top_models)))
        ax1.set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in top_models.index], 
                           fontsize=9)
        ax1.set_xlabel('Accuracy', fontsize=12, fontweight='600')
        ax1.set_title('Top 15 Models', fontsize=14, fontweight='600')
        ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Add value labels only if they fit
        for i, (bar, value) in enumerate(zip(bars1, top_models['accuracy'])):
            width = bar.get_width()
            if width > 0.1:  # Only add label if bar is wide enough
                ax1.text(width - 0.01, bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                        va='center', ha='right', fontsize=8, fontweight='600', color='white')
        
        # Bottom models
        bottom_models = self.results_df.nsmallest(15, 'accuracy')
        colors_bottom = sns.color_palette("Reds", n_colors=len(bottom_models))
        
        bars2 = ax2.barh(range(len(bottom_models)), bottom_models['accuracy'], color=colors_bottom)
        ax2.set_yticks(range(len(bottom_models)))
        ax2.set_yticklabels([name[:30] + '...' if len(name) > 30 else name for name in bottom_models.index],
                           fontsize=9)
        ax2.set_xlabel('Accuracy', fontsize=12, fontweight='600')
        ax2.set_title('Bottom 15 Models', fontsize=14, fontweight='600')
        ax2.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Add value labels only if they fit
        for i, (bar, value) in enumerate(zip(bars2, bottom_models['accuracy'])):
            width = bar.get_width()
            if width > 0.1:  # Only add label if bar is wide enough
                ax2.text(width - 0.01, bar.get_y() + bar.get_height()/2, f'{value:.3f}',
                        va='center', ha='right', fontsize=8, fontweight='600', color='white')
        
        plt.tight_layout()
        img_str = self._fig_to_base64(fig)
        plt.close()
        
        plots_html += f"""
        <div class="plot-container">
            <h3>Best vs Worst Performing Models</h3>
            <img src="data:image/png;base64,{img_str}" alt="Top vs Bottom Models">
        </div>
        """
        
        # 3. Component performance comparison
        # if 'scaler' in self.results_df.columns and 'model' in self.results_df.columns:
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
        #     # Scaler comparison
        #     scaler_stats = self.results_df.groupby('scaler')['accuracy'].agg(['mean', 'std']).sort_values('mean', ascending=True)
            
        #     bars = ax1.barh(range(len(scaler_stats)), scaler_stats['mean'], 
        #                    xerr=scaler_stats['std'], capsize=4, alpha=0.8, 
        #                    color='#34495e', ecolor='#2c3e50', linewidth=1.5)
        #     ax1.set_yticks(range(len(scaler_stats)))
        #     ax1.set_yticklabels(scaler_stats.index, fontsize=11)
        #     ax1.set_xlabel('Mean Accuracy', fontsize=12, fontweight='600')
        #     ax1.set_title('Scaler Performance Comparison', fontsize=14, fontweight='600')
        #     ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        #     ax1.spines['top'].set_visible(False)
        #     ax1.spines['right'].set_visible(False)
            
        #     # Model comparison
        #     model_stats = self.results_df.groupby('model')['accuracy'].agg(['mean', 'std']).sort_values('mean', ascending=True)
            
        #     bars = ax2.barh(range(len(model_stats)), model_stats['mean'],
        #                    xerr=model_stats['std'], capsize=4, alpha=0.8, 
        #                    color='#2c3e50', ecolor='#34495e', linewidth=1.5)
        #     ax2.set_yticks(range(len(model_stats)))
        #     ax2.set_yticklabels(model_stats.index, fontsize=11)
        #     ax2.set_xlabel('Mean Accuracy', fontsize=12, fontweight='600') 
        #     # ax2.set_title('Model Type Performance Comparison', fontsize=12, fontweight='600')
        #     # ax2.grid(axis='x', alpha=0.3, linestyle='-', linewidth=2)
        #     # ax2.spines['top'].set_visible(False)
        #     # ax2.spines['right'].set_visible(False)
            
        #     plt.tight_layout()
        #     img_str = self._fig_to_base64(fig)
        #     plt.close()
            
        #     plots_html += f"""
        #     <div class="plot-container">
        #         <h3>Component Performance Analysis</h3>
        #         <img src="data:image/png;base64,{img_str}" alt="Component Analysis">
        #     </div>
        #     """
        
        plots_html += "</div>"
        return plots_html
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        buffer.close()
        return img_base64
    
    def load_top_models(self, metric: str = "accuracy", top_k: int = 20):
        """Load the top K models into memory"""
        top_models = self.results_df.nlargest(top_k, metric)
        models_dir = os.path.join(self.experiment_dir, "models")
        
        print(f"Loading top {top_k} models by {metric}...")
        
        loaded_count = 0
        for model_name in top_models.index:
            model_path = os.path.join(models_dir, f"{model_name}.joblib")
            
            if os.path.exists(model_path):
                try:
                    self.loaded_models[model_name] = joblib.load(model_path)
                    loaded_count += 1
                except Exception as e:
                    print(f"Failed to load {model_name}: {str(e)}")
        
        print(f"Successfully loaded {loaded_count} models")
        return list(self.loaded_models.keys())


# Main analysis function
def create_professional_analysis(experiment_dir: str, top_k: int = 20, bottom_k: int = 10):
    """
    Create professional ML analysis report
    
    Args:
        experiment_dir: Path to experiment results
        top_k: Number of top models to analyze
        bottom_k: Number of bottom models to analyze
    
    Returns:
        analyzer: Professional analyzer instance
    """
    
    print("Creating Professional ML Analysis Report...")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = ProfessionalMLAnalyzer(experiment_dir)
    
    # Generate beautiful HTML report
    analyzer.generate_html_report(top_k=top_k, bottom_k=bottom_k)
    
    # Load top models
    loaded_models = analyzer.load_top_models(top_k=top_k)
    
    print(f"\nAnalysis Complete!")
    print(f"- Top {top_k} models analyzed and loaded")
    print(f"- Bottom {bottom_k} models analyzed")
    print(f"- Best model: {analyzer.results_df['accuracy'].idxmax()}")
    print(f"- Best accuracy: {analyzer.results_df['accuracy'].max():.4f}")
    
    return analyzer