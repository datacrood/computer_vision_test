import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_analyze_data(csv_file_path):
    """Load CSV data and perform initial analysis"""
    # Load the data
    df = pd.read_csv(csv_file_path)
    
    # Convert timecode to seconds for easier analysis
    def timecode_to_seconds(timecode):
        parts = timecode.split(':')
        hours, minutes, seconds = float(parts[0]), float(parts[1]), float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    df['time_seconds'] = df['Timecode'].apply(timecode_to_seconds)
    
    return df

def create_comprehensive_plots(df):
    """Create multiple visualization plots for threshold analysis"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Time series plot of all metrics
    ax1 = plt.subplot(3, 2, 1)
    metrics = ['content_val', 'delta_edges', 'delta_hue', 'delta_lum', 'delta_sat']
    for metric in metrics:
        plt.plot(df['time_seconds'], df[metric], label=metric, linewidth=2, alpha=0.8)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Detection Values')
    plt.title('All Detection Metrics Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Focus on content_val with potential thresholds
    ax2 = plt.subplot(3, 2, 2)
    plt.plot(df['time_seconds'], df['content_val'], 'b-', linewidth=2, label='content_val')
    
    # Add threshold lines
    content_mean = df['content_val'].mean()
    content_std = df['content_val'].std()
    thresholds = [
        content_mean + content_std,
        content_mean + 2*content_std,
        content_mean + 1.5*content_std
    ]
    
    colors = ['red', 'orange', 'green']
    labels = ['Mean + 1σ', 'Mean + 2σ', 'Mean + 1.5σ']
    
    for threshold, color, label in zip(thresholds, colors, labels):
        plt.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, label=f'{label}: {threshold:.2f}')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Content Value')
    plt.title('Content Value with Suggested Thresholds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Distribution analysis
    ax3 = plt.subplot(3, 2, 3)
    plt.hist(df['content_val'], bins=30, alpha=0.7, edgecolor='black')
    for threshold, color, label in zip(thresholds, colors, labels):
        plt.axvline(x=threshold, color=color, linestyle='--', linewidth=2, label=f'{label}: {threshold:.2f}')
    plt.xlabel('Content Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Content Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Correlation heatmap
    ax4 = plt.subplot(3, 2, 4)
    correlation_matrix = df[metrics].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Between Detection Metrics')
    
    # 5. Box plot comparison
    ax5 = plt.subplot(3, 2, 5)
    df_melted = pd.melt(df, id_vars=['Frame Number'], value_vars=metrics, 
                       var_name='Metric', value_name='Value')
    sns.boxplot(data=df_melted, x='Metric', y='Value')
    plt.xticks(rotation=45)
    plt.title('Distribution Comparison of All Metrics')
    plt.tight_layout()
    
    # 6. Derivative analysis for content_val
    ax6 = plt.subplot(3, 2, 6)
    content_derivative = np.gradient(df['content_val'])
    plt.plot(df['time_seconds'], content_derivative, 'purple', linewidth=2, label='Content Value Derivative')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Mark significant changes
    significant_changes = np.abs(content_derivative) > np.std(content_derivative) * 2
    plt.scatter(df['time_seconds'][significant_changes], 
               content_derivative[significant_changes], 
               color='red', s=50, alpha=0.7, label='Significant Changes')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Rate of Change')
    plt.title('Content Value Rate of Change (Potential Scene Cuts)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def interactive_threshold_explorer(df):
    """Create an interactive plot to explore different threshold values"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Initial threshold
    initial_threshold = df['content_val'].mean() + df['content_val'].std()
    
    # Plot 1: Time series with threshold
    line1, = ax1.plot(df['time_seconds'], df['content_val'], 'b-', linewidth=2, label='content_val')
    threshold_line, = ax1.plot(df['time_seconds'], [initial_threshold] * len(df), 
                              'r--', linewidth=2, label=f'Threshold: {initial_threshold:.2f}')
    
    # Mark detected scenes
    scenes = df['content_val'] > initial_threshold
    scene_points = ax1.scatter(df['time_seconds'][scenes], df['content_val'][scenes], 
                              color='red', s=50, alpha=0.7, label='Detected Scenes')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Content Value')
    ax1.set_title('Interactive Threshold Explorer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Statistics
    ax2.text(0.1, 0.8, f'Total Frames: {len(df)}', transform=ax2.transAxes, fontsize=12)
    ax2.text(0.1, 0.7, f'Detected Scenes: {np.sum(scenes)}', transform=ax2.transAxes, fontsize=12)
    ax2.text(0.1, 0.6, f'Detection Rate: {np.sum(scenes)/len(df)*100:.1f}%', transform=ax2.transAxes, fontsize=12)
    ax2.text(0.1, 0.5, f'Avg Time Between Scenes: {len(df)/np.sum(scenes)*0.04:.2f}s', 
             transform=ax2.transAxes, fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Add slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Threshold', 
                   df['content_val'].min(), df['content_val'].max(), 
                   valinit=initial_threshold, valstep=0.1)
    
    def update_threshold(val):
        threshold = slider.val
        threshold_line.set_ydata([threshold] * len(df))
        
        # Update scene detection
        scenes = df['content_val'] > threshold
        scene_points.set_offsets(np.column_stack([df['time_seconds'][scenes], df['content_val'][scenes]]))
        
        # Update statistics
        ax2.clear()
        ax2.text(0.1, 0.8, f'Total Frames: {len(df)}', transform=ax2.transAxes, fontsize=12)
        ax2.text(0.1, 0.7, f'Detected Scenes: {np.sum(scenes)}', transform=ax2.transAxes, fontsize=12)
        ax2.text(0.1, 0.6, f'Detection Rate: {np.sum(scenes)/len(df)*100:.1f}%', transform=ax2.transAxes, fontsize=12)
        if np.sum(scenes) > 0:
            ax2.text(0.1, 0.5, f'Avg Time Between Scenes: {len(df)/np.sum(scenes)*0.04:.2f}s', 
                     transform=ax2.transAxes, fontsize=12)
        ax2.text(0.1, 0.4, f'Current Threshold: {threshold:.2f}', transform=ax2.transAxes, fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        threshold_line.set_label(f'Threshold: {threshold:.2f}')
        ax1.legend()
        fig.canvas.draw()
    
    slider.on_changed(update_threshold)
    
    plt.tight_layout()
    return fig, slider

def generate_recommendations(df):
    """Generate threshold recommendations based on statistical analysis"""
    content_vals = df['content_val']
    
    # Statistical measures
    mean_val = content_vals.mean()
    std_val = content_vals.std()
    median_val = content_vals.median()
    q75 = content_vals.quantile(0.75)
    q90 = content_vals.quantile(0.90)
    q95 = content_vals.quantile(0.95)
    
    print("=== THRESHOLD RECOMMENDATIONS ===\n")
    
    print(f"Statistical Summary:")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Std Dev: {std_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  75th Percentile: {q75:.2f}")
    print(f"  90th Percentile: {q90:.2f}")
    print(f"  95th Percentile: {q95:.2f}\n")
    
    recommendations = [
        ("Conservative (fewer scenes)", q95, "Detects only the most significant changes"),
        ("Moderate", mean_val + 1.5*std_val, "Balanced approach, good starting point"),
        ("Sensitive", q75, "Detects more subtle changes"),
        ("Very Sensitive", median_val, "May catch too many minor changes")
    ]
    
    print("Recommended Thresholds:")
    for name, threshold, description in recommendations:
        scenes_detected = np.sum(content_vals > threshold)
        detection_rate = scenes_detected / len(df) * 100
        print(f"  {name}: {threshold:.2f}")
        print(f"    - {description}")
        print(f"    - Would detect {scenes_detected} scenes ({detection_rate:.1f}% of frames)")
        if scenes_detected > 0:
            avg_time_between = len(df) / scenes_detected * 0.04  # assuming 25fps
            print(f"    - Average time between scenes: {avg_time_between:.2f}s")
        print()

# Example usage:
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file = "/Users/spectatr/Downloads/general_scripts/model_improvement_experiments/stats.csv"
    
    # For demonstration, create sample data matching your format
    sample_data = {
        'Frame Number': [2, 3, 4, 5, 6],
        'Timecode': ['00:00:00.040', '00:00:00.080', '00:00:00.120', '00:00:00.160', '00:00:00.200'],
        'content_val': [5.787172670717593, 5.467077184606481, 6.017731843171297, 6.593442563657407, 6.533817997685186],
        'delta_edges': [7.532958984375, 7.67822265625, 7.59521484375, 9.061686197916666, 9.559733072916666],
        'delta_hue': [4.982503255208333, 4.397271050347222, 4.756483289930555, 4.802815755208333, 4.711371527777778],
        'delta_lum': [5.439073350694445, 5.376736111111111, 5.889512803819445, 6.590955946180555, 6.493543836805555],
        'delta_sat': [6.93994140625, 6.627224392361111, 7.407199435763889, 8.386555989583334, 8.396538628472221]
    }
    
    df = pd.DataFrame(sample_data)
    df['time_seconds'] = df['Frame Number'] * 0.04  # Assuming 25fps
    
    # Create visualizations
    fig1 = create_comprehensive_plots(df)
    plt.show()
    
    # Create interactive explorer
    fig2, slider = interactive_threshold_explorer(df)
    plt.show()
    
    # Generate recommendations
    generate_recommendations(df)
    
    print("\n=== USAGE INSTRUCTIONS ===")
    print("1. Replace the sample data with your actual CSV file")
    print("2. Use the comprehensive plots to understand your data patterns")
    print("3. Use the interactive explorer to test different thresholds")
    print("4. Consider the recommendations based on your specific needs")
    print("5. For football matches, start with the 'Moderate' recommendation")
    print("6. Adjust based on whether you're getting too many or too few scene cuts")