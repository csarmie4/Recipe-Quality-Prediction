import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

def plot_correlation_heatmap(df):
    """Plot the correlation heatmap for numeric features."""
    numeric_data = df.select_dtypes(include='number')
    sns.heatmap(numeric_data.corr(), annot=True)
    plt.show()

def plot_high_score_distribution(df):
    """Visualize the distribution of the target variable."""
    sns.countplot(x=df['HighScore'])
    plt.show()

def box_dist_plots(data):
    """Make box plots and distribution plots."""
    mean = data.mean()
    median = data.median()
    fig, (ax1, ax2) = plt.subplots(2, 1)
    sns.boxplot(data=data, ax=ax1)
    sns.histplot(data, kde=True, ax=ax2)
    ax2.axvline(mean, color='r', linestyle='--', linewidth=2)
    ax2.axvline(median, color='g', linestyle='-', linewidth=2)
    plt.legend({'Mean': mean, 'Median': median})
    plt.show()

def qqplot(data):
    """Generate a QQ plot for normality check."""
    sm.qqplot(data, line='s')
    plt.title("QQ Plot")
    plt.show()
