"""
Data visualization module for financial time series analysis.
Provides functionality for creating various plots and visualizations of financial data,
including NaN distributions, correlation matrices, and prediction results.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
class Plotter():
    """
    A class for creating and managing multiple types of financial data visualizations.
    
    Handles the creation, storage, and output of matplotlib plots for financial analysis,
    including data quality visualization, correlation analysis, and prediction comparisons.
    
    Args:
        demo (bool): Flag indicating if running in demo mode, affecting plot output behavior
    """
    def __init__(self, demo):
        self.plot_list=[]
        self.demo = demo

    def giga_plotter_function(self):
        """
        Processes and outputs all stored plots based on demo mode setting.
        
        In demo mode, saves plots to files with timestamps.
        In regular mode, displays plots interactively.
        Understanding this function is an exercise for the reader
        """
        for plot in self.plot_list:
            if self.demo:
                sauce_path = "images/" + str(time.time()) + ".png"
                plot.savefig(sauce_path)   # save the figure to file
                plot.close()
            else:
                plot.show()

    def dataframe_to_nan_image(self, df1, df2):
        """
        Creates visual representations of NaN distributions in two dataframes.
        
        Generates heatmap-style visualizations showing the presence of NaN values,
        with red indicating NaN and green indicating valid data.
        
        Args:
            df1 (pd.DataFrame): First dataframe to visualize
            df2 (pd.DataFrame): Second dataframe to visualize
        """
        # Create a boolean mask: True for NaN, False for non-NaN
        plt.figure(figsize=(15, 7))
        for idx, df in enumerate([df1, df2], 1):
            # Create subplot - 1 row, 2 columns, plot number idx
            plt.subplot(1, 2, idx)

            height, width = df.shape
            nan_mask = df.isna().to_numpy()  # Convert to numpy array explicitly

            # Create an RGB array with the same shape as the DataFrame
            img = np.zeros((height, width, 3), dtype=np.uint8)

            # Set red for NaN and green for non-NaN
            img[nan_mask] = [255, 0, 0]      # Red for NaN
            img[~nan_mask] = [0, 255, 0]     # Green for non-NaN

            # Display the image
            plt.imshow(img, aspect='auto')

            # Show axis labels but hide tick values
            ax = plt.gca()
            ax.set_xticklabels([])  # Hide x-axis tick values
            ax.set_yticklabels([])  # Hide y-axis tick values
            ax.set_xticks([])       # Hide x-axis ticks
            ax.set_yticks([])       # Hide y-axis ticks

            # Add axis labels and title with more information
            plt.xlabel('Tickers')
            plt.ylabel('Timestamps')
            plt.title(f'Dataset {idx}: {height}x{width} matrix\n'
                    f'NaN: {nan_mask.sum()} ({(nan_mask.sum() / (height * width) * 100):.1f}%)')
            self.plot_list.append(plt)


        plt.figure(figsize=(19, 15))
        plt.matshow(df2.corr())
        plt.xticks(range(df.select_dtypes(['number']).shape[1]), df2.select_dtypes(['number']).columns, fontsize=14, rotation=45)
        plt.yticks(range(df.select_dtypes(['number']).shape[1]), df2.select_dtypes(['number']).columns, fontsize=14)
        plt.axis('off')
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16)
        self.plot_list.append(plt)

    def plot_data_processing_pipeline(self, raw_df, processed_df, model_data, target_column):
        """
        Creates comprehensive visualizations of the data processing pipeline results.
        
        Generates multiple plots showing original time series, distributions,
        training progress, and PCA analysis results.
        
        Args:
            raw_df (pd.DataFrame): Original unprocessed data
            processed_df (pd.DataFrame): Processed and cleaned data
            model_data (dict): Dictionary containing model training results and metrics
            target_column (str): Name of the target variable column
        """
        fig1 = plt.figure(figsize=(15, 6))

        # Target Time Series
        ax1 = fig1.add_subplot(121)
        processed_df[target_column].plot(ax=ax1)
        ax1.set_title('Target Time Series')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True)

        # Target Distribution
        ax2 = fig1.add_subplot(122)
        processed_df[target_column].hist(bins=50, ax=ax2)
        ax2.set_title('Target Distribution')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True)

        plt.tight_layout()
        self.plot_list.append(plt)

        # Figure 2: Training Loss Analysis
        if 'epoch_losses' in model_data and model_data['epoch_losses']:
            fig2 = plt.figure(figsize=(15, 6))

            # Linear scale
            ax3 = fig2.add_subplot(121)
            epochs = range(1, len(model_data['epoch_losses']) + 1)
            ax3.plot(epochs, model_data['epoch_losses'], 'b-', label='Training Loss')
            ax3.set_title('Training Loss (Linear Scale)')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss')
            ax3.grid(True)
            ax3.legend()

            # Logarithmic scale
            ax4 = fig2.add_subplot(122)
            ax4.plot(epochs, model_data['epoch_losses'], 'r-', label='Training Loss')
            ax4.set_title('Training Loss (Log Scale)')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss')
            ax4.set_yscale('log')
            ax4.grid(True)
            ax4.legend()

            plt.tight_layout()
            self.plot_list.append(plt)

        # Figure 3: PCA Analysis
        if 'pca_components' in model_data and model_data['pca_components'] is not None:
            fig3 = plt.figure(figsize=(15, 6))

            # Cumulative explained variance
            ax5 = fig3.add_subplot(121)
            variance_ratio = model_data['pca_components'].explained_variance_ratio_
            cumulative_variance = np.cumsum(variance_ratio)
            ax5.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
            ax5.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
            ax5.set_title('Cumulative Explained Variance')
            ax5.set_xlabel('Number of Components')
            ax5.set_ylabel('Cumulative Explained Variance')
            ax5.grid(True)
            ax5.legend()

            # Individual component contributions
            ax6 = fig3.add_subplot(122)
            individual_variance = variance_ratio * 100
            positions = range(1, len(individual_variance) + 1)
            ax6.bar(positions, individual_variance)
            ax6.set_xscale('log')
            ax6.set_title('Individual Component Contributions (Log Scale)')
            ax6.set_xlabel('Principal Component (log scale)')
            ax6.set_ylabel('Explained Variance (%)')
            ax6.grid(True, which='both', axis='y')
            # Add minor grid lines for log scale
            ax6.grid(True, which='minor', axis='x', alpha=0.2)
            # Ensure x-axis ticks are at actual component numbers
            ax6.set_xticks(positions)
            ax6.set_xticklabels([str(i) for i in positions])

            plt.tight_layout()
            self.plot_list.append(plt)


    def plot_predictions(self, test_dates, real_values, one_by_one_predictions, all_at_once_predictions, target_column):
        """
        Creates comparison plots between actual values and different prediction strategies.
        
        Generates multiple subplots comparing actual values with one-by-one and
        all-at-once prediction approaches, including trading simulation results.
        
        Args:
            test_dates (array-like): Dates for the test period
            real_values (array-like): Actual values
            one_by_one_predictions (array-like): Predictions made one at a time
            all_at_once_predictions (array-like): Predictions made all at once
            target_column (str): Name of the target variable
        """
        plt.figure, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(10, 10))
        #plt.subplots_adjust(hspace=5)
        ax0.plot(test_dates, real_values,
                label='Actual Values', color='black', linewidth=2)
        ax0.plot(test_dates, all_at_once_predictions,
                label='All-at-once Predictions', color='blue', linestyle='--')
        ax0.plot(test_dates, one_by_one_predictions,
                label='One-by-one Predictions', color='red', linestyle=':')
        rot = 30
        ax0.set_title('Actual vs Predicted Values - Comparison of Prediction Strategies')
        ax0.set_xlabel('Date')
        ax0.set_ylabel(target_column)
        ax0.legend()
        ax0.grid(True, alpha=0.3)
        ax0.tick_params(axis='x', rotation=rot)

        # Simulate trading for each strategy
        real_history = simulate_trading(real_values)
        one_by_one_history = simulate_trading(one_by_one_predictions)
        all_at_once_history = simulate_trading(all_at_once_predictions)

        # Extract metrics for plotting
        real_metrics = extract_metrics(real_history)
        one_by_one_metrics = extract_metrics(one_by_one_history)
        all_at_once_metrics = extract_metrics(all_at_once_history)

        # Create three subplots


        # Plot 1: Total Portfolio Value
        ax1.plot(test_dates, real_metrics['total'], 'k-', label='Actual Values Strategy', linewidth=2)
        ax1.plot(test_dates, all_at_once_metrics['total'], 'b--', label='All-at-once Strategy', linewidth=2)
        ax1.plot(test_dates, one_by_one_metrics['total'], 'r:', label='One-by-one Strategy', linewidth=2)
        ax1.set_title('Total Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Total Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=rot)

        # Plot 2: Cash Holdings
        ax2.plot(test_dates, real_metrics['money'], 'k-', label='Actual Values Cash', linewidth=2)
        ax2.plot(test_dates, all_at_once_metrics['money'], 'b--', label='All-at-once Cash', linewidth=2)
        ax2.plot(test_dates, one_by_one_metrics['money'], 'r:', label='One-by-one Cash', linewidth=2)
        ax2.set_title('Cash Holdings Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cash Amount')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=rot)

        # Plot 3: Stock Holdings
        ax3.plot(test_dates, real_metrics['stock'], 'k-', label='Actual Values Stock', linewidth=2)
        ax3.plot(test_dates, all_at_once_metrics['stock'], 'b--', label='All-at-once Stock', linewidth=2)
        ax3.plot(test_dates, one_by_one_metrics['stock'], 'r:', label='One-by-one Stock', linewidth=2)
        ax3.set_title('Stock Holdings Over Time')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Stock Amount')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=rot)
        plt.tight_layout(pad=2.0, h_pad=4.0, w_pad=2.0)
        #plt.tight_layout()
        self.plot_list.append(plt)








def simulate_trading(prices):
    """ """
    money = 1.0  # Start with 1 unit of money
    stock = 1.0  # Start with no stock
    history = [{'money': money, 'stock': stock, 'total': money + stock}]

    for i in range(len(prices) - 1):
        current_price = prices[i]
        next_price = prices[i + 1]

        if next_price > current_price and money >= 0.1:  # Buy signal
            purchase_amount = 0.1
            money -= purchase_amount
            stock += purchase_amount / current_price * current_price
        elif next_price < current_price and stock > 0:  # Sell signal
            sale_amount = min(0.1, stock)
            stock -= sale_amount
            money += sale_amount * current_price

        history.append({
            'money': money,
            'stock': stock,
            'total': money + (stock * current_price)
        })

    return history

# Function to extract specific metrics from trading history
def extract_metrics(history):
    """ x"""
    return {
        'money': [h['money'] for h in history],
        'stock': [h['stock'] for h in history],
        'total': [h['total'] for h in history]
    }
    
def calculate_trading_metrics(metrics):
    """Calculate additional trading performance metrics"""
    results = {}
    for name, data in metrics.items():
        # Calculate returns
        returns = np.diff(data['total']) / np.array(data['total'][:-1])
        
        # Final return
        final_return = (data['total'][-1] - data['total'][0]) / data['total'][0]
        
        # Maximum drawdown
        peak = data['total'][0]
        max_drawdown = 0
        for value in data['total']:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 0 else 0
        
        results[name] = {
            'Final Return': final_return,
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio
        }
    
    return results