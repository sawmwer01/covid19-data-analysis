"""
COVID-19 Data Analysis System
Author: [Your Name]
Date: November 2025
Description: A comprehensive system for analyzing COVID-19 global data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class DataFetcher:
    """Handles fetching data from open data sources."""
    
    def __init__(self, data_url: str):
        """
        Initialize DataFetcher with data source URL.
        
        Args:
            data_url (str): URL of the COVID-19 data source
        """
        self.data_url = data_url
        self.raw_data = None
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch COVID-19 data from the specified URL.
        
        Returns:
            pd.DataFrame: Raw COVID-19 data
        """
        try:
            print("📥 Fetching data from source...")
            # Using pandas read_csv with proper headers
            self.raw_data = pd.read_csv(
                self.data_url,
                on_bad_lines='skip',
                low_memory=False
            )
            print(f"✓ Successfully fetched {len(self.raw_data)} records")
            print(f"✓ Columns available: {len(self.raw_data.columns)}")
            return self.raw_data
        except Exception as e:
            print(f"✗ Error fetching data: {e}")
            print("🔄 Attempting alternative method...")
            return self._fetch_alternative()
    
    def _fetch_alternative(self) -> pd.DataFrame:
        """
        Alternative method to fetch data using requests.
        
        Returns:
            pd.DataFrame: Raw COVID-19 data
        """
        try:
            import requests
            from io import StringIO
            
            response = requests.get(self.data_url, timeout=30)
            response.raise_for_status()
            
            self.raw_data = pd.read_csv(
                StringIO(response.text),
                on_bad_lines='skip',
                low_memory=False
            )
            print(f"✓ Alternative method successful: {len(self.raw_data)} records")
            return self.raw_data
        except Exception as e:
            print(f"✗ Alternative method failed: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample data for demonstration if fetch fails.
        
        Returns:
            pd.DataFrame: Sample COVID-19 data
        """
        print("⚠️  Using sample data for demonstration...")
        
        import numpy as np
        from datetime import datetime, timedelta
        
        countries = ['United States', 'India', 'Brazil', 'United Kingdom', 
                    'France', 'Germany', 'Italy', 'Spain', 'Japan', 
                    'South Korea', 'Canada', 'Australia', 'Mexico',
                    'Poland', 'Netherlands']
        
        dates = pd.date_range(start='2020-01-01', end='2024-11-30', freq='D')
        
        data_list = []
        for country in countries:
            base_cases = np.random.randint(100000, 10000000)
            base_deaths = int(base_cases * np.random.uniform(0.01, 0.03))
            population = np.random.randint(10000000, 350000000)
            
            for i, date in enumerate(dates):
                growth_factor = 1 + (i / len(dates)) * np.random.uniform(5, 15)
                cases = int(base_cases * growth_factor)
                deaths = int(base_deaths * growth_factor)
                
                data_list.append({
                    'location': country,
                    'date': date.strftime('%Y-%m-%d'),
                    'total_cases': cases,
                    'total_deaths': deaths,
                    'new_cases': int(cases * 0.001),
                    'new_deaths': int(deaths * 0.001),
                    'population': population
                })
        
        self.raw_data = pd.DataFrame(data_list)
        print(f"✓ Sample data created: {len(self.raw_data)} records")
        return self.raw_data


class DataProcessor:
    """Processes and cleans COVID-19 data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize DataProcessor with raw data.
        
        Args:
            data (pd.DataFrame): Raw COVID-19 data
        """
        self.data = data
        self.processed_data = None
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        print("\n🔧 Cleaning data...")
        
        # Identify required columns
        required_cols = ['location', 'date', 'total_cases']
        
        # Check which columns exist
        missing_cols = [col for col in required_cols 
                       if col not in self.data.columns]
        
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")
            print(f"Available columns: {list(self.data.columns)[:10]}")
        
        # Remove rows with missing critical values
        available_cols = [col for col in required_cols 
                         if col in self.data.columns]
        self.processed_data = self.data.dropna(subset=available_cols)
        
        # Convert date to datetime
        if 'date' in self.processed_data.columns:
            self.processed_data['date'] = pd.to_datetime(
                self.processed_data['date'], errors='coerce'
            )
        
        # Fill numeric NaN values with 0
        numeric_columns = self.processed_data.select_dtypes(
            include=['float64', 'int64']
        ).columns
        self.processed_data[numeric_columns] = (
            self.processed_data[numeric_columns].fillna(0)
        )
        
        print(f"✓ Data cleaned: {len(self.processed_data)} valid records")
        return self.processed_data
    
    def get_latest_data(self) -> pd.DataFrame:
        """
        Get the most recent data for each country.
        
        Returns:
            pd.DataFrame: Latest data per country
        """
        if 'date' in self.processed_data.columns:
            latest = self.processed_data.sort_values('date').groupby(
                'location'
            ).last().reset_index()
        else:
            latest = self.processed_data.groupby(
                'location'
            ).last().reset_index()
        return latest
    
    def get_top_countries(self, n: int = 10, 
                         metric: str = 'total_cases') -> pd.DataFrame:
        """
        Get top N countries by specified metric.
        
        Args:
            n (int): Number of top countries to return
            metric (str): Metric to sort by
            
        Returns:
            pd.DataFrame: Top N countries
        """
        latest = self.get_latest_data()
        
        if metric in latest.columns:
            top_countries = latest.nlargest(n, metric)
        else:
            print(f"⚠️  Metric '{metric}' not found, using first numeric column")
            numeric_cols = latest.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                top_countries = latest.nlargest(n, numeric_cols[0])
            else:
                top_countries = latest.head(n)
        
        return top_countries


class DataAnalyzer:
    """Performs statistical analysis on COVID-19 data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize DataAnalyzer with processed data.
        
        Args:
            data (pd.DataFrame): Processed COVID-19 data
        """
        self.data = data
    
    def generate_summary_statistics(self) -> Dict:
        """
        Generate summary statistics for the dataset.
        
        Returns:
            Dict: Dictionary containing summary statistics
        """
        print("\n📊 Generating summary statistics...")
        
        if 'date' in self.data.columns:
            latest_data = self.data.sort_values('date').groupby(
                'location'
            ).last()
        else:
            latest_data = self.data.groupby('location').last()
        
        stats = {
            'total_countries': len(latest_data),
            'total_cases_global': latest_data['total_cases'].sum() if 'total_cases' in latest_data.columns else 0,
            'total_deaths_global': latest_data['total_deaths'].sum() if 'total_deaths' in latest_data.columns else 0,
            'avg_cases_per_country': latest_data['total_cases'].mean() if 'total_cases' in latest_data.columns else 0,
            'max_cases_country': latest_data['total_cases'].idxmax() if 'total_cases' in latest_data.columns else 'N/A',
            'max_cases_value': latest_data['total_cases'].max() if 'total_cases' in latest_data.columns else 0
        }
        
        return stats
    
    def calculate_mortality_rate(self, country: str) -> float:
        """
        Calculate mortality rate for a specific country.
        
        Args:
            country (str): Country name
            
        Returns:
            float: Mortality rate as percentage
        """
        country_data = self.data[self.data['location'] == country]
        
        if len(country_data) == 0:
            return 0.0
        
        if 'date' in country_data.columns:
            latest = country_data.sort_values('date').iloc[-1]
        else:
            latest = country_data.iloc[-1]
        
        if 'total_cases' in latest and 'total_deaths' in latest:
            if latest['total_cases'] > 0:
                rate = (latest['total_deaths'] / latest['total_cases']) * 100
                return round(rate, 2)
        
        return 0.0


class DataVisualizer:
    """Creates visualizations for COVID-19 data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize DataVisualizer with processed data.
        
        Args:
            data (pd.DataFrame): Processed COVID-19 data
        """
        self.data = data
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
    
    def plot_top_countries_bar(self, top_countries: pd.DataFrame, 
                                metric: str = 'total_cases'):
        """
        Create bar chart of top countries.
        
        Args:
            top_countries (pd.DataFrame): Data for top countries
            metric (str): Metric to visualize
        """
        if metric not in top_countries.columns:
            print(f"⚠️  Metric '{metric}' not available for visualization")
            return
        
        plt.figure(figsize=(12, 6))
        colors = plt.cm.viridis(range(len(top_countries)))
        
        plt.barh(top_countries['location'], top_countries[metric], 
                 color=colors)
        plt.xlabel(metric.replace('_', ' ').title(), 
                  fontsize=12, fontweight='bold')
        plt.ylabel('Country', fontsize=12, fontweight='bold')
        plt.title(f'Top 10 Countries by {metric.replace("_", " ").title()}',
                  fontsize=14, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(top_countries[metric]):
            plt.text(v, i, f' {v:,.0f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        print(f"✓ Bar chart created for {metric}")
    
    def plot_time_series(self, countries: List[str], 
                         metric: str = 'total_cases'):
        """
        Create time series plot for specified countries.
        
        Args:
            countries (List[str]): List of country names
            metric (str): Metric to visualize
        """
        if 'date' not in self.data.columns or metric not in self.data.columns:
            print(f"⚠️  Cannot create time series: missing required columns")
            return
        
        plt.figure(figsize=(14, 7))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, country in enumerate(countries):
            country_data = self.data[self.data['location'] == country]
            
            if len(country_data) == 0:
                continue
            
            country_data = country_data.sort_values('date')
            color = colors[idx % len(colors)]
            
            plt.plot(country_data['date'], country_data[metric], 
                    label=country, linewidth=2.5, color=color, alpha=0.8)
        
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel(metric.replace('_', ' ').title(), 
                  fontsize=12, fontweight='bold')
        plt.title(f'{metric.replace("_", " ").title()} Over Time',
                  fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        print(f"✓ Time series chart created for {metric}")
    
    def plot_correlation_heatmap(self, latest_data: pd.DataFrame):
        """
        Create correlation heatmap for numeric variables.
        
        Args:
            latest_data (pd.DataFrame): Latest data for each country
        """
        numeric_cols = ['total_cases', 'total_deaths', 'population']
        available_cols = [col for col in numeric_cols 
                         if col in latest_data.columns]
        
        if len(available_cols) < 2:
            print("⚠️  Not enough numeric columns for correlation analysis")
            return
        
        correlation = latest_data[available_cols].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=2,
                   fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Correlation Heatmap', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        print("✓ Correlation heatmap created")


class COVIDAnalysisSystem:
    """Main system orchestrating the entire analysis pipeline."""
    
    def __init__(self, data_url: str):
        """
        Initialize the COVID-19 Analysis System.
        
        Args:
            data_url (str): URL of the COVID-19 data source
        """
        self.data_url = data_url
        self.fetcher = None
        self.processor = None
        self.analyzer = None
        self.visualizer = None
        self.processed_data = None
    
    def run_analysis(self):
        """Execute the complete analysis pipeline."""
        print("=" * 60)
        print("     COVID-19 DATA ANALYSIS SYSTEM")
        print("=" * 60)
        
        # Step 1: Fetch data
        self.fetcher = DataFetcher(self.data_url)
        raw_data = self.fetcher.fetch_data()
        
        if raw_data is None or len(raw_data) == 0:
            print("✗ Analysis terminated: No data available")
            return
        
        # Step 2: Process data
        self.processor = DataProcessor(raw_data)
        self.processed_data = self.processor.clean_data()
        
        # Step 3: Analyze data
        self.analyzer = DataAnalyzer(self.processed_data)
        stats = self.analyzer.generate_summary_statistics()
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("     SUMMARY STATISTICS")
        print("=" * 60)
        print(f"📍 Total Countries Analyzed: {stats['total_countries']}")
        print(f"🦠 Total Global Cases: {stats['total_cases_global']:,.0f}")
        print(f"💀 Total Global Deaths: {stats['total_deaths_global']:,.0f}")
        print(f"📊 Average Cases per Country: {stats['avg_cases_per_country']:,.0f}")
        print(f"🏆 Country with Most Cases: {stats['max_cases_country']}")
        print(f"📈 Maximum Cases: {stats['max_cases_value']:,.0f}")
        
        # Step 4: Visualize data
        print("\n" + "=" * 60)
        print("📈 GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        self.visualizer = DataVisualizer(self.processed_data)
        
        # Get top countries
        top_countries = self.processor.get_top_countries(10, 'total_cases')
        
        # Create visualizations
        print("\n1️⃣  Creating bar chart...")
        self.visualizer.plot_top_countries_bar(top_countries, 'total_cases')
        
        # Time series for selected countries
        print("\n2️⃣  Creating time series plot...")
        available_countries = self.processed_data['location'].unique()[:5]
        self.visualizer.plot_time_series(list(available_countries), 
                                        'total_cases')
        
        # Correlation heatmap
        print("\n3️⃣  Creating correlation heatmap...")
        latest_data = self.processor.get_latest_data()
        self.visualizer.plot_correlation_heatmap(latest_data)
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)


# Main execution
if __name__ == "__main__":
    # Primary data source
    DATA_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
    
    # Create and run the analysis system
    system = COVIDAnalysisSystem(DATA_URL)
    system.run_analysis()
