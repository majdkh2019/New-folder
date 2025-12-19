# ☕ Maven Roasters Coffee Shop Dashboard

A comprehensive, interactive dashboard for analyzing coffee shop sales data with AI-powered predictions and insights. Built with Streamlit, this application provides real-time analytics, forecasting, and business intelligence tools.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Dashboard Sections](#dashboard-sections)
- [AI Features](#ai-features)
- [Data Requirements](#data-requirements)
- [Authors](#authors)

## ✨ Features

### 📊 Main Dashboard
- **Net Sales Analysis**: Monthly sales visualization with store comparison
- **Hourly Transaction Patterns**: Average transaction volumes by hour of day
- **Cup Size Analysis**: Product size preferences across categories
- **Interactive Filtering**: Filter by store location and product category
- **Real-time Updates**: Dynamic charts that update based on selected filters

### 🤖 AI Predictions & Insights
- **Future Revenue Forecasting**: Polynomial regression-based sales predictions
- **Product Launch Simulator**: Predict impact of new products/sizes
- **Store Efficiency Analysis**: AI-predicted store sizes and efficiency metrics
- **Geographic Context**: Store-specific insights based on foot traffic and office density

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd coffeproject2/New\ folder
```

### Step 2: Install Dependencies
```bash
pip install streamlit pandas numpy plotly scikit-learn matplotlib
```

Or use a requirements file:
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Data
Ensure you have a CSV file named `Coffee Shop Sales.csv` in the project directory with the following columns:
- `transaction_id`
- `transaction_date` (format: DD/MM/YYYY)
- `transaction_time`
- `transaction_qty`
- `store_id`
- `store_location`
- `product_id`
- `unit_price`
- `product_category`
- `product_type`
- `product_detail`

## 💻 Usage

### Running the Dashboard
```bash
streamlit run app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

### Navigation
- Use the sidebar to switch between **Main Dashboard** and **AI Predictions**
- Filter data by **Store Location** and **Product Category** using the sidebar controls
- Hover over charts for detailed tooltips
- Use chart controls (zoom, pan, reset) for better exploration

## 📁 Project Structure

```
.
├── app.py                      # Main Streamlit application
├── Coffee Shop Sales.csv       # Sales data (required)
├── README.md                   # This file
├── final.ipynb                 # Jupyter notebook (optional)
└── app2.ipynb                  # Additional notebook (optional)
```

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning models (Linear Regression, Random Forest, Polynomial Features)
- **Matplotlib**: Static plotting support
- **NumPy**: Numerical computations

## 📈 Dashboard Sections

### 1. Net Sales by Month
- Bar chart showing monthly net sales per store
- Line graph overlay showing total purchased items
- Last 6 months of data displayed
- Supports filtering by store location

### 2. Hourly Transactions
- Average transaction volume per hour
- Standard deviation indicators
- Based on last 6 months of data
- Helps identify peak business hours

### 3. Cup Sizes per Category
- Stacked horizontal bar chart
- Shows Small, Medium, and Large size preferences
- Sorted by total quantity (ascending)
- Filterable by store and category

## 🤖 AI Features

### Future Revenue Prediction
- Uses polynomial regression with seasonal factors
- Considers temperature and tourist index
- Predicts up to 12 months ahead
- Store-specific forecasts

### Product Launch Simulator
- Random Forest model for quantity prediction
- Factors in:
  - Product category
  - Unit price
  - Store foot traffic
  - Office density
- Estimates monthly units and revenue

### Store Efficiency Analysis
- AI-predicted store sizes based on:
  - Total transaction volume
  - Bakery product ratio
- Calculates sales per square foot
- Compares efficiency across locations

## 📊 Data Requirements

The CSV file should contain transaction-level data with:
- **Date Format**: DD/MM/YYYY (European format)
- **Time Format**: HH:MM:SS
- **Required Columns**: All columns listed in Installation section
- **Data Quality**: No missing critical values (date, quantity, price)

## 🎨 UI Features

- **Dark Theme Support**: Automatically adapts to Streamlit theme
- **Glowing Selectboxes**: Interactive hover effects
- **Icon-based Navigation**: Visual store and category icons
- **Responsive Layout**: Wide layout optimized for data visualization
- **Custom Styling**: Modern, professional appearance

## 📝 Notes

- The application uses `dayfirst=True` for date parsing to support European date formats
- AI models are trained on-the-fly when needed
- All calculations are cached for performance
- The dashboard automatically handles missing data

## 👥 Authors

- **Hamza Tahboub**
- **Majd Igbarea**
- **Marysol Karwan**
- **Igor Kornev**

## 📄 License

This project is created for educational and business analysis purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Support

For questions or issues, please open an issue in the repository.

---

**Built with ❤️ using Streamlit**

