import os
import shutil
import ssl
import textwrap
from datetime import datetime
from io import StringIO
from statistics import median
import PyPDF2
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from financetoolkit import Toolkit
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.platypus import SimpleDocTemplate, Table, Spacer

good_pe = False

# Add industries
# Add historic values (?)

api_keys = [
    os.getenv('TOOLKIT_API_1'),
    os.getenv('TOOLKIT_API_2')
]


# HERE WE HAVE PDF-RELATED FUNCTIONS
def clean_reports_directory(directory='reports'):
    # Check if directory exists
    if os.path.exists(directory):
        # Remove all files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        # Create the directory if it doesn't exist
        os.makedirs(directory)


def merge_pdfs(report_folder, output_filename, stocks):
    title_page_path = os.path.join(report_folder, "0_title_page.pdf")
    c = canvas.Canvas(title_page_path, pagesize=letter)
    width, height = letter
    c.setFont("Times-Bold", 22)

    # Base title text without the stock names
    base_title_text = ("Financial report of"
                       "\nas of {}"
                       "\n\nProvided by G&M Inc.").format(datetime.now().strftime('%Y-%m-%d'))

    # Prepare the stock names as a comma-separated string
    stocks_string = ', '.join(stocks)

    # Use textwrap to wrap the stock names to a specified width
    wrapper = textwrap.TextWrapper(width=40)  # Adjust the width to fit your needs
    wrapped_stocks = wrapper.fill(text=stocks_string)

    # Combine the base title text with the wrapped stock names
    title_text = base_title_text.replace("of\n", "of\n{}\n".format(wrapped_stocks))

    y_position = height - 300  # Start position for the title
    line_height = 29  # Height of each line

    # Split the title text into lines and draw each line
    lines = title_text.split("\n")
    for line in lines:
        c.drawCentredString(width / 2.0, y_position, line)
        y_position -= line_height

    c.save()

    pdf_files = [f for f in os.listdir(report_folder) if f.endswith('.pdf')]
    pdf_writer = PyPDF2.PdfWriter()

    for filename in sorted(pdf_files):
        filepath = os.path.join(report_folder, filename)
        pdf_reader = PyPDF2.PdfReader(filepath)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_writer.add_page(page)

    with open(os.path.join(report_folder, output_filename), 'wb') as out:
        pdf_writer.write(out)

    print(f"\nMerged PDF saved as {output_filename} in {report_folder}.")


def generate_pdf_report(stock, analysis_results, file_name):
    c = canvas.Canvas(os.path.join('reports', file_name), pagesize=letter)
    width, height = letter
    current_date = datetime.now().strftime("%Y-%m-%d")

    c.setFont('Times-Roman', 16)
    c.drawCentredString(width / 2, height - 50, f"Finance Analysis Report for {stock} as of {current_date}")

    y_position = height - 100
    c.setFont('Times-Bold', 12)

    for model_name, model_results in analysis_results.items():
        c.drawString(40, y_position, model_name)
        y_position -= 20
        c.setFont('Times-Roman', 12)
        for key, value in model_results.items():
            c.drawString(60, y_position, f"{key}: {value}")
            y_position -= 20
        y_position -= 10  # Extra space between sections
        c.setFont('Times-Bold', 12)

    c.save()


def format_value_with_color(value, metric, stock_info):
    """Formats the value with color based on conditions specific to the metric."""
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']
    normal_style.alignment = TA_CENTER

    # Initialize default color as black
    color = "black"

    # Apply logic based on metric type
    if isinstance(metric, bool):  # Check if metric is a boolean value
        if metric:
            color = "green"
        else:
            color = "red"
    elif metric == "Payout Ratio":
        # Convert string value to float for comparison
        numeric_value = float(value) if value != "N/A" else None
        if numeric_value is not None:
            if numeric_value < 0.75:
                color = "green"
            else:
                color = "red"
    elif metric == "Forward P/E":
        company_pe = float(value) if value != "N/A" else None
        industry_pe = float(stock_info.get("Industry P/E", "N/A")) if stock_info.get("Industry P/E",
                                                                                     "N/A") != "N/A" else None
        if company_pe is not None and industry_pe is not None:
            if company_pe < industry_pe:
                color = "green"
            else:
                color = "red"
    elif metric == "Dividend Yield":
        dividend_yield = float(value) if value != "N/A" else None
        if dividend_yield is not None:
            if 0.01 < dividend_yield < 0.08:
                color = "green"
            else:
                color = "red"

    # Return formatted value with the determined color
    return Paragraph(f'<font color="{color}">{value}</font>', normal_style) if value != "N/A" else "N/A"


def create_financial_table_pdf(file_name, data):
    global good_pe
    doc = SimpleDocTemplate(os.path.join('reports', file_name), pagesize=letter)
    elements = []

    metrics = ["Industry", "Sector", "Payout Ratio", "Forward P/E", "Industry P/E", "EPS", "P/B", "Dividend Yield",
               "ROE", ]
    styles = getSampleStyleSheet()
    centered_style = styles['Normal']
    centered_style.alignment = TA_CENTER

    def create_table_chunk(chunk_data):
        table_data = [["Metric"]]
        tickers = [d["Ticker"] for d in chunk_data]
        top_row = [""] + tickers
        table_data[0] = top_row

        for metric in metrics:
            row = [metric]
            for item in chunk_data:
                value = "N/A"  # Default value

                if metric == "Industry P/E":
                    industry_pe = fetch_pe_ratios_by_industry(item.get("Industry", ""))
                    value = str(industry_pe) if industry_pe else "N/A"

                elif metric == "Forward P/E":
                    company_pe = item.get(metric, "N/A")
                    industry_pe = fetch_pe_ratios_by_industry(
                        item.get("industry", "")) if company_pe != "N/A" else "N/A"
                    if (company_pe != "N/A" and company_pe is not None and industry_pe != "N/A" and industry_pe is not
                            None):
                        if company_pe < industry_pe:
                            value = Paragraph(f'<font color="{"green"}">{round(company_pe, 2)}</font>', centered_style)
                            item[good_pe] = True
                            # value = format_value_with_color(round(company_pe, 2), (company_pe < industry_pe), None)
                    elif company_pe > industry_pe:
                        value = Paragraph(f'<font color="{"red"}">{round(company_pe, 2)}</font>', centered_style)
                        item[good_pe] = False
                    else:
                        value = Paragraph("N/A", centered_style)

                elif metric == "Dividend Yield":
                    dividend_yield = item.get(metric, "N/A")
                    if dividend_yield != "N/A" and dividend_yield is not None:
                        value = format_value_with_color(round(dividend_yield, 2),
                                                        0 < round(dividend_yield * 100, 2) < 8, None)
                    else:
                        value = "N/A"

                elif metric == "Payout Ratio":
                    payout_ratio = item.get(metric, "N/A")
                    if payout_ratio != "N/A" and payout_ratio is not None:
                        value = format_value_with_color(round(payout_ratio, 2), payout_ratio < 0.75, None)
                    else:
                        value = "N/A"

                else:
                    if metric in item and item[metric] is not None:
                        formatted_value = f"{item[metric]:.2f}" if isinstance(item[metric], float) else str(
                            item[metric])
                        value = Paragraph(formatted_value, centered_style)

                row.append(value)
            table_data.append(row)

        table = Table(table_data, style=[
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ])
        return table

    chunks = [data[i:i + 4] for i in range(0, len(data), 4)]
    for chunk in chunks:
        table = create_table_chunk(chunk)
        elements.append(table)
        elements.append(Spacer(1, 12))

    doc.build(elements)


def format_paragraph(text, color):
    """Utility function to format text with color."""
    styles = getSampleStyleSheet()
    style = styles['Normal']
    style.alignment = TA_CENTER
    return Paragraph(f'<font color="{color}">{text}</font>', style)


def create_summary_table_pdf(data, file_name):
    doc_path = os.path.join('reports', file_name)
    doc = SimpleDocTemplate(doc_path, pagesize=letter)
    elements = []

    # Define the metrics to be included in the table
    metrics = ["Ticker", "Payout Ratio", "Forward P/E", "Industry P/E", "Dividend Yield", "DGM Rec", "DCF Model Rec",
               "SDCF Rec"]

    # Prepare the styles
    styles = getSampleStyleSheet()
    centered_style = styles['Normal']
    centered_style.alignment = TA_CENTER

    # Prepare the table data starting with the metrics as headers
    table_data = [metrics]

    # Append data for each stock
    for stock_info in data:
        row = [stock_info.get("Ticker", "N/A")]
        for metric in metrics[1:]:  # Skip the first header (Ticker) since it's manually added
            # Check if the metric requires special formatting
            if metric == "Industry P/E":
                industry_pe = stock_info.get("Industry P/E", "")
                value = str(industry_pe) if industry_pe else "N/A"
                row.append(Paragraph(value, centered_style) if value != "N/A" else "N/A")

            elif metric == "Forward P/E":
                company_pe = stock_info.get(metric, "N/A")
                industry_pe = stock_info.get("Industry P/E", "")

                if company_pe != "N/A" and company_pe is not None and industry_pe != "N/A" and industry_pe is not None:
                    if company_pe < industry_pe:
                        value = Paragraph(f'<font color="{"green"}">{round(company_pe, 2)}</font>', centered_style)
                        stock_info[good_pe] = True
                    elif company_pe > industry_pe:
                        value = Paragraph(f'<font color="{"red"}">{round(company_pe, 2)}</font>', centered_style)
                        stock_info[good_pe] = False
                    else:
                        value = Paragraph("N/A", centered_style)
                else:
                    value = Paragraph("N/A", centered_style)
                row.append(value)

            elif metric == "Dividend Yield":
                dividend_yield = stock_info.get(metric, "N/A")
                if dividend_yield != "N/A" and dividend_yield is not None:
                    value = format_value_with_color(round(dividend_yield / 100, 2),
                                                    0 < round(dividend_yield / 100, 2) < 0.07, None)
                else:
                    value = "N/A"
                row.append(value)

            elif metric == "Payout Ratio":
                payout_ratio = stock_info.get(metric, "N/A")
                if payout_ratio != "N/A" and payout_ratio is not None:
                    value = format_value_with_color(round(payout_ratio, 2), payout_ratio < 75, None)
                else:
                    value = "N/A"
                row.append(value)

            else:
                # For recommendations or other string values
                recommendation = stock_info.get(metric, "N/A")
                recommendation_str = str(recommendation) if recommendation != "N/A" else "N/A"
                row.append(Paragraph(recommendation_str, centered_style))

        table_data.append(row)

    # Create the table with the data
    table = Table(table_data, style=[
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ], colWidths=75)  # Set the column widths

    # Wrap the table in a container to control the width
    container = []
    container.append(table)

    # Build the document with the wrapped table
    doc.build(container)


industry_pe_ratios = {}


def fetch_pe_ratios_by_industry(industry_name):
    url = "https://fullratio.com/pe-ratio-by-industry"
    industry_name = str(industry_name) if industry_name else ""
    response = requests.get(url)
    global industry_pe_ratios

    if response.status_code == 200:
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')

        html_str = str(soup)
        html_io = StringIO(html_str)  # Wrap the HTML string in StringIO

        # Find the table and convert it to a DataFrame
        tables = pd.read_html(html_io, flavor='bs4')

        # Assuming the first table is the one we want
        if industry_name in industry_pe_ratios:
            return industry_pe_ratios[industry_name]
        else:
            if tables:
                pe_ratios_df = tables[0]
                # Find the row for the specified industry
                industry_row = pe_ratios_df[pe_ratios_df['Industry'].str.contains(industry_name, case=False, na=False)]

                if not industry_row.empty:
                    # Extract the 'Average P/E ratio' value
                    average_pe_ratio = industry_row['Average P/E ratio'].values[0]
                    industry_pe_ratios[industry_name] = average_pe_ratio
                    return average_pe_ratio
                else:
                    print(f"Industry '{industry_name}' not found.")
                    return None

    else:
        print("Failed to fetch the webpage.")
        return None


def get_financial_data(tickers):
    data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        data.append({
            "Ticker": ticker,
            "Industry": info.get("industry"),
            "Sector": info.get("sector"),
            "Payout Ratio": info.get("payoutRatio"),
            "Forward P/E": info.get("forwardPE"),
            "EPS": info.get("forwardEps"),
            "P/B": info.get("priceToBook"),
            "Dividend Yield": info.get("dividendYield"),
            "ROE": info.get("returnOnEquity"),
        })
    return data


# HERE WE HAVE GROWTH RELATED FUNCTIONS
def get_average_fcf_growth_rate(ticker_symbol):
    """
    Calculates the average growth rate based on the Free Cash Flow (FCF) data from yfinance,
    correcting the order of years to be chronological.
    :param ticker_symbol: The symbol of the company to analyze.
    :return: The average FCF growth rate.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        cashflow = ticker.cashflow

        # Attempt to access the 'Free Cash Flow' data
        fcf = cashflow.loc['Free Cash Flow']

        # Convert index to datetime if they are string representations of years/dates
        fcf.index = pd.to_datetime(fcf.index)

        # Ensure chronological order by sorting the index
        fcf_sorted = fcf.sort_index()

        # Calculate year-over-year growth rates in chronological order
        growth_rates = fcf_sorted.pct_change().dropna()

        # Calculate the average growth rate
        average_growth_rate = growth_rates.mean()

    except KeyError:
        # Handle the case where 'Free Cash Flow' is not found
        print(f"'Free Cash Flow' data not found for {ticker_symbol}.")
        average_growth_rate = None  # You could use None or some other indicator that the data was not found

    return average_growth_rate


def get_historical_revenue_growth_rate(ticker_symbol):
    """
        Calculates the historical growth rate based on the average growth of "Total Revenue" from yfinance.

        :param ticker_symbol: Ticker symbol being analyzed.
        :return: Average growth rate of Total Revenue.
        """
    try:
        ticker = yf.Ticker(ticker_symbol)

        # Attempt to access the 'Total Revenue' data
        financials = ticker.financials
        revenue = financials.loc['Total Revenue']

        # Ensure revenue is in a numeric format to avoid future warnings
        revenue = revenue.astype(float)  # Convert to float before any operations

        # Reverse the data to order it from oldest to newest
        revenue = revenue[::-1]

        # Calculate yearly revenue growth rates
        revenue_growth_rates = revenue.pct_change().dropna()

        # Calculate the average of these growth rates over available years as an estimate
        average_growth_rate = revenue_growth_rates.mean()

    except KeyError:
        # Handle the case where 'Total Revenue' is not found
        print(f"'Total Revenue' data not found for {ticker_symbol}.")
        average_growth_rate = None  # You could use None or some other indicator that the data was not found

    return average_growth_rate


def get_five_year_growth_estimate(stock):
    """
    Takes the estimate for future company's growth from Yahoo Finance web page. The result
    is in whole numbers (if it is 16.03%, the function will return 16.03).

    :param stock: Stock ticker symbol.
    """
    url = f"https://finance.yahoo.com/quote/{stock}/analysis?p={stock}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    for table in soup.select("table"):
        ths = [th.text for th in table.find_all("th")]
        if "Growth Estimates" in ths:
            for tr in table.select("tr:has(td)"):
                tds = [td.text for td in tr.find_all("td")]
                if tds[0] == "Next 5 Years (per annum)":
                    # Remove the '%' character and convert to float
                    return round((float(tds[1].replace('%', ''))) / 100, 3)


# HERE WE HAVE MODELS-RELATED FUNCTIONS
def simplistic_dcf_model(stock, info, revenue_growth):
    forward_eps = info.get('forwardEps')
    pe_ratio = info.get('forwardPE')
    min_rate_of_return = 0.05
    margin_of_safety = 0.15
    years = 10
    current_price = info.get('currentPrice')

    payout_ratio = info.get("payoutRatio")
    if payout_ratio is None:
        payout_ratio_display = "N/A"
    else:
        payout_ratio_display = round(payout_ratio * 100, 2)

    div_yield = info.get("dividendYield")
    if div_yield is None:
        div_yield_display = "N/A"
    else:
        div_yield_display = round(payout_ratio * 100, 2)

    result = {
        "Ticker": stock,
        "Payout Ratio": payout_ratio_display,
        "Forward P/E": round(pe_ratio, 2),
        "Dividend Yield": div_yield_display,
        "Minimum Return": round(min_rate_of_return, 2),
        "Intrinsic Value ($)": None,
        "Margin of Safety": round(margin_of_safety, 2),
        "Buy Price": None,
    }

    if forward_eps and pe_ratio and revenue_growth:
        present_value = forward_eps * pe_ratio * ((1 + revenue_growth) ** (years - 1)) / (
                (1 + min_rate_of_return) ** (years - 1))
        buy_price = present_value * (1 - margin_of_safety)

        if buy_price >= current_price:
            recommendation = "BUY"
        elif buy_price < current_price:
            recommendation = "DO NOT BUY"
        else:
            recommendation = "N/A"

        result["SDCF Rec"] = recommendation
        result["Intrinsic Value ($)"] = round(present_value, 2)
        result["Buy Price"] = round(buy_price, 2)
    else:
        result["SDCF Rec"] = "N/A"

    return result


def dcf_valuation(stock, current_price, revenue_growth, api_key):
    """
    Function to calculate the intrinsic value of a stock, based on the financetoolkit package.

    :param stock: the stock symbol (ex. "PEP")
    :param current_price:
    :param revenue_growth:
    :param api_key:
    :return:
    """
    companies = Toolkit([stock], api_key=api_key, start_date="2017-12-31")
    wacc = companies.models.get_weighted_average_cost_of_capital()
    average_wacc = wacc.loc["Weighted Average Cost of Capital"].mean()
    perpetual_growth_rate = 0.04

    # Assuming revenue_growth is already obtained and passed to this function
    if 0 < revenue_growth < 0.25:
        intrinsic_valuation = companies.models.get_intrinsic_valuation(
            growth_rate=revenue_growth,
            perpetual_growth_rate=perpetual_growth_rate,
            weighted_average_cost_of_capital=average_wacc,
            rounding=2)

        # Assuming the intrinsic value is the last value in the DataFrame's first column
        intrinsic_value = intrinsic_valuation.iloc[-1].values[0]
        # Assuming get_profile()['DCF'] returns a Series with a single item
        intrinsic_value_ext = companies.get_profile().loc['DCF'].iloc[0]
        intrinsic_value_ext = float(intrinsic_value_ext)

        if intrinsic_value_ext > current_price:
            ext_recommendation = "BUY"
        else:
            ext_recommendation = "DO NOT BUY"

        if intrinsic_value > current_price:
            our_recommendation = "BUY"
        else:
            our_recommendation = "DO NOT BUY"

        return {"Model Name": "DCF Model",
                "Ticker": stock,
                "Revenue Growth (%)": round(revenue_growth * 100, 2),
                "Perpetual Growth Rate (%)": round(perpetual_growth_rate * 100),
                "WACC (%)": round(average_wacc * 100, 2),
                "Our Intrinsic Value ($)": round(intrinsic_value, 2),
                "External Intrinsic Value ($)": round(intrinsic_value_ext, 2),
                "Current Price ($)": round(current_price, 2),
                "DCF Model Rec": ext_recommendation,
                "Ours DCF Rec": our_recommendation}
    else:
        return {"Model Name": "DCF Model",
                "Info": f"The growth rate is {round(revenue_growth * 100, 2)}%. The model is not appropriate.",
                "Rec": "N/A"}


def dividend_growth_model_valuation(stock, current_price, dividend_rate, beta, ticker):
    """
    Calculate the intrinsic value using the Dividend Growth Model and suggest an action.

    :param stock: Stock ticker symbol.
    :param current_price: Current price of the stock.
    :param dividend_rate: Dividend rate.
    :param beta: Beta value of the stock.
    :param ticker: yfinance Ticker object for the stock.
    """

    if dividend_rate != 0:
        # Calculate the average dividend growth rate directly within this function
        dividends = ticker.dividends
        annual_dividends = dividends.groupby(dividends.index.year).sum()
        last_5_years = annual_dividends[-6:-1]  # Get the last 5 years of annual dividends
        growth_rates = last_5_years.pct_change().dropna()
        average_growth_rate = growth_rates.mean()

        Risk_Free_yield = yf.Ticker('^IRX').info.get('previousClose', 0) / 100  # Converted to decimal

        expected_market_return = 0.1102

        K_parameter = Risk_Free_yield + beta * (expected_market_return - Risk_Free_yield)

        intrinsic_value_dividend = dividend_rate / (K_parameter - average_growth_rate)

        if intrinsic_value_dividend > current_price:
            recommendation = "BUY"
        else:
            recommendation = "DO NOT BUY"

        return {"Model Name": "Dividend Growth Model",
                "Ticker": stock,
                "Dividend ($)": dividend_rate,
                "Beta": beta,
                "Average Dividend Growth Rate (%)": round(average_growth_rate * 100, 2),
                "Risk-free Yield (%)": round(Risk_Free_yield * 100, 2),
                "Expected Market Return (%)": round(expected_market_return * 100, 2),
                "K (Discount Rate, %)": round(K_parameter * 100, 2),
                "Intrinsic Value ($)": round(intrinsic_value_dividend, 2),
                "Current Price ($)": round(current_price, 2),
                "DGM Rec": recommendation}
    else:
        return {"Model Name": "Dividend Growth",
                "DGM Rec": "Model is not applicable due to stock not distributing dividends"}


def setup():
    ssl._create_default_https_context = ssl._create_unverified_context
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('future.no_silent_downcasting', True)


def main():
    setup()

    clean_reports_directory()

    selected_for_analysis = ["MCD", "PEP", "KO", "VZ", "SBUX", "JPM", "PFE", "LMT", "CVX", "TSM", "IBM", "AAPL", "AMZN",
                             "AMD", "INTC", "MSFT"]

    summary_data = []

    user_input = input(f"\nThose companies were selected: "
                       f"{selected_for_analysis}. Should I proceed? (Y/n): ").strip().lower()

    if (user_input == 'y') or (user_input == ''):
        user_input_for_DCF = input(f"\nOne more question. Would you like me to do the DCF? (Y/n): ").strip().lower()

        for stock in selected_for_analysis:
            print(f"\nAnalyzing {stock}...")
            results = {}
            ticker = yf.Ticker(stock)
            current_price = ticker.info.get('currentPrice')
            industry_pe = fetch_pe_ratios_by_industry(ticker.info.get("industry", ""))

            earningsGrowth = ticker.info.get('earningsGrowth', 0)
            revenueGrowth = ticker.info.get('revenueGrowth', 0)
            histGrowth = get_historical_revenue_growth_rate(stock)
            yGrowthEstimate = get_five_year_growth_estimate(stock)
            fcfGrowthEstimate = get_average_fcf_growth_rate(stock)

            growth_rates = [earningsGrowth, revenueGrowth, histGrowth, yGrowthEstimate, fcfGrowthEstimate]

            # Filter out any non-numeric values or None (if necessary)
            growth_rates = [g for g in growth_rates if isinstance(g, (int, float))]

            # Calculate the median of the growth rates
            growth = median(growth_rates)

            dividend_growth_results = dividend_growth_model_valuation(stock, current_price,
                                                                      ticker.info.get('dividendRate', 0),
                                                                      ticker.info.get('beta', 0), ticker)
            if ticker.info.get('dividendRate', 0) == 0:
                dgm_rec = 'N/A'
            else:
                dgm_rec = dividend_growth_results['DGM Rec']

            results["Dividend Growth Model"] = dividend_growth_model_valuation(stock, current_price,
                                                                               ticker.info.get('dividendRate', 0),
                                                                               ticker.info.get('beta', 0), ticker)

            if growth >= 0.25:
                results["Simplistic DCF Model"] = {"Model Name": "DCF Model",
                                                   "Info": f"The growth rate is {round(growth * 100, 2)}%. "
                                                           f"The company is experiencing rapid expansion, so model is "
                                                           f"not appropriate."}
                sdcf_results = 'N/A'
            elif 0 < growth < 0.25:
                results["Simplistic DCF Model"] = simplistic_dcf_model(stock, ticker.info, growth)
                sdcf_results = results["Simplistic DCF Model"]['SDCF Rec']
            elif 0 > growth:
                results["Simplistic DCF Model"] = {"Model Name": "DCF Model",
                                                   "Info": f"The growth rate is negative. "
                                                           f"The model is not appropriate."}
                sdcf_results = 'N/A'

            if (user_input_for_DCF == 'y') or (user_input_for_DCF == ''):
                if 0 < growth < 0.25:
                    try:
                        results["DCF Model"] = dcf_valuation(stock, current_price, growth, api_keys[1])
                        if 'DCF Model' in results:
                            dcf_results = results["DCF Model"].get('DCF Model Rec', 'N/A')
                        else:
                            dcf_results = 'N/A'
                    except Exception as e:
                        print(f"Exception: {e}\nTrying with next API key.")
                        results["DCF Model"] = dcf_valuation(stock, current_price, growth, api_keys[0])
                        if 'DCF Model' in results:
                            dcf_results = results["DCF Model"].get('DCF Model Rec', 'N/A')
                        else:
                            dcf_results = 'N/A'
                else:
                    results["DCF Model"] = {"Model Name": "DCF Model",
                                            "Info": f"The growth rate is {round(growth * 100, 2)}%. "
                                                    f"The model is not appropriate."}
                    dcf_results = "N/A"

            payout_ratio = ticker.info.get("payoutRatio")
            if payout_ratio is None:
                payout_ratio_display = "N/A"
            else:
                payout_ratio_display = round(payout_ratio * 100, 2)

            div_yield = ticker.info.get("dividendYield")
            if div_yield is None:
                div_yield_display = "N/A"
            else:
                div_yield_display = round(div_yield * 100, 2)

            stock_summary = {
                "Ticker": stock,
                "Payout Ratio": payout_ratio_display,
                "Forward P/E": round(ticker.info.get('forwardPE'), 2),
                "Dividend Yield": div_yield_display,
                "DGM Rec": dgm_rec,
                "DCF Model Rec": dcf_results,
                "SDCF Rec": sdcf_results,
                "good_pe": good_pe,
                "Industry P/E": industry_pe,
            }

            summary_data.append(stock_summary)

            # Generate report for the stock
            report_file_name = f"{stock}_Finance_Report.pdf"
            generate_pdf_report(stock, results, report_file_name)
            print(f"Analysis for {stock} completed. Report generated as {report_file_name}.")

        report_folder = 'reports'
        financial_data = get_financial_data(selected_for_analysis)
        create_financial_table_pdf("1_financial_data_table.pdf", financial_data)

        create_summary_table_pdf(summary_data, "z_final_summary_report.pdf")

        merge_pdfs(report_folder, '!Report.pdf', stocks=selected_for_analysis)

    else:
        print("Analysis aborted.")


if __name__ == "__main__":
    main()
