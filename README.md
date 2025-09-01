# FinModeling

This project started out around February 2024, when I felt the need to analyze some stocks.
So far, the project became more than stock analysis, and now it includes the code for:
- **Portfolio optimization** (made using three techniques:)
  - Regular optimization;
  - Optimization by Sharpe Ratio;
  - Optimization by minimizing the left tail;
- **Report generation**
    - Using the newest information provided by Yahoo Finance and financetoolkit library, the code produces PDFs that include:
        - The table with base financial parameters (P/E, EPS, Dividend Yield, etc.);
        - For each of the provided stocks, the code runs three valuation models (Dividend Growth Model, Simplistic DCF, and Regular DCF);
        - At the end, the table with color coded parameters and suggestions (BUY/DO NOT BUY) is displayed.
- **Telegram Bot** (@PortfolioSecretaryBot)
  - The bot is tasked with recording the transactions made by the admins, and actively monitoring the portfolio.
    Using appropriate commands, it will make a graph of the current
    returns of the portfolio, and a table with the numeric values for each of the positions.
  - All the data regarding portfolio is stored at the MySQL database.
  - For security purposes, the bot will only answer to the admin users. 

The project also contains the Dockerfile for the usage of the Telegram bot.
