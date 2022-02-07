# Business Intelligence (BI)

## OLAP (Online analytical processing)

- What are the basic analytical operations of OLAP?

1. Roll-up
2. Drill-down
3. Slice and dice
4. Pivot (rotate)

1) Roll-up:

Roll-up is also known as “consolidation” or “aggregation.” The Roll-up operation
can be performed in 2 ways:

- Reducing dimensions
- Climbing up concept hierarchy. Concept hierarchy is a system of grouping
  things based on their order or level.

Example: Moving from cities to countries

2. Drill-down

In drill-down data is fragmented into smaller parts. It is the opposite of the rollup
process. It can be done via:

- Moving down the concept hierarchy
- Increasing a dimension

E.g., Moving from quarters to months

3. Slice:

Here, one dimension is selected, and a new sub-cube is created.

E.g., Dimension Time is Sliced with Q1 as the filter.

Dice:

This operation is similar to a slice. The difference in dice is you select 2 or more dimensions that result in the creation of a sub-cube.

4. Pivot

In Pivot, you rotate the data axes to provide a substitute presentation of data.

## Data Analysis Expressions (DAX)

Data Analysis Expressions is the native formula and query language for Microsoft PowerPivot, Power BI Desktop and SQL Server Analysis Services Tabular models.

### Basic Aggregate and Math functions

#### Total Sales, Cost, and Profit

- SUM

```
total_sales = SUM('TableName'[SalesAmount])
total_cost = SUM('TableName'[Cost])
profit = [total_sales] - [total_cost]
```

- DIVIDE

```
profit_margin = DIVIDE([profit], [total_sales])
```

- COUNTROWS (Transactions)

```
transactions = COUNTROWS('Table')
```

- Related table count

```
transactions = COUNTROWS(RELATEDTABLE('Table'))
```

- Conditional Count

```
count = CALCULATE(
    SUM(DISTINCTCOUNT('TableName'[ID])),
    'TableName'[Type] = 1
)
```

#### Month To Date Sales

Month-to-date (MTD): a period starting at the beginning of the current calendar month and
ending at the current date.

Example: If today is the 15th of the month, and your manager asks you for the month to
date sales figures, you will want to add your sales from the 1st of the month up to the 14th
(as the 15th is not complete yet).

- TOTALMTD

```
mtd_sales = TOTALMTD([total_sales], 'TableName'[date_column])
```

#### Year To Date Sales

• Year To Date (YTD) sales formulas: the amount of profit (or loss) realized by an investment
since the first trading day of the current calendar year.

• YTD calculations are commonly used by investors and analysts to assess the performance of
a portfolio or to compare the recent performance of a group of stocks.

• Using the YTD period sets a common time frame for assessing the performance of securities
against each other and their benchmarks. A YTD period is also useful for measuring price
movements relative to other data, such as the economic indicators.

• Example: Year to date value of sales could be the summary of all sales from the 1st of
January of that year to a specified date.

- TOTALYTD

```
ytd_sales = TOTALYTD([total_sales], 'TableName'[date_column])
```

```
# with optional parameter specifying the fiscal year end date
ytd_sales = TOTALYTD([total_sales], 'TableName'[date_column], "05/31")
```

#### Prior Year Sales

• Prior Year Sales formulas: used to track your business's performance by comparing a statistic
for a select period with the same period from the previous year.

• Example: Let's say your business revenue rose 25% last month. Before you celebrate, check
that against the income from the same month last year. Maybe your sales usually rise this time
of year. If sales typically rise 35% this month, then at 25% your revenue is down year-over-year.
Your business is doing worse, not better.

- SAMEPERIODLASTYEAR

```
prior_year_profit = CALCULATE([profit], SAMEPERIODLASTYEAR('TableName'[date_column]))
year_over_year_profit = [profit] - [prior_year_profit]
lastyear_ytd_sales = CALCULATE([ytd_sales], SAMEPERIODLASTYEAR('TableName'[date_column]))
```

#### Moving Totals

• The Moving Average (MA) formula: a technique to get an overall idea of the trends in a data
set; this technique is an average of any subset of numbers.

• The Moving Average is very useful for forecasting long-term trends. You can calculate it for a
certain period of time.

• For example: If you have sales data for a twenty-year period, you can calculate a five-year
moving average, a four-year moving average, and so on.

```
# calculated measure that returns a rolling 12 months total for profit
rolling_12_months_profit = CALCULATE(
    [profit],
    DATESBETWEEN('TableName'[date_column]),
    NEXTDAY(
        SAMEPERIODLASTYEAR(
            LASTDATE('TableName'[date_column])
        )
    ),
    LASTDATE('TableName'[date_column])
)
```

```
# 7 day moving average profit
seven_day_moving_average = AVERAGEX(
    FILTER(
        ALL('TableName'),
        'TableName'[FullDateAlternateKey] > MAX('TableName'[FullDateAlternateKey] - 7) &&
        'TableName'[FullDateAlternateKey] <= MAX('TableName'[FullDateAlternateKey])
    ),
    [profit]
)
```
