# Raw Data Description

**Dataset Name**: Lending Club Loan Data (2007-2017)
**Source File**: `accepted_2007_to_2017.csv` (extracted from `datasetzip.zip`)

## Overview
This dataset contains complete loan data for all loans issued through the 2007-2015 period, including the current loan status (Current, Late, Fully Paid, etc.) and latest payment information.

## Selected Columns
The `src/data_loading.py` script optimizes data loading by selecting only the following relevant columns:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `loan_amnt` | float32 | The listed amount of the loan applied for by the borrower. |
| `term` | str | The number of payments on the loan. Values are in months and can be either 36 or 60. |
| `int_rate` | float32 | Interest Rate on the loan. |
| `installment` | float32 | The monthly payment owed by the borrower if the loan originates. |
| `grade` | str | LC assigned loan grade. |
| `sub_grade` | str | LC assigned loan subgrade. |
| `emp_length` | str | Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. |
| `home_ownership` | str | The home ownership status provided by the borrower during registration. |
| `annual_inc` | float32 | The self-reported annual income provided by the borrower during registration. |
| `verification_status` | str | Indicates if income was verified by LC, not verified, or if the income source was verified. |
| `issue_d` | str | The month which the loan was funded. |
| `loan_status` | str | Current status of the loan. |
| `purpose` | str | A category provided by the borrower for the loan request. |
| `addr_state` | str | The state provided by the borrower in the loan application. |
| `dti` | float32 | A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income. |
| `delinq_2yrs` | float32 | The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years. |
| `earliest_cr_line` | str | The month the borrower's earliest reported credit line was opened. |
| `inq_last_6mths` | float32 | The number of inquiries in past 6 months (excluding auto and mortgage inquiries). |
| `open_acc` | float32 | The number of open credit lines in the borrower's credit file. |
| `pub_rec` | float32 | Number of derogatory public records. |
| `revol_bal` | float32 | Total credit revolving balance. |
| `revol_util` | float32 | Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit. |
| `total_acc` | float32 | The total number of credit lines currently in the borrower's credit file. |
| `last_pymnt_d` | str | Last month payment was received. |
| `last_credit_pull_d` | str | The most recent month LC pulled credit for this loan. |
| `application_type` | str | Indicates whether the loan is an individual application or a joint application with two co-borrowers. |
| `tot_coll_amt` | float32 | Total collection amounts ever owed. |
| `tot_cur_bal` | float32 | Total current balance of all accounts. |
| `total_rev_hi_lim` | float32 | Total revolving high credit/credit limit. |
