import requests
import pdfplumber
import re
import os
import concurrent.futures


def download_annual_report(company, year, filepath):
    if year == '2022':
        url = f'https://www.annualreports.com/HostedData/AnnualReports/PDF/{company}_{year}.pdf'
        response = requests.get(url)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded {company} {year}")
            return True
    else:
        for prefix in [chr(97 + i) for i in range(26)]:  # Iterates through all lowercase letters (a to z)
            current_url = f'https://www.annualreports.com/HostedData/AnnualReportArchive/{prefix}/{company}_{year}.pdf'
            response = requests.get(current_url)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Successfully downloaded {company} {year}")
                return True
    print(f"Failed to download {company} {year}")
    return False


def download_annual_reports_parallel(companies, years, reports_dir):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor: # Adjust max_workers as needed
        futures = []
        for company in companies:
            for year in years:
                filename = f"{company}_{year}.pdf"
                filepath = os.path.join(reports_dir, filename)
                futures.append(executor.submit(download_annual_report, company, year, filepath))


if __name__ == '__main__':
    # Directory where reports will be saved
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)

    # List of company PDFs to check
    companies = ['NASDAQ_TSLA', 'NASDAQ_AAPL', 'NASDAQ_MSFT', 'NASDAQ_AMZN', 'NYSE_BRK-A']
    years = ['2022', '2021', '2020']

    download_annual_reports_parallel(companies, years, reports_dir)

    print('end')