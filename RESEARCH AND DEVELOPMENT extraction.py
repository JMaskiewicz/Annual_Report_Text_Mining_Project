import requests
import pdfplumber
import re
import os
import concurrent.futures

from functions.download_reports import download_annual_reports_parallel

# Directory where reports will be saved
reports_dir = 'reports'
os.makedirs(reports_dir, exist_ok=True)

# List of company PDFs to check
companies = ['NASDAQ_TSLA', 'NASDAQ_AAPL', 'NASDAQ_MSFT', 'NASDAQ_AMZN',
             'NYSE_BRK-A', 'NYSE_PFE', 'NASDAQ_CCBG'
]
years = ['2022', '2021', '2020', '2019']
download_annual_reports_parallel(companies, years, reports_dir)

def extract_discussion(text):
    pass

def is_toc_page(text):
    toc_patterns = [
        r'\bcontents\b',
        r'\bindex\b',
        r'\btable of contents\b',
        r'\.\.\.\s+\d+',
        r'[A-Za-z].*\.\.\.\s+\d+'
    ]

    for pattern in toc_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False

def extract_toc(pdf):
    toc_text = ''
    toc_started = False
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            if is_toc_page(text):
                toc_text += text + "\n"
                toc_started = True
            elif toc_started:
                break
    return toc_text

# Dictionary to store results
toc_extraction_results = {}

for company in companies:
    for year in years:
        filename = f"{company}_{year}.pdf"
        filepath = os.path.join(reports_dir, filename)

        if os.path.exists(filepath):
            with pdfplumber.open(filepath) as pdf:
                toc_text = extract_toc(pdf)
                if toc_text:
                    toc_extraction_results[f"{company}_{year}"] = {'Status': 'Success', 'ToC': toc_text}
                else:
                    toc_extraction_results[f"{company}_{year}"] = {'Status': 'Failed', 'ToC': ''}
        else:
            toc_extraction_results[f"{company}_{year}"] = {'Status': 'Download Failed', 'ToC': ''}


def extract_section_pages(toc_text, section_keywords):
    section_pages = []
    lines = toc_text.split('\n')

    for i, line in enumerate(lines):
        for keyword in section_keywords:
            if keyword.lower() in line.lower():
                # Match page number at the end of the line
                page_number_pattern = r'(\d+)$'  # Matches numbers at the end of a line
                page_match = re.search(page_number_pattern, line)

                if page_match:
                    start_page = int(page_match.group())  # Convert to integer
                    end_page = None

                    # Find the start of the next section to determine the end page
                    for j in range(i + 1, len(lines)):
                        next_section_match = re.search(page_number_pattern, lines[j])
                        if next_section_match:
                            next_page = int(next_section_match.group())  # Convert to integer
                            end_page = next_page - 1 if next_page > start_page else start_page
                            break

                    if not end_page:
                        end_page = start_page  # Fallback if no end page is found
                    section_pages.append((start_page, end_page))
                    break

    return section_pages


def find_pdf_page_number(page):
    # Extract text and find the last number which might be the actual page number
    text = page.extract_text()
    if text:
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[-1])  # Return the last number found as the potential page number
    return None


def extract_pages_from_pdf(filepath, page_ranges):
    extracted_text = {}
    with pdfplumber.open(filepath) as pdf:
        # Create a mapping of actual PDF page numbers to their indices
        pdf_page_map = {find_pdf_page_number(page): i for i, page in enumerate(pdf.pages)}

        for section, (toc_start, toc_end) in page_ranges.items():
            section_text = ""
            for toc_page in range(toc_start, toc_end + 1):
                pdf_page_index = pdf_page_map.get(toc_page)
                if pdf_page_index is not None:
                    page_text = pdf.pages[pdf_page_index].extract_text()
                    if page_text:
                        section_text += page_text + "\n"

            extracted_text[section] = section_text

    return extracted_text


# Dictionary to store results
toc_extraction_results = {}

# Directory for saving discussions
discussion_dir = 'discussion'
os.makedirs(discussion_dir, exist_ok=True)

for company in companies:
    for year in years:
        filename = f"{company}_{year}.pdf"
        filepath = os.path.join(reports_dir, filename)
        if os.path.exists(filepath):
            with pdfplumber.open(filepath) as pdf:
                toc_text = extract_toc(pdf)
                if toc_text:
                    toc_extraction_results[f"{company}_{year}"] = {'Status': 'Success', 'ToC': toc_text}
                else:
                    toc_extraction_results[f"{company}_{year}"] = {'Status': 'Failed', 'ToC': ''}
        else:
            toc_extraction_results[f"{company}_{year}"] = {'Status': 'Download Failed', 'ToC': ''}

for key, result in toc_extraction_results.items():
    if result['Status'] == 'Success':
        filepath = os.path.join(reports_dir, f"{key}.pdf")
        toc_text = result['ToC']
        discussion_analysis_pages = extract_section_pages(toc_text, ['management’s discussion', 'management’s report', 'discussion'])

        if discussion_analysis_pages:
            extracted_sections = extract_pages_from_pdf(filepath, {"Discussion and Analysis": discussion_analysis_pages[0]})
            discussion_text = extracted_sections.get("Discussion and Analysis", "Section not found")

            output_filename = os.path.join(discussion_dir, f"{key}_DISCUSSION.txt")
            with open(output_filename, 'w', encoding='utf-8') as file:
                file.write(discussion_text)

            print(f"Discussion for {key} saved as {output_filename}")
        else:
            print(f"Discussion section not found for {key}")

print('end')