import os
import requests
from bs4 import BeautifulSoup
import json
import time
import re

# Define Paths
BASE_URL = 'https://www.whitehouse.gov/presidential-actions/'

RAW_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/raw"))


# Ensure the raw data directory exists
if not os.path.exists(RAW_DATA_DIR):
    os.makedirs(RAW_DATA_DIR)

def sanitize_filename(title):
    """
    Sanitizes the title to create a valid filename.
    - Removes special characters.
    - Replaces spaces with underscores.
    - Limits filename length to avoid OS restrictions.
    """
    title = re.sub(r"[^\w\s-]", "", title)  # Remove special characters
    # Capitalize the title
    title = title.capitalize()
    title = title.strip().replace(" ", "_")  # Replace spaces with underscores
    
    return title  # Limit filename length

def scrape_order(order_url):
    """
    Scrape the content of a single executive order and save it as a JSON file.  
    """
    response = requests.get(order_url)
    if response.status_code != 200:
        print(f"Failed to scrape {order_url}")
        return
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract the EO title, date and description
    title = soup.find('h1', class_ = "wp-block-whitehouse-topper__headline").text.strip()
    
    # Extract the date
    date = soup.find('div', class_='wp-block-post-date').text.strip()
        
    # Extract the description
    content_div = soup.find('div', class_='entry-content wp-block-post-content has-global-padding is-layout-constrained wp-block-post-content-is-layout-constrained')
    paragraphs = content_div.find_all('p') if content_div else []
    description = '\n'.join([p.get_text(strip=True) for p in paragraphs])
    
    
    # Prepare the JSON data
    data = {'ExecutiveOrder': [ 
        {'Title': title,
        'DateSigned': date,
        'Description': description,
        'URL': order_url
    }
    ]
    }
    
    # Generate a sanitized filename
    file_name = sanitize_filename(title)
    file_path = os.path.join(RAW_DATA_DIR, f"{file_name}.json")
    
    # Save the JSON data to a file
    
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    # print(f"Saved: {file_path}")
    

def scrape_all_orders(page_url):
    """
    Scrape all the Executive Orders from a single page and return the next page URL.
    """
    response = requests.get(page_url)
    if response.status_code != 200:
        print(f"Failed to scrape {page_url}")
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract and process executive order links in a single step
    orders = soup.select("ul.wp-block-post-template.is-layout-flow.wp-block-post-template-is-layout-flow h2.wp-block-post-title.has-heading-4-font-size a[href]")
    for order_url in orders:
        scrape_order(order_url['href'])
    
    # Find the 'NEXT' button for pagination 
    next_page = soup.select_one('nav.wp-block-query-pagination.is-layout-flex.wp-block-query-pagination-is-layout-flex a.wp-block-query-pagination-next')
    return next_page['href'] if next_page else None
    
def scrape(url=BASE_URL):
    """ 
    Main function to scrape EO across all paginated pages
    """
    url = BASE_URL
    while url:
        
        # print(f"Scraping: {current_url}")
        next_page_url = scrape_all_orders(url)
        
        # Add a delay to avoid flooding the website with requests
        time.sleep(1)
        
        if next_page_url:
            url = next_page_url
        else:
            break
        
if __name__ == "__main__":
    scrape()
    print(f'Scraping complete')
    