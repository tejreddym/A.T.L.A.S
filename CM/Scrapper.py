import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_wbjee_data(url):
    """
    This function scrapes the data from the provided URL,
    handles pagination to collect data from all pages,
    and returns a clean DataFrame without duplicates.
    """
    # Base URL for the website
    base_url = "https://admissions.nic.in/wbjeeb/Applicant/report/orcrreport.aspx"

    # Start with the first page
    current_page = 1
    all_data = []
    
    # Use a session object to maintain cookies and other session data
    with requests.Session() as session:
        # First, get the initial page to extract hidden form values
        # that are necessary for subsequent requests.
        try:
            response = session.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            soup = BeautifulSoup(response.content, 'html.parser')
        except requests.exceptions.RequestException as e:
            print(f"Error fetching initial URL: {e}")
            return None

        # --- Handling Pagination ---
        # ASP.NET pages often use hidden form fields like __VIEWSTATE to manage state.
        # To go to the next page, we need to send these values back to the server.
        while True:
            print(f"Scraping page {current_page}...")

            # Extract headers from the first page
            if not all_data:
                table = soup.find('table', id='ORCRGridView')
                if not table:
                    print("Could not find the data table on the page.")
                    break
                headers = [th.text.strip() for th in table.find('thead').find_all('th')]

            # Extract data from the current page's table
            table = soup.find('table', id='ORCRGridView')
            if not table:
                break # Stop if no table is found

            rows = table.find('tbody').find_all('tr')
            if not rows:
                break # Stop if no more rows are found

            for row in rows:
                cols = [ele.text.strip() for ele in row.find_all('td')]
                all_data.append(cols)

            # --- Logic to go to the next page ---
            # Find the link or button for the next page to determine if there is a next page.
            # In many ASP.NET tables, if the "Next" link is disabled or not present, it's the last page.
            next_page_link = soup.find('a', text='Next')
            if not next_page_link or next_page_link.has_attr('disabled'):
                break

            # To go to the next page, you typically need to make a POST request
            # with specific form data. You need to inspect the 'Next' button's
            # JavaScript function or the network request it triggers to find the
            # exact payload.
            
            # Extract hidden form fields required for the POST request
            viewstate = soup.find('input', {'name': '__VIEWSTATE'})['value']
            eventvalidation = soup.find('input', {'name': '__EVENTVALIDATION'})['value']
            viewstategenerator = soup.find('input', {'name': '__VIEWSTATEGENERATOR'})['value']

            # This is a common payload structure for ASP.NET pagination.
            # You might need to adjust '__EVENTTARGET' based on what you find
            # by inspecting the page's network activity in your browser's developer tools.
            form_data = {
                '__EVENTTARGET': 'ORCRGridView',  # This is often the ID of the table
                '__EVENTARGUMENT': f'Page${current_page + 1}',
                '__VIEWSTATE': viewstate,
                '__EVENTVALIDATION': eventvalidation,
                '__VIEWSTATEGENERATOR': viewstategenerator,
            }

            try:
                # Make the POST request to get the next page
                response = session.post(url, data=form_data)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                current_page += 1
            except requests.exceptions.RequestException as e:
                print(f"Error fetching page {current_page + 1}: {e}")
                break

    if not all_data:
        print("No data was scraped.")
        return None

    # Create a pandas DataFrame
    df = pd.DataFrame(all_data, columns=headers)
    
    # --- Remove Duplicates ---
    # The drop_duplicates() method will remove any rows that are identical.
    df.drop_duplicates(inplace=True)
    
    return df

if __name__ == "__main__":
    # The URL from which to scrape the data
    target_url = "https://admissions.nic.in/wbjeeb/Applicant/report/orcrreport.aspx?enc=Nm7QwHILXclJQSv2YVS+7ud0s9OnRxxLItScoKR31F4qbKNJ7YB3loiJ7DTFho11"
    
    # Scrape the data
    scraped_df = scrape_wbjee_data(target_url)
    
    if scraped_df is not None:
        # Display the first 10 rows of the scraped data
        print("\n--- Scraped Data ---")
        print(scraped_df.head(10))
        
        # Display the total number of unique rows found
        print(f"\nTotal unique rows scraped: {len(scraped_df)}")
        
        # --- Save the data to a CSV file ---
        # You can uncomment the line below to save the data to a CSV file.
        scraped_df.to_csv("wbjee_data.csv", index=False)
        print("\nData saved to wbjee_data.csv")