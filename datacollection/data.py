#!/usr/bin/python3

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

# Set up WebDriver
service = Service("path/to/chromedriver")  # Update with your driver path
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

# Workday Jobs URL
url = "https://your-workday-job-url.com"  # Update with the correct URL
driver.get(url)
wait = WebDriverWait(driver, 10)

job_data = []

def scrape_jobs():
    while True:
        # Get job listings
        jobs = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".job-listing-class")))  # Update selector
        
        for job in jobs:
            try:
                job_title = job.find_element(By.CSS_SELECTOR, ".job-title-class").text
                job.click()  # Click to open details
                time.sleep(2)

                # Extract job details
                job_desc = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".job-description-class"))).text
                date_posted = driver.find_element(By.CSS_SELECTOR, ".date-posted-class").text  # Update selector
                
                job_data.append({"Title": job_title, "Description": job_desc, "Date Posted": date_posted})

                driver.back()  # Go back to job list
                time.sleep(2)

            except Exception as e:
                print("Error scraping job:", e)
        
        # Check for next page
        try:
            next_button = driver.find_element(By.CSS_SELECTOR, ".next-page-button-class")  # Update selector
            if "disabled" in next_button.get_attribute("class"):
                break
            next_button.click()
            time.sleep(3)
        except:
            break

# Run the scraper
scrape_jobs()

# Save data to CSV
df = pd.DataFrame(job_data)
df.to_csv("workday_jobs.csv", index=False)

# Close WebDriver
driver.quit()
