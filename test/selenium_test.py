from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup
from time import sleep, time
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver import ChromeOptions

# chrome_options = Options()
# chrome_options.add_argument('--no-sandbox')
# chrome_options.add_argument('--disable-dev-shm-usage')
# chrome_options.add_argument('--headless')
# chrome_options.add_argument('blink-settings=imagesEnabled=false')
# chrome_options.add_argument('--disable-gpu')

# 规避检测
option = ChromeOptions()
option.add_argument('--no-sandbox')
option.add_argument('--disable-dev-shm-usage')
# option.add_argument('--headless')
option.add_argument('blink-settings=imagesEnabled=false')
option.add_argument('--disable-gpu')
option.add_experimental_option('excludeSwitches', ['enable-automation'])
option.add_argument("--proxy-server=http://192.168.130.1:7891")
desired_capabilities = DesiredCapabilities.CHROME
desired_capabilities["pageLoadStrategy"] = "none"

browser = webdriver.Chrome(options=option)
begin = time()
browser.implicitly_wait(10)
url = 'http://dx.doi.org/10.1016/j.joi.2022.101258'
browser.get(url)
sleep(3)
print(browser)
element = browser.find_element(by=By.CLASS_NAME, value="text-xs")
print(element.text)
end = time()
print(end-begin)
