#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


## 선언
import os
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
from urllib.parse import quote_plus

baseUrl = 'https://www.google.com/?&bih=939&biw=1680&rlz=1C1SQJL_koKR895KR895&hl=ko'

save_root = 'downloads'
if not os.path.exists(save_root):os.makedirs(save_root)

def get_images(query='apple', limit=20):
    save_path = os.path.join(save_root, query)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 한글 검색 자동 변환
    url = baseUrl + quote_plus(query)
    html = urlopen(url)
    soup = bs(html, "html.parser")
    img = soup.find_all(class_='_img', limit=limit)

    n = 1
    for i in img:
        imgUrl = i['data-source']
        with urlopen(imgUrl) as f:
            with open(os.path.join(save_path, str(n)+'.jpg'),'wb') as h: # w - write b - binary
                img = f.read()
                h.write(img)
        n += 1
    print('%s download complete' % (query))


# In[ ]:


queries =  ['헬멧',
            '오토바이',
            '얼굴',
            ]

num_limit = 1100
    
for query in queries:
    get_images(query=query, limit=num_limit)
    
print('done!!');beep = lambda x: os.system("echo -n '\a';sleep 0.3;" * x);beep(3);


# In[ ]:


get_ipython().system('pip install Selenium')
get_ipython().system('apt-get update # to update ubuntu to correctly run apt install')
get_ipython().system('apt install chromium-chromedriver')
get_ipython().system('cp /usr/lib/chromium-browser/chromedriver /usr/bin')
import sys
sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
from selenium import webdriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
wd = webdriver.Chrome('chromedriver',chrome_options=chrome_options)
wd.get("https://www.webite-url.com")


# In[ ]:


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os

class ImageCrawler(object):
    def __init__(self, search_term, num_to_search, out_dir):
        url = "https://www.google.co.in/search?q=" + search_term + "&tbm=isch"
        browser = webdriver.Chrome("chromedriver.exe")
        browser.get(url)


# In[ ]:


from selenium import webdriver
wd = webdriver.Chrome('chromedriver',chrome_options=chrome_options)
search_term = '헬멧'
url = "https://www.google.co.in/search?q=" + search_term + "&tbm=isch"
wd.get(url)

for i in range(200):
    wd.execute_script('window.scrollBy(0,10000)')

for idx, el in enumerate(wd.find_elements_by_class_name("rg_ic")):
    el.screenshot(str(idx) + ".png")


first_list = ['헬멧', '얼굴', '사람얼굴']
second_list = ['오토바이', '오토바이탄사람', '정면얼굴', '측면', '오토바이사진']

for first in first_list:
    for second in second_list:
        new_query = first + " " + second
        url = "https://www.google.co.in/search?q=" + new_query + "&tbm=isch"
        wd.get(url)
       


# In[ ]:


from selenium import webdriver
import requests as req
import time
from selenium.webdriver.common.keys import Keys
from urllib.request import urlopen

#브라우저를 크롬으로 만들어주고 인스턴스를 생성해준다.
browser = wd 
#브라우저를 오픈할 때 시간간격을 준다.
browser.implicitly_wait(3)

count = 0
검색어 = '헬멧'

photo_list = []
before_src=""

#개요에서 설명했다시피 google이 아니라 naver에서 긁어왔으며, 
#추가적으로 나는 1027x760이상의 고화질의 해상도가 필요해서 아래와 같이 추가적인 옵션이 달려있다.
경로 = "https://search.naver.com/search.naver?where=image&section=image&query="+검색어+"&res_fr=786432&res_to=100000000&sm=tab_opt&face=0&color=0&ccl=0&nso=so%3Ar%2Ca%3Aall%2Cp%3Aall&datetype=0&startdate=0&enddate=0&start=1"

#해당 경로로 브라우져를 오픈해준다.
browser.get(경로)
time.sleep(1)

SCROLL_PAUSE_TIME = 1.0
reallink = []

while True:
    pageString = browser.page_source
    bsObj = BeautifulSoup(pageString, 'lxml')

    for link1 in bsObj.find_all(name='div', attrs={"class":"Nnq7C weEfm"}):
        for i in range(3):
            title = link1.select('a')[i]
            real = title.attrs['href']
            reallink.append(real)

    last_height = browser.execute_script('return document.body.scrollHeight')
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE_TIME)
    new_height = browser.execute_script("return document.body.scrollHeight")

    if new_height == last_height:
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = browser.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            break
        else:
            last_height = new_height
            continue

#아래 태그의 출처는 사진에서 나오는 출처를 사용한것이다.
#여기서 주의할 점은 find_element가 아니라 elements를 사용해서 아래 span태그의 img_border클래스를 
#모두 가져왔다.
photo_list = browser.find_elements_by_tag_name("span.img_border")


for index, img in enumerate(photo_list[0:]):
    #위의 큰 이미지를 구하기 위해 위의 태그의 리스트를 하나씩 클릭한다.
    img.click()
    
    #한번에 많은 접속을 하여 image를 크롤링하게 되면 naver, google서버에서 우리의 IP를 10~20분
    #정도 차단을 하게된다. 때문에 Crawling하는 간격을 주어 IP 차단을 피하도록 장치를 넣어주었다.
    time.sleep(2)
    
    #확대된 이미지의 정보는 img태그의 _image_source라는 class안에 담겨있다.
    html_objects = browser.find_element_by_tag_name('img._image_source')
    current_src = html_objects.get_attribute('src')
    print("=============================================================")
    print("현재 src :" +current_src)
    print("이전 src :" +before_src)
    if before_src == current_src:  
        continue
    elif before_src != current_src:
        t = urlopen(current_src).read()
        if index < 1000 :
            filename = "Car_"+str(count)+".jpg"
            with open(filename, "wb") as f:
                f.write(t)
                count += 1
                before_src = current_src
                current_src = ""
            print("Img Save Success")
        else:
            browser.close()       
            break


# In[ ]:


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib.request


# In[ ]:


browser = webdriver.Chrome('chromedriver',chrome_options=chrome_options)
browser.get('https://www.google.com')
browser.close()


# In[ ]:


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from urllib.request import urlretrieve
from tqdm import tqdm
import time
import os

os.chdir("/content/drive/My Drive/smartcity_project")
def get_images(keyword):

    print("접속중")
    driver = webdriver.Chrome('chromedriver',chrome_options=chrome_options)
    driver.implicitly_wait(3)

    url='https://search.naver.com/search.naver?sm=tab_hty.top&where=image&query={}'.format(keyword)
    driver.get(url)

    
    body=driver.find_element_by_css_selector('body')
    
    num_of_pagedwons=10

    while num_of_pagedwons:
      body.send_keys(Keys.PAGE_DOWN)
      time.sleep(1)
      num_of_pagedwons -= 1
      try:
        driver.find_element_by_xpath("""//*[@id="btn_more _more"]/tCR""").click()
      except:
        None


    #이미지 링크 수집
    imgs= driver.find_elements_by_css_selector("img._img")
    result=[]
    for img in tqdm(imgs):
        if 'http' in img.get_attribute('src'):
            result.append(img.get_attribute('src'))

    driver.close() # 크롬창 자동 종료
    print("수집 완료")

    #폴더생성
    print("폴더 생성")
    #폴더가 없을때만 생성
    if not os.path.isdir('./{}'.format(keyword)):
        os.mkdir('./{}'.format(keyword))

    #다운로드
    for index, link in tqdm(enumerate(result)): #tqdm은 작업현황을 알려줌.
        start=link.rfind('.') #뒤쪽부터 검사
        end=link.rfind('&')
        filetype=link[start:end] # .jpg , .png 같은게 뽑힘

        urlretrieve(link,'./{}/{}{}{}'.format(keyword,keyword,index,filetype))

    print('다운로드 완료')

if __name__ == '__main__':
    keyword=input("수집할 키워드를 입력하세요: ")
    get_images(keyword)


# In[ ]:


def crawling():
    global crawled_count

    print("ㅡ 크롤링 시작 ㅡ")

    # 이미지 고급검색 중 이미지 유형 '사진'
    url = f"https://www.google.com/search?as_st=y&tbm=isch&hl=ko&as_q={query}&as_epq=&as_oq=&as_eq=&cr=&as_sitesearch=&safe=images&tbs=itp:photo"
    driver.get(url)
    driver.maximize_window()
    scroll_down()

    div = driver.find_element_by_xpath('//*[@id="islrg"]/div[1]')
    img_list = div.find_elements_by_css_selector(".rg_i.Q4LuWd")
    os.makedirs(path + date + '/' + query)
    print(f"ㅡ {path}{date}/{query} 생성 ㅡ")

    for index, img in enumerate(img_list):
        try:
            click_and_retrieve(index, img, len(img_list))

        except ElementClickInterceptedException:
            print("ㅡ ElementClickInterceptedException ㅡ")
            driver.execute_script("window.scrollTo(0, window.scrollY + 100)")
            print("ㅡ 100만큼 스크롤 다운 및 3초 슬립 ㅡ")
            time.sleep(3)
            click_and_retrieve(index, img, len(img_list))

        except NoSuchElementException:
            print("ㅡ NoSuchElementException ㅡ")
            driver.execute_script("window.scrollTo(0, window.scrollY + 100)")
            print("ㅡ 100만큼 스크롤 다운 및 3초 슬립 ㅡ")
            time.sleep(3)
            click_and_retrieve(index, img, len(img_list))

        except ConnectionResetError:
            print("ㅡ ConnectionResetError & 패스 ㅡ")
            pass

        except URLError:
            print("ㅡ URLError & 패스 ㅡ")
            pass

        except socket.timeout:
            print("ㅡ socket.timeout & 패스 ㅡ")
            pass

        except socket.gaierror:
            print("ㅡ socket.gaierror & 패스 ㅡ")
            pass

        except ElementNotInteractableException:
            print("ㅡ ElementNotInteractableException ㅡ")
            break

    try:
        print("ㅡ 크롤링 종료 (성공률: %.2f%%) ㅡ" % (crawled_count / len(img_list) * 100.0))

    except ZeroDivisionError:
        print("ㅡ img_list 가 비어있음 ㅡ")

    driver.quit()


# In[ ]:


def filtering():
    print("ㅡ 필터링 시작 ㅡ")
    filtered_count = 0
    dir_name = path + date + '/' + query
    for index, file_name in enumerate(os.listdir(dir_name)):
        try:
            file_path = os.path.join(dir_name, file_name)
            img = Image.open(file_path)

            # 이미지 해상도의 가로와 세로가 모두 350이하인 경우
            if img.width < 351 and img.height < 351:
                img.close()
                os.remove(file_path)
                print(f"{index} 번째 사진 삭제")
                filtered_count += 1

        # 이미지 파일이 깨져있는 경우
        except OSError:
            os.remove(file_path)
            filtered_count += 1

    print(f"ㅡ 필터링 종료 (총 갯수: {crawled_count - filtered_count}) ㅡ")


# In[ ]:


def checking():
    # 입력 받은 검색어가 이름인 폴더가 존재하면 중복으로 판단
    for dir_name in os.listdir(path):
        file_list = os.listdir(path + dir_name)
        if query in file_list:
            print(f"ㅡ 중복된 검색어 ({dir_name}) ㅡ")
            return True


# In[ ]:


def playing_mp3():
    mp3 = "Mococo_Seed.mp3"
    mixer.init()
    mixer.music.load(mp3)
    mixer.music.play()
    while mixer.music.get_busy():
        pass
    print(f"ㅡ 검색어: {query} ㅡ")


# In[ ]:


# clickAndRetrieve() 과정에서 urlretrieve 이 너무 오래 걸릴 경우를 대비해 타임 아웃 지정
socket.setdefaulttimeout(30)

# 이미지들이 저장될 경로 및 폴더 이름
path = "C:/Data/"
date = "2020.07.17"

# 드라이버 경로 지정 (Microsoft Edge)
driver = webdriver.Edge("C:/Users/user/msedgedriver")

# 크롤링한 이미지 수
crawled_count = 0

# 검색어 입력 받기
query = input("입력: ")
# 이미 크롤링했던 검색어일 때
while checking() is True:
    query = input("입력: ")

crawling()
filtering()
playing_mp3()

