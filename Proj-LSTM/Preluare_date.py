from urllib.parse import urlparse,urlencode
import ipaddress
import re
from bs4 import BeautifulSoup
import whois
import urllib
import urllib.request
from datetime import datetime
import requests


#1 adresa IP

#unele atacuri de phishing folosesc adrese IP in loc de domenii
#ptr ca e mai greu de filtrat
def haveIP(url):
    try:
        ipaddress.ip_address(url) #converteste url u intr o adresa ip
        ip = 1
    except:
        ip = 0
    return ip

#2. @ in url

def haveSign(url):
    if "@" in url:
        sim = 1
    else:
        sim = 0
    return sim

#3 lungimea URL

def getLength(url):
    if len(url) < 54:
        length = 0
    else:
        length = 1
    return length

#4 numarul de subpagini

def getSubpages(url):
    path = urlparse(url).path.split('/')
    depth = 0

    for i in range(len(path)):
        if len(path[i]) != 0:
            depth = depth + 1
    return depth

#5 verifica prezenta pozitiei //

def redirection(url):
    sym = url.rfind('//')
    if sym > 6:
        if sym > 7:
            return 1 #phishing
        else:
            return 0
    else:
        return 0

#6 verifica https in domeniu

def httpDomain(url):
    domain = urlparse(url).netloc
    if 'https' in domain:
        return 1
    else:
        return 0
    
#shortening services
shortening_services = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|" \
                      r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|" \
                      r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|" \
                      r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|" \
                      r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|" \
                      r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|" \
                      r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|" \
                      r"tr\.im|link\.zip\.net"

#8. verifica metode de scurtare a URL

def tinyURL(url):
    match = re.search(shortening_services, url)
    if match:
        return 1
    else:
        return 0
    
#9. phisherii tind sa adauge '-' la domenii pentru a face url ul mai credibil

def presenceDash(url):
    if '-' in urlparse(url).netloc:
        return 1 #phishing
    else:
        return 0
    
# caracteristicile legate de domeniu
#dns record
#trafic
#varsta domeniului
#perioada de final a domeniului

#10 trafic web

def web_traffic(url):
    try:
        url = urllib.parse.quote(url)
        rank = BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + url).read(), "xml").find(
        "REACH")['RANK']
        rank = int(rank)
    except TypeError:
        return 1
    if rank < 100000:
        return 1
    else:
        return 0
    
#11 varsta domeniului - diferenta dintre sfarsit - creatie

def domainAge(domain_name):
    creation_date = domain_name.creation_date
    expiration_date = domain_name.expiration_date

    if (isinstance(creation_date,str) or isinstance(expiration_date,str)):
        try:
            creation_date = datetime.strptime(creation_date,'%Y-%m-%d')
            expiration_date = datetime.strptime(expiration_date,"%Y-%m-%d")
        except:
            return 1
        if ((expiration_date is None) or (creation_date is None)):
            return 1
        elif ((type(expiration_date) is list) or (type(creation_date) is list)):
            return 1
        else:
            ageDomain = abs((expiration_date - creation_date).days)
            if ((ageDomain/30) < 6):
                age = 1
            else:
                age = 0
        return age

        

#12 cand se termina domeniul
def domainEnd(domain_name):
    expiration_date = domain_name.expiration_date
    if isinstance(expiration_date,str):
        try:
            expiration_date = datetime.strptime(expiration_date,"%Y-%m-%d")
        except:
            return 1
    if (expiration_date is None):
        return 1
    elif (type(expiration_date) is list):
        return 1
    else:
        today = datetime.now()
        end = abs((expiration_date - today).days)
        if ((end/30) < 6):
            end = 0
        else:
            end = 1
    return end


#html si javascript features

#IFrame redirection
#status Bar costumization
#disabling right click
#website forwarding

#iframe este un html tag folosit pentru a afisa un webpage in plus in webpage-ul afisat curent

def IFrame(response):
    if response == "":
        return 1
    else:
      if re.findall(r"[<iframe>|<frameBorder>]", response.text):
          return 0
      else:
          return 1
      


# 16.Checks the effect of mouse over on status bar (Mouse_Over)
def mouseOver(response): 
  if response == "" :
    return 1
  else:
    if re.findall("<script>.+onmouseover.+</script>", response.text):
      return 1
    else:
      return 0
    
# verifica statusul atributului de right-click

def rightClick(response):
  if response == "":
    return 1
  else:
    if re.findall(r"event.button ?== ?2", response.text):
      return 0
    else:
      return 1
    

#verifica numarul de forwardinguri
def forwarding(response):
  if response == "":
    return 1
  else:
    if len(response.history) <= 2:
      return 0
    else:
      return 1

    
#creem o lista si o functie care apeleaza celalalte functii si 
#stocheaza toate caracteristicile URL-ului in lista

def featureExtraction(url):
    features = []
    features.append(haveIP(url))
    features.append(haveSign(url))
    features.append(getLength(url))
    features.append(getSubpages(url))
    features.append(redirection(url))
    features.append(httpDomain(url))
    features.append(tinyURL(url))
    features.append(presenceDash(url))

