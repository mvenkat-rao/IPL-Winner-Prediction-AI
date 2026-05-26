import urllib.request
import re

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
urllib.request.install_opener(opener)

pages = {
    'rcb': 'https://en.wikipedia.org/wiki/Royal_Challengers_Bengaluru',
    'srh': 'https://en.wikipedia.org/wiki/Sunrisers_Hyderabad',
    'rr': 'https://en.wikipedia.org/wiki/Rajasthan_Royals',
    'lsg': 'https://en.wikipedia.org/wiki/Lucknow_Super_Giants'
}

for code, url in pages.items():
    try:
        html = urllib.request.urlopen(url).read().decode('utf-8')
        match = re.search(r'src="(//upload\.wikimedia\.org/wikipedia/en/thumb/[^"]+\.(?:png|svg))', html)
        if not match:
            match = re.search(r'src="(//upload\.wikimedia\.org/wikipedia/en/[^"]+\.(?:png|svg))', html)
        
        if match:
            img_url = 'https:' + match.group(1)
            ext = 'png' if '.png' in img_url.lower() else 'svg'
            print(f'Found {code}: {img_url}')
            urllib.request.urlretrieve(img_url, f'static/logos/{code}.{ext}')
        else:
            print(f'Not found for {code}')
    except Exception as e:
        print(f'Failed {code}: {e}')
