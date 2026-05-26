import urllib.request
import json
import os

teams = {
    'rcb': 'Royal Challengers Bengaluru',
    'srh': 'Sunrisers Hyderabad',
    'rr': 'Rajasthan Royals',
    'lsg': 'Lucknow Super Giants'
}

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')]
urllib.request.install_opener(opener)

for code, team in teams.items():
    try:
        url = f'https://en.wikipedia.org/w/api.php?action=query&titles={team.replace(" ", "%20")}&prop=pageimages&format=json&pithumbsize=500'
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            pages = data['query']['pages']
            for page_id in pages:
                if 'thumbnail' in pages[page_id]:
                    img_url = pages[page_id]['thumbnail']['source']
                    print(f'{code}: {img_url}')
                    ext = img_url.split('.')[-1]
                    urllib.request.urlretrieve(img_url, f'static/logos/{code}.{ext}')
                else:
                    print(f'{code}: NO_IMAGE')
    except Exception as e:
        print(f'{code}: ERROR {e}')
