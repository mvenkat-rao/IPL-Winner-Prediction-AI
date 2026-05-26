import wikipedia
import urllib.request
import os

teams = {
    'rcb': 'Royal Challengers Bengaluru',
    'srh': 'Sunrisers Hyderabad',
    'rr': 'Rajasthan Royals',
    'lsg': 'Lucknow Super Giants'
}

os.makedirs('static/logos', exist_ok=True)
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
urllib.request.install_opener(opener)

for code, name in teams.items():
    try:
        page = wikipedia.page(name, auto_suggest=False)
        logo_url = None
        for img in page.images:
            if img.lower().endswith('.svg') or img.lower().endswith('.png'):
                if 'logo' in img.lower() or 'crest' in img.lower():
                    logo_url = img
                    break
        if not logo_url and page.images:
            for img in page.images:
                if img.lower().endswith('.svg') or img.lower().endswith('.png'):
                    logo_url = img
                    break
                    
        if logo_url:
            ext = logo_url.split('.')[-1]
            urllib.request.urlretrieve(logo_url, f'static/logos/{code}.{ext}')
            print(f"Downloaded {code} from {logo_url}")
        else:
            print(f"No suitable image found for {code}")
    except Exception as e:
        print(f"Failed {code}: {e}")
