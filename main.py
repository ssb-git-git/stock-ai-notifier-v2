import os
import json
import urllib.request
import time
import re
import asyncio
import aiohttp
import requests
import base64
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from playwright.async_api import async_playwright

# --------------------------------------------------
# Secrets（環境変数）から各種キーを取得
# --------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
IMGBB_API_KEY = os.environ.get("IMGBB_API_KEY")
LINE_ACCESS_TOKEN = os.environ.get("LINE_ACCESS_TOKEN")
LINE_DESTINATION_ID = os.environ.get("LINE_DESTINATION_ID")

# ==========================================
# 1. 設定情報
# ==========================================
TARGET_SOURCES = ["フィスコ", "ロイター"]
MAX_ARTICLES = 10
BASE_URL = "https://finance.yahoo.co.jp/news/market"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

PAID_ARTICLE_KEYWORDS = [
    "続きをお読みいただくには", "VIP倶楽部", "有料会員限定", "この記事は会員限定です"
]

# ==========================================
# 2. ヘルパー関数
# ==========================================
def clean_body_text(text):
    if not text: return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = [l for l in lines if not any(x in l for x in ["ログインしてポートフォリオを表示", "情報提供会社のリンクは", "最終更新:"])]
    return "\n".join(cleaned)

def extract_published_time(soup):
    for meta_name in ["article:published_time", "pubdate", "parsely-pub-date"]:
        meta = soup.find("meta", {"property": meta_name}) or soup.find("meta", {"name": meta_name})
        if meta and meta.get("content"):
            return meta["content"].strip()
    text = soup.get_text()
    match = re.search(r'\d{1,2}/\d{1,2}\([月火水木金土日]\)\s+\d{1,2}:\d{2}', text)
    if match: return match.group(0)
    return "時間不明"

# ==========================================
# 3. ニュース収集ロジック（非同期処理）
# ==========================================
async def fetch_article_details(session, url, semaphore):
    """個別の記事ページにアクセスして本文を抽出（並列処理用）"""
    async with semaphore:  # 同時アクセス数を制限してサーバー負荷を下げる
        await asyncio.sleep(0.5) # 最低限の礼儀としてのディレイ
        try:
            async with session.get(url, headers=HEADERS) as resp:
                if resp.status != 200:
                    return None
                html = await resp.text()
                soup = BeautifulSoup(html, "html.parser")
                page_text = soup.get_text()

                source_found = next((s for s in TARGET_SOURCES if s in page_text), None)
                if not source_found: return None

                title_elem = soup.find("h1")
                title_text = title_elem.get_text().strip() if title_elem else "タイトル不明"
                pub_time = extract_published_time(soup)

                body_text = ""
                script_tag = soup.find("script", id="__NEXT_DATA__")
                if script_tag and script_tag.string:
                    try:
                        data = json.loads(script_tag.string)
                        def find_body(d):
                            if isinstance(d, dict):
                                if "body" in d and isinstance(d["body"], str) and len(d["body"]) > 50:
                                    return d["body"]
                                for k, v in d.items():
                                    res = find_body(v)
                                    if res: return res
                            elif isinstance(d, list):
                                for item in d:
                                    res = find_body(item)
                                    if res: return res
                            return None
                        extracted = find_body(data)
                        if extracted:
                            body_text = BeautifulSoup(extracted, "html.parser").get_text(separator='\n').strip()
                    except:
                        pass

                if not body_text:
                    paragraphs = soup.find_all("p")
                    body_text = "\n".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20 and "JavaScript" not in p.get_text()])

                if any(k in body_text for k in PAID_ARTICLE_KEYWORDS):
                    return None

                clean_body = clean_body_text(body_text)
                if clean_body:
                    print(f"取得成功 ({source_found}): {title_text[:20]}...")
                    return {
                        "title": title_text, "source": source_found,
                        "published_at": pub_time, "url": url, "body": clean_body
                    }
        except Exception:
            return None
    return None

async def get_market_news_async(max_count=10):
    print("ニュースのURL一覧を収集中...")
    news_urls = []
    
    # URL一覧は同期的にサクッと取得
    for page in range(1, 4):
        resp = requests.get(f"{BASE_URL}?page={page}", headers=HEADERS)
        if resp.status_code != 200: break
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/news/detail/" in href:
                full_url = href if href.startswith("http") else f"https://finance.yahoo.co.jp{href}"
                if full_url not in news_urls:
                    news_urls.append(full_url)
        if len(news_urls) >= max_count * 2: break

    print(f"{len(news_urls)}件の候補URLから、非同期で本文を抽出します...")
    
    collected = []
    # 同時アクセスを最大3つまでに制限
    semaphore = asyncio.Semaphore(3)
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_article_details(session, url, semaphore) for url in news_urls]
        results = await asyncio.gather(*tasks)
        
        for res in results:
            if res and len(collected) < max_count:
                collected.append(res)
                
    return collected

# ==========================================
# 4. AI分析＆スライド用HTML生成
# ==========================================
def analyze_macro_market_for_slide(articles):
    if not articles:
        return None

    formatted_input = ""
    for idx, item in enumerate(articles, 1):
        formatted_input += f"■ 記事[{idx}] ({item['published_at']})\nタイトル: {item['title']}\n本文:\n{item['body'][:800]}\n---\n"

    prompt = f"""あなたはプロの個人投資家（スイングトレーダー）向けに、相場解説レポートを作成するアナリストです。
提供された市況ニュースを分析し、スマートフォンでの閲覧に最適化された**縦長1枚のHTMLレポート**として出力してください。

【出力要件：デザイン】
- スマホ閲覧用のため、横並び（カラム分け）は絶対にせず、全て「縦1列（flex-direction: column）」で構成すること。
- 文字サイズは大きく（基準を24px程度）、余白（padding）を十分に取り、見出しを強調すること。
- 背景色はダークネイビー（#1e293b）、文字は白や明るい色を基調とすること。
- 要素の高さを固定（height: 100vh等）しないこと。コンテンツ量に合わせて縦に伸びるようにすること。
- 直接 `<!DOCTYPE html>` から出力し、マークダウンやコードブロックは含めないこと。

【出力要件：情報量と分析の深さ】
以下の構成で、システムトレードの判断材料となるよう、事象の背景（なぜ？）を省略せずに詳細に記載すること。
1. 本日の相場サマリー（タイトル）
2. マクロ環境と地合い（全体がリスクオンかオフか。為替や金利などのファンダメンタルズ要因がどう影響したか）
3. 資金流入・流出セクターの詳細（どの業種が買われ/売られているか。単なる事実だけでなく、ニュースに基づく論理的な背景を厚めに書く）
4. スイング戦略へのインサイト（目先のリスクイベントや、テクニカル・ファンダメンタルズ両面からの具体的な立ち回り方針）

【インプットデータ】
{formatted_input}
"""

    print("AIにリクエストを送信中（HTML縦長レポート生成）...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(temperature=0.4)
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt,
        config=config
    )
    
    html_out = response.text.strip()
    if html_out.startswith("```html"):
        html_out = html_out[7:-3]
    return html_out.strip()

# ==========================================
# 5. PlaywrightでHTMLを画像化
# ==========================================
async def generate_slide_image(html_content, output_path="slide.png"):
    print("Playwrightでスライド画像を生成中...")
    
    font_injection = """
    <style>
    @import url('[https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap](https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap)');
    body { font-family: 'Noto Sans JP', sans-serif !important; margin: 0; padding: 0; }
    </style>
    """
    if "</head>" in html_content:
        html_content = html_content.replace("</head>", f"{font_injection}</head>")
    else:
        html_content = f"{font_injection}{html_content}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # スマホの画面幅をシミュレート（縦幅はfull_pageで自動拡張されるため仮の数値でOK）
        page = await browser.new_page(viewport={'width': 800, 'height': 1200})
        await page.set_content(html_content, wait_until="networkidle")
        
        # 画面の途中で切れないよう、ページ全体のフルスクリーンショットを撮る
        await page.screenshot(path=output_path, full_page=True)
        await browser.close()
    print("画像を保存しました:", output_path)

# ==========================================
# 6. ImgBBへのアップロード（時限消去付き）
# ==========================================
def upload_to_imgbb(image_path):
    print("ImgBBへ画像をアップロード中...")
    with open(image_path, "rb") as file:
        payload = {
            "key": IMGBB_API_KEY,
            "image": base64.b64encode(file.read()),
            "expiration": 3600  # 1時間（3600秒）後に自動削除
        }
        res = requests.post("https://api.imgbb.com/1/upload", data=payload)
        
        if res.status_code == 200:
            url = res.json()["data"]["url"]
            print(f"アップロード成功: {url}")
            return url
        else:
            print("ImgBBアップロード失敗:", res.text)
            return None

# ==========================================
# 7. LINE画像送信
# ==========================================
def send_line_image(image_url):
    url = "https://api.line.me/v2/bot/message/push"
    payload = {
        "to": LINE_DESTINATION_ID,
        "messages": [
            {
                "type": "image",
                "originalContentUrl": image_url,
                "previewImageUrl": image_url
            }
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                print("✅ LINEへの画像送信に成功しました！")
    except Exception as e:
        print(f"❌ LINE送信エラー: {e}")

# ==========================================
# メイン実行 (非同期イベントループ)
# ==========================================
async def main():
    if not all([GEMINI_API_KEY, IMGBB_API_KEY, LINE_ACCESS_TOKEN, LINE_DESTINATION_ID]):
        print("❌ 環境変数が不足しています。Secretsの設定を確認してください。")
        return

    # 1. ニュース取得（非同期）
    articles = await get_market_news_async(max_count=MAX_ARTICLES)
    
    # 2. AI分析 ＆ HTML生成
    html_content = analyze_macro_market_for_slide(articles)
    if not html_content: return

    # 3. HTMLを画像化
    image_path = "slide.png"
    await generate_slide_image(html_content, image_path)

    # 4. ImgBBへアップロード
    image_url = upload_to_imgbb(image_path)

    # 5. LINEへ送信
    if image_url:
        send_line_image(image_url)

if __name__ == "__main__":
    asyncio.run(main())
