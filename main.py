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

# ==========================================
# 0. Secrets（環境変数）から各種キーを取得
# ==========================================
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
        await asyncio.sleep(0.5)
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
    semaphore = asyncio.Semaphore(3)
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_article_details(session, url, semaphore) for url in news_urls]
        results = await asyncio.gather(*tasks)
        
        for res in results:
            if res and len(collected) < max_count:
                collected.append(res)
                
    return collected

# ==========================================
# 4. AI分析 ＆ 3枚のスライド用HTML生成（JSON出力）
# ==========================================
def analyze_macro_market_for_slide(articles):
    if not articles:
        return None

    formatted_input = ""
    for idx, item in enumerate(articles, 1):
        formatted_input += f"■ 記事[{idx}] ({item['published_at']})\nタイトル: {item['title']}\n本文:\n{item['body'][:800]}\n---\n"

    prompt = f"""あなたはプロのスイングトレーダー向けに市況解説を行うレポートアナリストです。
提供された市況ニュース（マクロ・クロスマセット情報）を分析し、スマホ閲覧に最適化された**3枚のスライドカード用HTML**を作成してください。

出力は必ず以下のJSONフォーマットのみとし、他のテキストやコードブロック（```json）は含めないでください。

{{
  "card1_html": "<!DOCTYPE html>...",
  "card2_html": "<!DOCTYPE html>...",
  "card3_html": "<!DOCTYPE html>..."
}}

【共通デザイン要件】
- 幅800px、高さ1200px（アスペクト比 2:3）の固定枠に美しく収まるレイアウト（`box-sizing: border-box; padding: 40px;`）。
- 背景色：ダークネイビー（#0f172a）、カード背景：(#1e293b)、テキスト：(#f8fafc)。
- タイトルや重要箇所の視認性を高める装飾（グラデーション、バッジ等）を施すこと。
- フォントサイズは大きめ（本文18px〜20px、見出し28px以上）。

【各カードの内容構成】
■ カード1：マクロ環境と地合い（全体概況）
- 世界・日本のマクロ動向（インフレ、金利、為替、先物等）と市場心理（リスクオン/オフ）。
- 事象の「背景（なぜそうなったか）」をロジカルに記載。

■ カード2：物色テーマと注目銘柄のロジック（市場の動き）
- 市況ニュースで取り上げられている「物色テーマ（決算優良株、成長株、資源等）」や特定銘柄の動き。
- 単なるセクター分けではなく、資金が向かっている「根拠と動機」を整理。

■ カード3：スイング戦略へのインサイト（立ち回り）
- テクニカル（節目、移動平均線）とファンダメンタルズ（主要指標・イベント）の交差点。
- 翌営業日以降の具体的な立ち回り方針・リスク管理。

【インプットデータ】
{formatted_input}
"""

    print("AIにリクエストを送信中（gemini-3.5-flash / JSON形式で3カード分取得）...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(
        temperature=0.3,
        response_mime_type="application/json"
    )
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt,
        config=config
    )
    
    try:
        data = json.loads(response.text.strip())
        return [data["card1_html"], data["card2_html"], data["card3_html"]]
    except Exception as e:
        print(f"❌ JSONパースエラー: {e}")
        return None

# ==========================================
# 5. Playwrightで3枚の画像を個別レンダリング（描画・空画像対策強化）
# ==========================================
async def generate_slide_images(html_contents):
    print("Playwrightでスライド画像（3枚）を生成中...")
    image_paths = []
    
    font_injection = """
    <style>
    @import url('[https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap](https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap)');
    body { font-family: 'Noto Sans JP', sans-serif !important; margin: 0; padding: 0; background-color: #0f172a; }
    </style>
    """

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 800, 'height': 1200})
        
        for idx, html in enumerate(html_contents, 1):
            if "</head>" in html
