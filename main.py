import os

def main():
    print("GitHub Actionsの定期実行テスト成功！")

    imgbb_key = os.environ.get("IMGBB_API_KEY", "")
    line_token = os.environ.get("LINE_ACCESS_TOKEN", "")

    if imgbb_key and line_token:
        print("✅ 環境変数の読み込みに成功しました！")
    else:
        print("❌ 環境変数が設定されていません。Secretsを確認してください。")

if __name__ == "__main__":
    main()
