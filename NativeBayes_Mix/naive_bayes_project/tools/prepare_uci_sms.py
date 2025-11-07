# -*- coding: utf-8 -*-
import os, io, csv, zipfile, argparse, urllib.request, tempfile, sys

DEFAULT_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
]

def download(url, to_path):
    if url.startswith("file:///"):
        src = url[len("file:///"):]
        with open(src, "rb") as fsrc, open(to_path, "wb") as fdst:
            fdst.write(fsrc.read())
    else:
        urllib.request.urlretrieve(url, to_path)

def extract_raw_text(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            if name.lower().endswith("smsspamcollection"):
                return zf.read(name).decode('utf-8', errors='ignore')
    raise FileNotFoundError("未在 zip 中找到 SMSSpamCollection")

def convert_to_csv(raw_text, out_csv):
    lines = raw_text.splitlines()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', encoding='utf-8', newline='') as fout:
        w = csv.writer(fout); w.writerow(['label','text'])
        for line in lines:
            line = line.strip()
            if not line: continue
            lab, txt = None, None
            if '\t' in line:
                lab, txt = line.split('\t', 1)
            else:
                parts = line.split(',', 1)
                if len(parts) == 2 and parts[0].lower() in ('spam','ham'):
                    lab, txt = parts
            if lab is None or txt is None: continue
            w.writerow([lab.strip().lower(), txt.strip()])
    return out_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', type=str, default=None, help='UCI zip 的 URL，或 file:///本地路径')
    ap.add_argument('--out', required=True, help='输出 CSV 路径')
    args = ap.parse_args()

    urls = [args.url] if args.url else DEFAULT_URLS
    with tempfile.TemporaryDirectory() as td:
        zip_path = os.path.join(td, "smsspamcollection.zip")
        ok = False
        for u in urls:
            try: download(u, zip_path); ok=True; break
            except Exception as e: print("下载失败：", e, file=sys.stderr); continue
        if not ok: raise SystemExit("下载失败，请手动下载 zip 并用 --url file:///路径 指定本地文件")
        raw = extract_raw_text(zip_path)
    out_csv = convert_to_csv(raw, args.out)
    print("已生成：", out_csv)

if __name__ == '__main__':
    main()
