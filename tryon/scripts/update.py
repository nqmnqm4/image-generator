"""ユニクロ・GU 試着ボード用の商品データと切り抜き画像を作るスクリプト。

GitHub Actions で毎日実行し、tryon/data/products.json と tryon/img/ を更新する。
- 商品一覧は各サイトの公開API（サイトの商品一覧ページが使っているもの）から取得
- 画像は背景を切り抜いて webp で保存（すでにある画像は作り直さない）
- 丈の目安は、モデル写真（身長がAPIで分かる）から服の上端・下端を測って出す
"""
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
IMG = os.path.join(ROOT, "img")
CACHE = os.path.join(DATA, "measure_cache.json")
OUT = os.path.join(DATA, "products.json")

PER_CLASS = int(os.environ.get("TRYON_PER_CLASS", "8"))
MAX_COLORS = 6
UA = "Mozilla/5.0 (compatible; tryon-board/1.0; personal use)"

BRANDS = [
    {
        "key": "UQ", "label": "UNIQLO", "host": "https://www.uniqlo.com", "gender": 1071,
        "classes": [
            (1479, "top", "アウター"), (94385, "top", "シャツ・ブラウス"),
            (1480, "top", "Tシャツ・スウェット"), (94381, "top", "ニット"),
            (1481, "bottom", "パンツ"), (51226, "mix", "スカート・ワンピース"),
        ],
    },
    {
        "key": "GU", "label": "GU", "host": "https://www.gu-global.com", "gender": 2256,
        "classes": [
            (36368, "top", "アウター"), (94706, "top", "ニット"),
            (36369, "top", "Tシャツ"), (125267, "top", "スウェット"),
            (94697, "top", "シャツ・ブラウス"), (36371, "bottom", "パンツ"),
            (36372, "bottom", "スカート"), (36370, "dress", "ワンピース"),
        ],
    },
]
SKIP_WORDS = ["ソックス", "タイツ", "レギンス", "インナー", "ブラ", "ショーツ", "キャミソール",
              "マスク", "チャーム", "バッグ", "帽子", "キャップ", "マフラー", "手袋", "パジャマ"]


def get(url, binary=False, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=40) as r:
                body = r.read()
            time.sleep(0.4)
            return body if binary else json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (i + 1))
        except Exception:
            time.sleep(2 * (i + 1))
    return None


# ---------------- segmentation ----------------
_sessions = {}


def session(name):
    if name not in _sessions:
        from rembg import new_session
        _sessions[name] = new_session(name)
    return _sessions[name]


def rows_extent(mask, frac=0.2):
    r = mask.sum(1)
    if r.max() == 0:
        return None
    ys = np.where(r > r.max() * frac)[0]
    return int(ys.min()), int(ys.max())


def cloth_masks(im):
    """u2net_cloth_seg: 上半身・下半身・全身の3枚が縦に並んだマスクを返す"""
    from rembg import remove
    out = np.array(remove(im, session=session("u2net_cloth_seg"), only_mask=True))
    ch = out[..., 0] if out.ndim == 3 else out
    h = im.height
    return {k: ch[i * h:(i + 1) * h] > 128 for i, k in enumerate(["upper", "lower", "full"])}


def person_mask(im):
    from rembg import remove
    m = np.array(remove(im, session=session("u2net_human_seg"), only_mask=True))
    if m.ndim == 3:
        m = m[..., 0]
    return m > 128


def garment_mask(masks, kind):
    if kind == "top":
        return masks["upper"]
    if kind == "bottom":
        return masks["lower"]
    full = masks["full"]
    if full.sum() > 0.02 * full.size:
        return full
    return masks["upper"] | masks["lower"]


def measure(model_img, kind, model_cm):
    """モデル写真から、服の上端と裾の位置（頭からかかとまでの比率）を測る。全身が写っていなければ None"""
    im = model_img.convert("RGB")
    H = im.height
    pm = person_mask(im)
    ext = rows_extent(pm, 0.02)
    if not ext:
        return None
    head, feet = ext
    if head < H * 0.01 or feet > H * 0.995 or (feet - head) < H * 0.6:
        return None  # 顔か足元が切れている
    gm = garment_mask(cloth_masks(im), kind)
    g = rows_extent(gm, 0.2)
    if not g:
        return None
    body = feet - head
    top_frac = (g[0] - head) / body
    hem_frac = (g[1] - head) / body
    if not (0 <= top_frac < 0.8 and 0.1 < hem_frac - top_frac and hem_frac <= 1.02):
        return None
    # topFrac: 見えている服の上端（頭=0, かかと=1）, hemFrac: 裾の位置
    return {"topFrac": round(top_frac, 4), "hemFrac": round(hem_frac, 4)}


def save_cut(rgba, path):
    a = np.array(rgba)
    alpha = a[..., 3]
    ys, xs = np.where(alpha > 20)
    if len(ys) < 500:
        return False
    crop = rgba.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    crop.thumbnail((700, 900))
    crop.save(path, "WEBP", quality=82, method=6)
    return True


def cut_flat(img):
    from rembg import remove
    return remove(img.convert("RGB"), session=session("u2net"))


def cut_from_model(img, kind):
    im = img.convert("RGB")
    m = garment_mask(cloth_masks(im), kind)
    rgba = im.convert("RGBA")
    a = np.array(rgba)
    a[..., 3] = np.where(m, 255, 0).astype(np.uint8)
    return Image.fromarray(a)


def load_img(url):
    b = get(url, binary=True)
    if not b:
        return None
    try:
        return Image.open(io.BytesIO(b))
    except Exception:
        return None


# ---------------- main ----------------
def product_kind(cls_kind, name):
    if cls_kind == "mix":
        return "dress" if ("ワンピース" in name or "ドレス" in name) else "bottom"
    return cls_kind


def list_products(brand, cls_id):
    q = urllib.parse.urlencode({"path": f"{brand['gender']},{cls_id},,", "limit": 36, "offset": 0})
    j = get(f"{brand['host']}/jp/api/commerce/v5/ja/products?{q}")
    if not j or j.get("status") != "ok":
        return []
    return j["result"].get("items", [])


def model_info(brand, item):
    j = get(f"{brand['host']}/jp/api/commerce/v5/ja/products/{item['productId']}/price-groups/"
            f"{item.get('priceGroup', '00')}/details?includeModelSize=true&httpFailure=true")
    if not j or j.get("status") != "ok":
        return None
    res = j["result"]
    sizes = {s.get("code"): s.get("name") for s in res.get("sizes", [])}
    for c, v in (res.get("images", {}).get("main") or {}).items():
        for m in v.get("model") or []:
            h = ((m.get("modelSize") or {}).get("universal") or {}).get("height")
            if h:
                code = m.get("sizeCode") or m.get("size") or ""
                if isinstance(code, dict):
                    code = code.get("code", "")
                return {"color": c, "image": v.get("image"), "heightCm": float(h),
                        "size": sizes.get(code, code if isinstance(code, str) else "")}
    return None


def main():
    os.makedirs(IMG, exist_ok=True)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    products, used = [], set()
    stats = {"items": 0, "flat": 0, "model": 0, "measured": 0}

    for brand in BRANDS:
        seen = set()
        for cls_id, cls_kind, cls_label in brand["classes"]:
            count = 0
            for it in list_products(brand, cls_id):
                if count >= PER_CLASS:
                    break
                name = it.get("name", "")
                l1 = it.get("l1Id")
                if not l1 or l1 in seen:
                    continue
                if it.get("genderCategory") in ("MEN", "KIDS", "BABY"):
                    continue
                if any(w in name for w in SKIP_WORDS):
                    continue
                main_imgs = (it.get("images") or {}).get("main") or {}
                if not main_imgs:
                    continue
                kind = product_kind(cls_kind, name)
                key = f"{brand['key']}_{l1}"

                # 丈の測定（商品ごとに1回、結果はキャッシュ）
                if key not in cache:
                    mi = model_info(brand, it)
                    meas = None
                    if mi and mi.get("image"):
                        mimg = load_img(mi["image"])
                        if mimg is not None:
                            try:
                                meas = measure(mimg, kind, mi["heightCm"])
                            except Exception as e:
                                print("measure error", key, e, file=sys.stderr)
                    cache[key] = {"meas": meas, "modelCm": mi and mi["heightCm"], "modelSize": mi and mi["size"]}

                colors = []
                for c in list(main_imgs.keys())[:MAX_COLORS]:
                    fn = f"{key}_{c}.webp"
                    path = os.path.join(IMG, fn)
                    src = cache.get(key, {}).get("src", {}).get(c)
                    if not os.path.exists(path):
                        flat_url = (f"https://image.uniqlo.com/{brand['key']}/ST3/AsianCommon/imagesgoods/"
                                    f"{l1}/item/goods_{c}_{l1}_3x4.jpg")
                        ok = False
                        fimg = load_img(flat_url)
                        if fimg is not None:
                            try:
                                ok = save_cut(cut_flat(fimg), path)
                                src = "flat" if ok else None
                            except Exception as e:
                                print("flat error", fn, e, file=sys.stderr)
                        if not ok:
                            mimg = load_img(main_imgs[c]["image"])
                            if mimg is not None:
                                try:
                                    ok = save_cut(cut_from_model(mimg, kind), path)
                                    src = "model" if ok else None
                                except Exception as e:
                                    print("model cut error", fn, e, file=sys.stderr)
                        if not ok:
                            continue
                        cache.setdefault(key, {}).setdefault("src", {})[c] = src
                    used.add(fn)
                    chip = (f"https://image.uniqlo.com/{brand['key']}/ST3/AsianCommon/imagesgoods/"
                            f"{l1}/chip/goods_{c}_{l1}_chip.jpg")
                    colors.append({"code": c, "img": f"img/{fn}", "chip": chip, "src": src})
                    stats[src or "flat"] = stats.get(src or "flat", 0) + 1
                if not colors:
                    continue

                prices = it.get("prices") or {}
                price = ((prices.get("promo") or {}).get("value")) or ((prices.get("base") or {}).get("value"))
                c0 = cache.get(key, {})
                if c0.get("meas"):
                    stats["measured"] += 1
                products.append({
                    "id": key, "brand": brand["key"], "brandLabel": brand["label"],
                    "name": name, "kind": kind, "category": cls_label, "price": price,
                    "url": f"{brand['host']}/jp/ja/products/{it['productId']}/{it.get('priceGroup', '00')}",
                    "meas": c0.get("meas"), "modelCm": c0.get("modelCm"), "modelSize": c0.get("modelSize"),
                    "colors": colors,
                })
                seen.add(l1)
                count += 1
                stats["items"] += 1
            print(brand["key"], cls_label, count, flush=True)

    # 掲載が終わった商品の画像を消す
    for fn in os.listdir(IMG):
        if fn.endswith(".webp") and fn not in used:
            os.remove(os.path.join(IMG, fn))

    jst = timezone(timedelta(hours=9))
    if not products:
        print("no products fetched; keep previous data", file=sys.stderr)
        sys.exit(1)
    json.dump({"updated": datetime.now(jst).strftime("%Y-%m-%d %H:%M"), "products": products},
              open(OUT, "w"), ensure_ascii=False, indent=1)
    json.dump(cache, open(CACHE, "w"), ensure_ascii=False, indent=1)
    print("stats", stats)


if __name__ == "__main__":
    main()
