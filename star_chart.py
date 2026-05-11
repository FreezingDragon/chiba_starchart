#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
star_chart.py
正距方位図法（天頂中心）による星図PDF生成
  北上・東左（空を見上げる向き）
  恒星・星座線・星座名（日本語）・惑星・月・天の川

使い方:
  python star_chart.py --date 2026-05-08 --time 21:00 \
      --lat 35.664 --lon 139.990 --location "ふなばし三番瀬"
  python star_chart.py --gui
"""

import os, sys, json, argparse, warnings, math
import numpy as np
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Polygon as MplPolygon
import matplotlib.font_manager as fm
from datetime import datetime, timezone, timedelta

# ═══════════════════════════════════════════════════════
#  パス設定
# ═══════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'star_chart_data')

# skyfield が de421.bsp を探すディレクトリをDATA_DIRに固定
os.makedirs(DATA_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════
#  天体色 / 惑星設定
# ═══════════════════════════════════════════════════════
PLANET_TABLE = {
    'mercury':            ('水星', '#c0c0c0', 5.0),
    'venus':              ('金星', '#fff8a0', 7.0),
    'mars':               ('火星', '#ff6633', 6.0),
    'jupiter barycenter': ('木星', '#ffd8a0', 6.5),
    'saturn barycenter':  ('土星', '#ffe870', 6.0),
    'uranus barycenter':  ('天王星','#a0ffff', 4.5),
    'neptune barycenter': ('海王星','#8888ff', 4.0),
}

# ═══════════════════════════════════════════════════════
#  日本語フォント
# ═══════════════════════════════════════════════════════
_JP_FONT = None

def get_jp_font(size=9):
    global _JP_FONT
    if _JP_FONT is None:
        # matplotlibのフォントマネージャーで検索するキーワード（優先順）
        candidates = [
            'hiragino',        # macOS: ヒラギノ
            'yu gothic',       # macOS/Windows: 游ゴシック
            'yugothic',
            'bizudgothic',     # BIZ UDゴシック
            'bizud',
            'applegothic',     # macOS: Apple Gothic
            'apple gothic',
            'apple sd gothic',
            'noto sans cjk',   # Linux: Noto CJK
            'ipagothic',       # Linux/macOS: IPA Gothic
            'ipamj',           # IPA明朝
            'meiryo',          # Windows: メイリオ
            'msgothic',        # Windows: MSゴシック
            'ms gothic',
            'yumin',           # 游明朝
        ]
        import matplotlib.font_manager as fm2
        for fname in fm2.findSystemFonts():
            try:
                fp = fm2.FontProperties(fname=fname)
                n = fp.get_name().lower()
                if any(x in n for x in candidates):
                    _JP_FONT = fname
                    break
            except:
                pass

        # matplotlibで見つからない場合はmacOSのフォントディレクトリを直接スキャン
        if _JP_FONT is None:
            import os as _os
            direct_search = [
                # TTFを優先（matplotlibとの相性が良い）
                '/Users/{}/Library/Fonts/ipamjm.ttf',
                '/Users/{}/Library/Fonts/yumin.ttf',
                '/Users/{}/Library/Fonts/YuGothR.ttc',
                '/Users/{}/Library/Fonts/BIZ-UDGothicR.ttc',
                '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
                '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
                '/System/Library/Fonts/Hiragino Sans.ttc',
            ]
            import getpass
            user = getpass.getuser()
            for tmpl in direct_search:
                path = tmpl.format(user)
                if _os.path.exists(path):
                    _JP_FONT = path
                    break

    if _JP_FONT:
        fp = fm.FontProperties(fname=_JP_FONT, size=size)
        # rcParamsにフォントファミリーを登録しておく（凡例・軸ラベル等に効かせるため）
        family = fp.get_name()
        if family and family not in plt.rcParams['font.family']:
            fm.fontManager.addfont(_JP_FONT)
            plt.rcParams['font.family'] = [family, 'DejaVu Sans']
        return fp
    return fm.FontProperties(size=size)

# ═══════════════════════════════════════════════════════
#  座標変換ユーティリティ
# ═══════════════════════════════════════════════════════

def altaz_to_xy(alt_deg, az_deg, R=1.0):
    """高度・方位角 → 星図XY（天頂中心・北上・東左）。地平線以下はNaN。"""
    alt = np.asarray(alt_deg, dtype=float)
    az  = np.asarray(az_deg,  dtype=float)
    r   = (90.0 - alt) / 90.0 * R
    r   = np.where(alt >= 0, r, np.nan)
    az_rad = np.radians(az)
    x = -r * np.sin(az_rad)   # 東を左
    y =  r * np.cos(az_rad)   # 北を上
    return x, y


def compute_lst_deg(dt_utc: datetime, lon_deg: float) -> float:
    """UTC datetime → Local Sidereal Time（度）"""
    J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    d = (dt_utc.replace(tzinfo=timezone.utc) - J2000).total_seconds() / 86400.0
    gmst = (280.46061837 + 360.98564736629 * d) % 360.0
    return (gmst + lon_deg) % 360.0


def radec_to_altaz_np(ra_deg, dec_deg, lat_deg, lon_deg, dt_utc):
    """numpy一括変換: 赤経(度)・赤緯(度) → 高度・方位角(度)"""
    lst = compute_lst_deg(dt_utc, lon_deg)
    H   = np.radians(lst - np.asarray(ra_deg, float))
    lat = np.radians(lat_deg)
    dec = np.radians(np.asarray(dec_deg, float))

    sin_alt = np.sin(dec)*np.sin(lat) + np.cos(dec)*np.cos(lat)*np.cos(H)
    sin_alt = np.clip(sin_alt, -1.0, 1.0)
    alt     = np.arcsin(sin_alt)
    cos_alt = np.cos(alt)

    denom_az = np.where(np.abs(cos_alt) > 1e-9, cos_alt, 1e-9)
    denom_la = np.where(np.abs(cos_alt * np.cos(lat)) > 1e-9,
                        cos_alt * np.cos(lat), 1e-9)
    sin_az   = -np.sin(H) * np.cos(dec) / denom_az
    cos_az   = (np.sin(dec) - np.sin(alt)*np.sin(lat)) / denom_la

    az = np.degrees(np.arctan2(sin_az, cos_az)) % 360.0
    return np.degrees(alt), az


def d3ra_to_deg(ra_d3: float) -> float:
    """d3-celestial RA（-180…180°）→ 通常の赤経（0…360°）"""
    return ra_d3 % 360.0


# ═══════════════════════════════════════════════════════
#  B-V → 星の色
# ═══════════════════════════════════════════════════════

def ballesteros_color(bv):
    """Ballesteros (2012) 式: B-V → 色温度 → RGB
    T = 4600 × (1/(0.92×BV+1.7) + 1/(0.92×BV+0.62))
    RGB: Tanner Helland (2012) 黒体近似
    検証: Vega(BV=-0.01)→10250K→青白, Sun(BV=0.65)→5780K, Arcturus(BV=1.23)→4250K→橙
    """
    try:
        bv = float(bv)
        if math.isnan(bv):
            bv = 0.0
    except:
        bv = 0.0
    bv = max(-0.40, min(2.00, bv))

    # Ballesteros 色温度
    T = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))
    T = max(1000.0, min(40000.0, T))
    t = T / 100.0

    # 赤チャンネル
    R = 255.0 if T < 6600.0 else 329.698727 * math.pow(t - 60.0, -0.1332047592)
    # 緑チャンネル
    G = (99.4708025979 * math.log(t) - 161.1195681661) if T < 6600.0 \
        else 288.1221695283 * math.pow(t - 60.0, -0.0755148492)
    # 青チャンネル
    if   T <= 1900.0: B = 0.0
    elif T <  6600.0: B = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    else:             B = 255.0

    return (
        max(0.0, min(1.0, R / 255.0)),
        max(0.0, min(1.0, G / 255.0)),
        max(0.0, min(1.0, B / 255.0)),
    )


def mag_to_radius(mag):
    """等級 → プロット半径。1等差ごとに×0.6の比率。
    1等: 0.013 / 2等: 0.0078 / 3等: 0.0047 / 5等: 0.0017  (1等/5等 ≈ 7.7倍)
    """
    r = 0.013 * (0.6 ** (float(mag) - 1.0))
    return float(np.clip(r, 0.0010, 0.028))

# ═══════════════════════════════════════════════════════
#  データロード
# ═══════════════════════════════════════════════════════

def load_hip_catalog(mag_limit=6.5):
    """ヒッパルコス星表 → DataFrame {hip, magnitude, ra_deg, dec_deg, bv}"""
    import pandas as pd

    hip_path = os.path.join(DATA_DIR, 'hip_main.dat')
    if not os.path.exists(hip_path):
        raise FileNotFoundError(f"Hipparcos catalog not found: {hip_path}")

    rows = []
    with open(hip_path) as f:
        for line in f:
            fields = line.split('|')
            if len(fields) < 41:
                continue
            try:
                hip  = int(fields[1].strip())
                mag  = float(fields[5].strip())
                ra   = float(fields[8].strip())   # 赤経（度）
                dec  = float(fields[9].strip())   # 赤緯（度）
                bv_s = fields[40].strip()
                bv   = float(bv_s) if bv_s else np.nan
                if mag <= mag_limit:
                    rows.append((hip, mag, ra, dec, bv))
            except:
                pass

    df = pd.DataFrame(rows, columns=['hip','magnitude','ra_deg','dec_deg','bv'])
    df.set_index('hip', inplace=True)
    return df


def load_constellation_data():
    """d3-celestialデータ読み込み"""
    # 日本標準式（sternenkarten.com / IAU準拠）を優先、なければd3-celestialにフォールバック
    jp_lines = os.path.join(DATA_DIR, 'constellations.lines.jp.json')
    lines_path  = jp_lines if os.path.exists(jp_lines) else os.path.join(DATA_DIR, 'constellations.lines.json')
    labels_path = os.path.join(DATA_DIR, 'constellations.json')
    with open(lines_path,  encoding='utf-8') as f:
        lines_data  = json.load(f)
    with open(labels_path, encoding='utf-8') as f:
        labels_data = json.load(f)
    return lines_data, labels_data


def load_milkyway_data():
    # mw.json は現在未使用（Stellariumテクスチャに移行済み）
    return {}

def _ensure_gaia_milkyway_data():
    """天の川レンダリング用データを初回生成・キャッシュ
    ① Stellarium milkyway.png → ガウスブラー処理済み RGBA .npy
    ② Gaia DR3 ag_gspphot (nside=128) — 消光マスク用
    ③ 指数円盤モデル輝度 (nside=128) — 補助用
    """
    import urllib.request
    from scipy.ndimage import gaussian_filter
    from PIL import Image

    mw_png  = os.path.join(DATA_DIR, 'milkyway.png')
    mw_npy  = os.path.join(DATA_DIR, 'milkyway_blurred.npy')
    ag_file = os.path.join(DATA_DIR, 'gaia_ag_nside128.npy')

    # ① Stellarium テクスチャ取得・処理
    if not os.path.exists(mw_npy):
        if not os.path.exists(mw_png):
            print("  [milkyway] Stellarium テクスチャ取得中...")
            url = ("https://raw.githubusercontent.com/Stellarium/stellarium"
                   "/master/textures/milkyway.png")
            urllib.request.urlretrieve(url, mw_png)
        print("  [milkyway] ガウスブラーで星除去処理中...")
        arr = np.array(Image.open(mw_png)).astype(np.float32) / 255.0
        blurred = gaussian_filter(arr, sigma=[8, 8, 0])
        lum = blurred.mean(axis=2)
        alpha = np.power(np.clip(lum / 0.35, 0, 1), 0.7)
        rgba = np.zeros((*arr.shape[:2], 4), dtype=np.float32)
        rgba[:, :, :3] = blurred
        rgba[:, :, 3]  = alpha
        np.save(mw_npy, rgba)
        print(f"  [milkyway] 保存完了: {os.path.basename(mw_npy)}")

    # ② Gaia 消光マップ
    if not os.path.exists(ag_file):
        ag_file32 = os.path.join(DATA_DIR, 'gaia_ag_nside32.npy')
        if not os.path.exists(ag_file32):
            import urllib.parse, json as _json
            import healpy as hp
            TAP = 'https://gea.esac.esa.int/tap-server/tap/sync'
            def tap(q):
                url = (TAP + '?REQUEST=doQuery&LANG=ADQL&FORMAT=json'
                       '&MAXREC=1000000&QUERY=' + urllib.parse.quote(q))
                with urllib.request.urlopen(url, timeout=300) as r:
                    return _json.load(r).get('data', [])
            nside=128; npix=hp.nside2npix(nside)
            print("  [Gaia] 消光マップ取得中（nside=128, 約30秒）...")
            rows = tap("SELECT gaia_healpix_index(7, source_id) AS hpx,"
                       "AVG(ag_gspphot) AS mean_ag "
                       "FROM gaiadr3.gaia_source "
                       "WHERE phot_g_mean_mag<=17.5 AND ag_gspphot IS NOT NULL "
                       "GROUP BY hpx")
            ag = np.zeros(npix, dtype=np.float32)
            for r in rows:
                if r[1] is not None: ag[int(r[0])] = float(r[1])
            np.save(ag_file, ag)
            print("  [Gaia] 消光マップ保存完了")


# ═══════════════════════════════════════════════════════
#  メイン描画クラス
# ═══════════════════════════════════════════════════════

class StarChart:
    BG      = '#080e1f'
    HORIZON = '#4466aa'
    GRID    = '#1a2a44'
    CONLINE = '#ffffff'
    CONNAME = '#add8f0'

    def __init__(self, dt_utc, lat, lon, location_name='', mag_limit=6.5, star_scale=1.0):
        self.dt_utc     = dt_utc.replace(tzinfo=timezone.utc)
        self.lat        = lat
        self.lon        = lon
        self.name       = location_name
        self.mag_lim    = mag_limit
        self.star_scale = max(0.1, float(star_scale))   # 星サイズ倍率
        self.R          = 1.0
        self._clip      = None

    # ─── メイン ──────────────────────────────────────────

    def generate_fig(self, progress_cb=None):
        """星図 Figure を生成して返す（ファイルへの保存は行わない）。
        progress_cb(msg: str) を渡すと各ステップの進捗を受け取れる。"""
        # テキストをアウトライン化せずフォント埋め込みで保持する
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype']  = 42

        def _p(msg):
            if progress_cb: progress_cb(msg)
            else: print(msg)

        _p('[1/7] 星表ロード中...')
        stars = load_hip_catalog(self.mag_lim)
        stars['alt'], stars['az'] = radec_to_altaz_np(
            stars['ra_deg'].values, stars['dec_deg'].values,
            self.lat, self.lon, self.dt_utc)

        _p('[2/7] 星座・天の川データロード中...')
        con_lines, con_labels = load_constellation_data()
        mw_data = load_milkyway_data()
        _ensure_gaia_milkyway_data()

        _p('[3/7] 惑星・月の位置計算中 (skyfield)...')
        sky_objects = self._compute_sky_objects()

        _p('[4/7] 描画中...')
        fig = plt.figure(figsize=(11, 11), facecolor=self.BG)
        ax  = fig.add_axes([0.05, 0.05, 0.90, 0.90])
        ax.set_facecolor(self.BG)
        ax.set_xlim(-1.18, 1.18)
        ax.set_ylim(-1.18, 1.18)
        ax.set_aspect('equal')
        ax.axis('off')

        clip = Circle((0, 0), self.R, transform=ax.transData)
        self._clip = clip

        self._draw_milkyway(ax, mw_data)
        self._draw_grid(ax)
        self._draw_constellation_lines(ax, con_lines, stars)
        self._draw_stars(ax, stars)
        self._draw_constellation_names(ax, con_labels)
        self._draw_dso(ax)
        self._draw_sky_objects(ax, sky_objects)
        self._draw_frame(ax)
        self._draw_title(ax)

        return fig

    def generate(self, output_path: str):
        is_svg = output_path.lower().endswith('.svg')
        if is_svg:
            plt.rcParams['svg.fonttype'] = 'none'   # SVGはテキストをそのまま保持

        fig = self.generate_fig()
        ext = 'SVG' if is_svg else 'PDF'
        print(f"[5/7] {ext}保存中: {output_path}")
        save_kw = dict(bbox_inches='tight', facecolor=self.BG, edgecolor='none')
        if not is_svg:
            save_kw['dpi'] = 200
        fig.savefig(output_path, **save_kw)
        plt.close(fig)
        print(f"完了: {output_path}")

    def _draw_milkyway(self, ax, mw_data):
        """天の川: Stellariumテクスチャ(GPL) から拡散光を抽出して3段階離散表示
        座標系: s=atan2(sin(RA),cos(RA))/(2π)+0.5, t=(90-Dec)/180
        個別星はσ=8px ガウスブラーで除去済み
        """
        from scipy.ndimage import map_coordinates
        import healpy as hp
        import warnings as _w

        tex_file = os.path.join(DATA_DIR, 'milkyway_blurred.npy')
        ag_file  = os.path.join(DATA_DIR, 'gaia_ag_nside128.npy')
        if not os.path.exists(ag_file):
            ag_file = os.path.join(DATA_DIR, 'gaia_ag_nside32.npy')
        if not os.path.exists(tex_file):
            return

        tex = np.load(tex_file)          # (1024, 2048, 4) float32
        TH, TW = tex.shape[:2]
        lum_tex = tex[:, :, :3].mean(axis=2)   # 輝度のみ使用

        ag_map = np.load(ag_file).astype(np.float64) if os.path.exists(ag_file) else None

        # ── 1. チャートXY → RA/Dec ──────────────────────────────
        M = 500
        lin    = np.linspace(-1.0, 1.0, M)
        XX, YY = np.meshgrid(lin, lin)
        RR     = np.sqrt(XX**2 + YY**2)
        mask   = RR <= 1.0

        lat_r = np.radians(self.lat)
        lst   = compute_lst_deg(self.dt_utc, self.lon)
        az_r  = np.radians(np.degrees(np.arctan2(-XX, YY)) % 360.0)
        alt_r = np.radians(90.0 * (1.0 - np.clip(RR, 0, 1.0)))

        sin_dec = np.sin(alt_r)*np.sin(lat_r)+np.cos(alt_r)*np.cos(lat_r)*np.cos(az_r)
        dec_r   = np.arcsin(np.clip(sin_dec,-1,1))
        cos_dec = np.cos(dec_r)
        safe    = (cos_dec>1e-9) & mask
        sin_H   = np.where(safe,-np.sin(az_r)*np.cos(alt_r)/np.where(safe,cos_dec,1e-9),0)
        cos_H   = np.where(safe,(np.sin(alt_r)-np.sin(lat_r)*sin_dec)/
                           np.where(safe,cos_dec*np.cos(lat_r),1e-9),0)
        ra_deg  = (np.degrees(np.arctan2(sin_H,cos_H))*-1+lst)%360
        dec_deg = np.degrees(dec_r)

        # ── 2. テクスチャ座標 (正しい公式: s=(90-RA)%360/360) ────
        # RA=18h(射手座)が中央, RA=6h(双子座)が左端
        s = ((90.0 - ra_deg) % 360.0) / 360.0 * (TW - 1)
        t = np.clip((90 - dec_deg) / 180 * (TH - 1), 0, TH-1)

        # ── 3. バイリニア補間サンプリング ────────────────────────
        lum = map_coordinates(lum_tex,
                              [t.ravel(), s.ravel()],
                              order=1, mode='wrap').reshape(M, M)
        lum[~mask] = 0.0

        # ── 4. contourf でベクターパス化（Illustrator 完全編集可）──
        from scipy.ndimage import gaussian_filter as _gf
        import warnings as _w2
        valid = mask & (lum > 1e-4)
        if valid.sum() < 100:
            return

        p1 = float(np.nanpercentile(lum[valid], 78))
        p2 = float(np.nanpercentile(lum[valid], 88))
        p3 = float(np.nanpercentile(lum[valid], 92))

        # 輝度をスムージングして滑らかな等値線パスを生成
        SIGMA = 7
        lum_s = _gf(lum * mask.astype(np.float32), sigma=SIGMA)
        lum_s[~mask] = 0.0
        lum_max = float(lum_s.max()) + 1e-6

        layers = [
            (p1, (0.28, 0.38, 0.68), 0.50, 1),
            (p2, (0.38, 0.50, 0.78), 0.60, 2),
            (p3, (0.52, 0.65, 0.90), 0.65, 3),
        ]
        for thresh, rgb, alpha, zord in layers:
            if thresh >= lum_max:
                continue
            n_before = len(ax.collections)
            cf = ax.contourf(XX, YY, lum_s,
                             levels=[thresh, lum_max],
                             colors=[rgb], alpha=alpha,
                             zorder=zord, antialiased=True)
            for coll in ax.collections[n_before:]:
                coll.set_clip_path(self._clip)

        # ── 5. Gaia 消光オーバーレイ (大スムージング) ─────────────
        if ag_map is not None:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                ag_sp = hp.smoothing(ag_map, fwhm=np.radians(3.0))

            ra_GP=np.radians(192.85948); dec_GP=np.radians(27.12825); L_NCP=122.93192
            ra_r = np.radians(ra_deg)
            sin_b=(np.sin(dec_GP)*np.sin(dec_r)+np.cos(dec_GP)*np.cos(dec_r)*
                   np.cos(ra_r-ra_GP))
            b_deg=np.degrees(np.arcsin(np.clip(sin_b,-1,1)))
            cos_b=np.cos(np.radians(b_deg))
            sin_ll=np.cos(dec_r)*np.sin(ra_r-ra_GP)/np.where(cos_b>1e-9,cos_b,1e-9)
            cos_ll=(np.sin(dec_r)-sin_b*np.sin(dec_GP))/                   np.where(cos_b*np.cos(dec_GP)>1e-9,cos_b*np.cos(dec_GP),1e-9)
            l_deg=(np.degrees(np.arctan2(sin_ll,cos_ll))+L_NCP)%360.0

            ag=hp.get_interp_val(ag_sp,np.radians(90-b_deg).ravel(),
                                 np.radians(l_deg).ravel()).reshape(M,M)
            ag[~mask]=0.0

            dark_a=(np.clip((ag-0.85)/2.0,0.0,0.60)*mask).astype(np.float32)
            ax.imshow(np.stack([np.full((M,M),0.02,np.float32),
                                np.full((M,M),0.03,np.float32),
                                np.full((M,M),0.07,np.float32),
                                dark_a],axis=-1),
                      extent=[-1,1,-1,1], origin='lower',
                      aspect='equal', zorder=2, interpolation='bilinear')

    # ─── グリッド ─────────────────────────────────────────

    def _draw_grid(self, ax):
        """高度圏・方位線"""
        # 高度圏
        for alt in [0, 30, 60]:
            r = (90 - alt) / 90
            lw = 1.2 if alt == 0 else 0.5
            col = self.HORIZON if alt == 0 else self.GRID
            c = Circle((0, 0), r, fill=False, color=col,
                       linewidth=lw, zorder=2, linestyle='-')
            ax.add_patch(c)
            if alt > 0:
                ax.text(0.012, r + 0.02, f'{alt}°',
                        color='#2a3a55', fontsize=6.5,
                        ha='left', va='bottom', zorder=2)

        # 方位線（8方向）
        for az in range(0, 360, 45):
            az_rad = np.radians(az)
            x = -np.sin(az_rad); y = np.cos(az_rad)
            ax.plot([0, x], [0, y], color=self.GRID,
                    linewidth=0.5, zorder=2, linestyle='-')

    # ─── 星座線 ───────────────────────────────────────────

    def _draw_constellation_lines(self, ax, con_lines, stars_df):
        segments = []
        for feature in con_lines.get('features', []):
            geom = feature.get('geometry', {})
            if geom.get('type') != 'MultiLineString':
                continue
            for line in geom['coordinates']:
                for i in range(len(line) - 1):
                    p1, p2 = line[i], line[i+1]
                    ra1 = d3ra_to_deg(p1[0]); dec1 = p1[1]
                    ra2 = d3ra_to_deg(p2[0]); dec2 = p2[1]
                    alt1, az1 = radec_to_altaz_np(ra1, dec1, self.lat, self.lon, self.dt_utc)
                    alt2, az2 = radec_to_altaz_np(ra2, dec2, self.lat, self.lon, self.dt_utc)
                    if float(alt1) < 0 and float(alt2) < 0:
                        continue
                    x1, y1 = altaz_to_xy(float(alt1), float(az1))
                    x2, y2 = altaz_to_xy(float(alt2), float(az2))
                    if not (np.isnan(x1) or np.isnan(x2)):
                        segments.append([(x1, y1), (x2, y2)])

        lc = LineCollection(segments, colors=self.CONLINE,
                            linewidths=0.6, alpha=0.65, zorder=3)
        lc.set_clip_path(self._clip)
        ax.add_collection(lc)

    # ─── 恒星 ──────────────────────────────────────────────

    @staticmethod
    def _star_patch(x, y, r, n, rot_deg, color, inner_ratio=0.40):
        """n頂点の星形 PathPatch を返す。r は外接円半径（データ座標）
        rot_deg=0 のとき最初の頂点が真上を向く"""
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch
        angles = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
        angles += np.radians(rot_deg) + np.pi / 2   # +π/2 で最初の頂点が真上
        radii  = np.where(np.arange(2 * n) % 2 == 0, r, r * inner_ratio)
        verts  = list(zip(x + radii * np.cos(angles),
                          y + radii * np.sin(angles)))
        verts.append(verts[0])
        codes  = [Path.MOVETO] + [Path.LINETO] * (2 * n - 1) + [Path.CLOSEPOLY]
        return PathPatch(Path(verts, codes), facecolor=color, edgecolor='none')

    def _draw_stars(self, ax, stars_df):
        visible = stars_df[stars_df['alt'] >= 0]
        # 暗い星から明るい星の順に描く（明るい星が上に来る）
        visible = visible.sort_values('magnitude', ascending=False)

        for _, row in visible.iterrows():
            x, y = altaz_to_xy(row['alt'], row['az'])
            if np.isnan(x):
                continue
            bv    = row['bv'] if not np.isnan(row['bv']) else 0.0
            color = ballesteros_color(bv)

            # 暗い星ほど彩度を落として白に近づける（Purkinje効果に相当）
            # mag ≤ 3.0: 彩度100% → mag ≥ 5.5: 彩度0%（白に近い）
            sat   = float(np.clip((5.5 - row['magnitude']) / 2.5, 0.0, 1.0))
            color = tuple(c * sat + 1.0 * (1.0 - sat) for c in color)
            mag   = float(row['magnitude'])
            r     = mag_to_radius(mag) * self.star_scale

            # 1等星: 5頂点星形・固定サイズ / 2等星: 4頂点手裏剣形 / 3等星: 円 / 4等星以下: 円(縮小)
            if mag <= 1.5:
                r_fixed = mag_to_radius(1.0) * 1.5 * self.star_scale
                patch = self._star_patch(x, y, r_fixed, n=5, rot_deg=0,
                                         color=color, inner_ratio=0.40)
            elif mag <= 2.5:
                patch = self._star_patch(x, y, r * 1.95, n=4, rot_deg=0,
                                         color=color, inner_ratio=0.35)
            elif mag <= 3.5:
                patch = Circle((x, y), r, color=color)
            else:
                patch = Circle((x, y), r * 0.7, color=color)

            patch.set_zorder(5)
            patch.set_clip_path(self._clip)
            ax.add_patch(patch)

    # ─── 星座名 ─────────────────────────────────────────────

    def _draw_constellation_names(self, ax, con_labels):
        fp = get_jp_font(6.5)
        for feature in con_labels.get('features', []):
            props = feature.get('properties', {})
            name_ja = props.get('ja', '')
            if not name_ja:
                continue
            coords = feature.get('geometry', {}).get('coordinates', [])
            if len(coords) < 2:
                continue
            ra_deg  = d3ra_to_deg(coords[0])
            dec_deg = coords[1]
            alt, az = radec_to_altaz_np(ra_deg, dec_deg, self.lat, self.lon, self.dt_utc)
            if float(alt) < 2:
                continue
            x, y = altaz_to_xy(float(alt), float(az))
            if np.isnan(x):
                continue
            ax.text(x, y, name_ja, color=self.CONNAME,
                    fontsize=6.5, ha='center', va='center',
                    alpha=0.85, zorder=6, fontproperties=fp)

    # ─── 惑星・月 ────────────────────────────────────────────

    def _draw_dso(self, ax):
        """深宇宙天体（DSO）のマーカーを描画。現在はM31のみ。"""
        from matplotlib.patches import Ellipse

        # ── M31 アンドロメダ銀河 ──
        # RA=10.685°  Dec=+41.269°  赤道座標系での長軸位置角 PA≈35°（北から東へ）
        M31_RA  = 10.685
        M31_DEC = 41.269
        M31_PA  = 35.0   # degrees, North→East

        alt0, az0 = radec_to_altaz_np(
            np.array([M31_RA]), np.array([M31_DEC]),
            self.lat, self.lon, self.dt_utc
        )
        alt0, az0 = float(alt0[0]), float(az0[0])
        if alt0 < 0:
            return

        cx, cy = altaz_to_xy(alt0, az0)
        if np.isnan(cx):
            return

        # ── 画面上の傾き角を投影で計算 ──
        # 赤道座標系で長軸方向に微小オフセットした点を投影し、画面角度を求める
        d = 0.5   # オフセット量（度）
        pa_rad = np.radians(M31_PA)
        # 位置角PAの方向へのオフセット（北→東で正）
        dDec = d * np.cos(pa_rad)
        dRA  = d * np.sin(pa_rad) / np.cos(np.radians(M31_DEC))

        alt1, az1 = radec_to_altaz_np(
            np.array([M31_RA + dRA]), np.array([M31_DEC + dDec]),
            self.lat, self.lon, self.dt_utc
        )
        x1, y1 = altaz_to_xy(float(alt1[0]), float(az1[0]))

        # 画面上のベクトルから角度を算出（matplotlib angle: 水平右が0°、反時計回り正）
        if np.isnan(x1) or np.isnan(y1):
            screen_angle = 35.0
        else:
            dx, dy = x1 - cx, y1 - cy
            screen_angle = np.degrees(np.arctan2(dy, dx))

        # 楕円サイズ（データ座標）
        r_maj = 0.055 * 0.7
        r_min = 0.022 * 0.7
        color  = '#aabbdd'

        # 外側の淡い楕円（ハロー）＋内側の明るい楕円
        for r_scale, alpha in [(1.0, 0.55), (0.65, 0.70)]:
            e = Ellipse(
                (cx, cy),
                width=r_maj * 2 * r_scale,
                height=r_min * 2 * r_scale,
                angle=screen_angle,
                facecolor=color,
                edgecolor='none',
                alpha=alpha,
                zorder=4,
            )
            e.set_clip_path(self._clip)
            ax.add_patch(e)

        # 輪郭線
        e_edge = Ellipse(
            (cx, cy),
            width=r_maj * 2,
            height=r_min * 2,
            angle=screen_angle,
            facecolor='none',
            edgecolor='#8899cc',
            linewidth=0.8,
            alpha=0.8,
            zorder=4,
        )
        e_edge.set_clip_path(self._clip)
        ax.add_patch(e_edge)

        # ラベル（長軸端の右側に配置）
        lx = cx + (r_maj + 0.014) * np.cos(np.radians(screen_angle))
        ly = cy + (r_maj + 0.014) * np.sin(np.radians(screen_angle))
        fp = get_jp_font(6)
        ax.text(lx, ly, 'M31',
                color='#8899cc', fontsize=6.5,
                ha='left', va='center',
                fontproperties=fp, zorder=4)

    def _compute_sky_objects(self):
        """skyfield で惑星・月の位置を計算"""
        from skyfield.api import Loader, wgs84

        loader   = Loader(DATA_DIR)
        ts       = loader.timescale()
        t        = ts.from_datetime(self.dt_utc)
        eph      = loader('de421.bsp')
        observer = (eph['earth'] + wgs84.latlon(self.lat, self.lon)).at(t)

        result = {}
        for key, (name_ja, color, size) in PLANET_TABLE.items():
            try:
                body    = eph[key]
                app     = observer.observe(body).apparent()
                alt, az, _ = app.altaz()
                result[key] = {
                    'name': name_ja, 'color': color, 'size': size,
                    'alt': alt.degrees, 'az': az.degrees
                }
            except Exception as e:
                pass

        # 月
        moon    = eph['moon']
        app_m   = observer.observe(moon).apparent()
        alt_m, az_m, _ = app_m.altaz()
        result['moon'] = {
            'name': '月', 'color': '#e8e8c8', 'size': 9.0,
            'alt': alt_m.degrees, 'az': az_m.degrees
        }
        return result

    def _draw_sky_objects(self, ax, sky_objects):
        fp = get_jp_font(8)
        for key, obj in sky_objects.items():
            alt, az = obj['alt'], obj['az']
            if alt < 0:
                continue
            x, y = altaz_to_xy(alt, az)
            if np.isnan(x):
                continue
            r     = obj['size'] * 0.003
            color = obj['color']

            if key == 'moon':
                # 月：大きめの円＋輪郭
                body = Circle((x, y), r, color=color, zorder=8,
                              linewidth=0.8, edgecolor='#aaaaaa')
                ax.add_patch(body)
                # グロー
                glow = Circle((x, y), r * 2.2, color=color, alpha=0.15, zorder=7)
                ax.add_patch(glow)
            else:
                # 惑星：小さめの円
                body = Circle((x, y), r, color=color, zorder=8,
                              linewidth=0.5, edgecolor='#888888')
                ax.add_patch(body)

            # ラベル（中心より少し右上）
            ax.text(x + r + 0.015, y + r + 0.015,
                    obj['name'], color=color,
                    fontsize=8 if key == 'moon' else 7.5,
                    ha='left', va='bottom', zorder=9,
                    fontproperties=fp,
                    bbox=dict(boxstyle='round,pad=0.1',
                              facecolor=self.BG, alpha=0.6,
                              edgecolor='none'))

    # ─── フレーム（地平線・方角） ─────────────────────────────

    def _draw_frame(self, ax):
        fp_big   = get_jp_font(12)
        fp_small = get_jp_font(8)

        dirs = {
            0:   ('北', fp_big),
            90:  ('東', fp_big),
            180: ('南', fp_big),
            270: ('西', fp_big),
            45:  ('北東', fp_small),
            135: ('南東', fp_small),
            225: ('南西', fp_small),
            315: ('北西', fp_small),
        }
        for az_deg, (label, fp) in dirs.items():
            az_rad = np.radians(az_deg)
            offset = 1.09
            x = -np.sin(az_rad) * offset
            y =  np.cos(az_rad) * offset
            ax.text(x, y, label, color='#7799bb',
                    ha='center', va='center', zorder=10,
                    fontproperties=fp)

        # 目盛（方位角を10°ごと）
        for az_tick in range(0, 360, 10):
            az_rad = np.radians(az_tick)
            r0 = 1.0
            r1 = 1.04 if az_tick % 90 == 0 else \
                 1.03 if az_tick % 45 == 0 else 1.015
            x0, y0 = -np.sin(az_rad)*r0, np.cos(az_rad)*r0
            x1, y1 = -np.sin(az_rad)*r1, np.cos(az_rad)*r1
            ax.plot([x0, x1], [y0, y1], color=self.HORIZON,
                    linewidth=1.0 if az_tick % 90 == 0 else 0.5,
                    zorder=10)

        # 地平線の円（再描画して目盛を隠さないよう最前面に）
        horizon = Circle((0, 0), self.R, fill=False,
                         color=self.HORIZON, linewidth=1.2, zorder=11)
        ax.add_patch(horizon)

    # ─── タイトル ──────────────────────────────────────────────

    def _draw_title(self, ax):
        fp = get_jp_font(13)
        jst = self.dt_utc + timedelta(hours=9)
        loc_str = f"　{self.name}" if self.name else ""
        title = f"{jst.strftime('%Y年%m月%d日  %H:%M')} JST{loc_str}"

        ax.text(0, 1.17, title,
                color='#99aacc', ha='center', va='center',
                fontproperties=fp, zorder=12)

        # 等級凡例（最下部）
        fp_s = get_jp_font(7)
        LEG_Y = -1.13
        ax.text(-1.16, LEG_Y, '等級', color='#556677',
                ha='left', va='center', fontproperties=fp_s)
        legend_items = [(1, '1', 'star'), (2, '2', 'shuriken'), (3, '3', 'circle')]
        for i, (mag, label, shape) in enumerate(legend_items):
            r  = mag_to_radius(mag) * self.star_scale
            lx = -1.02 + i * 0.14
            if shape == 'star':
                r_fixed = mag_to_radius(1.0) * 1.5 * self.star_scale
                patch = self._star_patch(lx, LEG_Y, r_fixed, n=5, rot_deg=0,
                                         color='white', inner_ratio=0.40)
            elif shape == 'shuriken':
                patch = self._star_patch(lx, LEG_Y, r * 1.95, n=4, rot_deg=0,
                                         color='white', inner_ratio=0.35)
            else:
                patch = Circle((lx, LEG_Y), r, color='white')
            patch.set_zorder(12)
            ax.add_patch(patch)
            ax.text(lx + 0.05, LEG_Y, label, color='#556677',
                    fontsize=6, ha='left', va='center',
                    fontproperties=fp_s)

        ax.text(1.16, LEG_Y, f'mag ≤ {self.mag_lim}',
                color='#556677', ha='right', va='center',
                fontproperties=fp_s)

# ═══════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════

def cli():
    parser = argparse.ArgumentParser(
        description='正距方位図法 星図PDF生成',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""例:
  python star_chart.py --date 2026-08-15 --time 21:00 \\
      --lat 35.664 --lon 139.990 --location "ふなばし三番瀬"
  python star_chart.py --gui"""
    )
    parser.add_argument('--date',     default=None,
                        help='日付 YYYY-MM-DD（省略時: 今日）')
    parser.add_argument('--time',     default=None,
                        help='時刻 HH:MM JST（省略時: 現在時刻）')
    parser.add_argument('--lat',      type=float, default=35.664,
                        help='緯度（デフォルト: 35.664 ふなばし三番瀬）')
    parser.add_argument('--lon',      type=float, default=139.990,
                        help='経度（デフォルト: 139.990）')
    parser.add_argument('--location', default='ふなばし三番瀬',
                        help='場所名（PDF タイトルに表示）')
    parser.add_argument('--output',   default='star_chart.pdf',
                        help='出力ファイル名（デフォルト: star_chart.pdf）')
    parser.add_argument('--mag',      type=float, default=6.5,
                        help='等級制限（デフォルト: 6.5）')
    parser.add_argument('--gui',      action='store_true',
                        help='GUIモードで起動')
    args = parser.parse_args()

    if args.gui:
        run_gui()
        return

    if args.date and args.time:
        dt_jst = datetime.strptime(f"{args.date} {args.time}", '%Y-%m-%d %H:%M')
    elif args.date:
        dt_jst = datetime.strptime(args.date, '%Y-%m-%d').replace(hour=21)
    else:
        dt_jst = datetime.now()

    dt_utc = dt_jst - timedelta(hours=9)
    chart  = StarChart(dt_utc, args.lat, args.lon, args.location, args.mag)
    chart.generate(args.output)

# ═══════════════════════════════════════════════════════
#  GUI（tkinter）
# ═══════════════════════════════════════════════════════

def run_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title('星図生成ツール')
    root.resizable(False, False)

    pad = {'padx': 8, 'pady': 4}
    frame = ttk.Frame(root, padding=16)
    frame.grid()

    vars_ = {}

    def row(r, label, default, width=22):
        ttk.Label(frame, text=label).grid(row=r, column=0, sticky='e', **pad)
        v = tk.StringVar(value=default)
        ttk.Entry(frame, textvariable=v, width=width).grid(row=r, column=1, **pad)
        vars_[label] = v
        return v

    today  = datetime.now().strftime('%Y-%m-%d')
    row(0, '場所名',                 'ふなばし三番瀬')
    row(1, '緯度',                   '35.664')
    row(2, '経度',                   '139.990')
    row(3, '日付 (YYYY-MM-DD)',       today)
    row(4, '時刻 (HH:MM) JST',       '21:00')
    row(5, '等級制限',               '6.5')

    out_var = tk.StringVar(value='star_chart.pdf')
    ttk.Label(frame, text='出力ファイル名').grid(row=6, column=0, sticky='e', **pad)
    ttk.Entry(frame, textvariable=out_var, width=22).grid(row=6, column=1, **pad)

    status = tk.StringVar(value='')
    ttk.Label(frame, textvariable=status, foreground='#0055aa').grid(
        row=8, columnspan=2, pady=4)

    def generate():
        try:
            loc    = vars_['場所名'].get()
            lat    = float(vars_['緯度'].get())
            lon    = float(vars_['経度'].get())
            date_s = vars_['日付 (YYYY-MM-DD)'].get()
            time_s = vars_['時刻 (HH:MM) JST'].get()
            mag    = float(vars_['等級制限'].get())
            out    = out_var.get()

            dt_jst = datetime.strptime(f"{date_s} {time_s}", '%Y-%m-%d %H:%M')
            dt_utc = dt_jst - timedelta(hours=9)

            status.set('生成中... しばらくお待ちください')
            root.update()

            chart = StarChart(dt_utc, lat, lon, loc, mag)
            chart.generate(out)

            status.set(f'完了: {out}')
            messagebox.showinfo('完了', f'星図を保存しました:\n{os.path.abspath(out)}')
        except Exception as e:
            import traceback; traceback.print_exc()
            messagebox.showerror('エラー', str(e))

    ttk.Button(frame, text='  星図を生成  ', command=generate).grid(
        row=7, columnspan=2, pady=10)
    root.mainloop()

# ═══════════════════════════════════════════════════════
#  エントリポイント
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    cli()
