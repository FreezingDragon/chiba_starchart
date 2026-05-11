"""
streamlit_app.py — ふなばし三番瀬 今夜の星空

設計方針:
  - 訪問者は画像を受け取るだけ（重い計算はすべてサーバー側）
  - データロードは1回だけ（@st.cache_resource で全訪問者が共有）
  - 星図生成は時刻ごとにキャッシュ（同じ時刻は再生成しない）
  - スマホ対応レイアウト
"""

import streamlit as st
import datetime as dt
import io, os, sys, urllib.request, gzip, shutil
from pathlib import Path

st.set_page_config(
    page_title='今夜の星空 | ふなばし三番瀬',
    page_icon='🌟',
    layout='centered',
    initial_sidebar_state='collapsed',
)

st.markdown("""
<style>
  .stSlider label { font-size: 1.05rem !important; }
  .stButton button { font-size: 1.1rem !important; padding: 0.6rem 1rem !important; }
  .stDownloadButton button { font-size: 1rem !important; }
  .block-container { padding-top: 1.5rem !important; }
  h1 { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

LOCATION_NAME = 'ふなばし三番瀬'
LAT  = 35.664
LON  = 139.990
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'star_chart_data'
DATA_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(BASE_DIR))

@st.cache_resource(show_spinner=False)
def _load_backend():
    def _dl(fname, url, gz=False):
        dst = DATA_DIR / fname
        if dst.exists():
            return
        tmp = str(dst) + '.tmp'
        try:
            urllib.request.urlretrieve(url, tmp)
            if gz:
                with gzip.open(tmp, 'rb') as fi, open(str(dst), 'wb') as fo:
                    shutil.copyfileobj(fi, fo)
                os.remove(tmp)
            else:
                os.rename(tmp, str(dst))
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise RuntimeError(f'{fname} のダウンロードに失敗: {e}')

    _dl('hip_main.dat',
        'https://cdsarc.cds.unistra.fr/ftp/I/239/hip_main.dat.gz', gz=True)
    _dl('de421.bsp',
        'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de421.bsp')
    _dl('milkyway.png',
        'https://raw.githubusercontent.com/Stellarium/stellarium/master/nebulae/default/milkyway.png')
    _dl('constellations.lines.json',
        'https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/constellations.lines.json')
    _dl('constellations.json',
        'https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/constellations.json')

    import star_chart as _sc
    return _sc

@st.cache_data(ttl=3600, show_spinner=False)
def _make_chart_png(date_str: str, hour: int) -> bytes:
    """スマホ閲覧用 PNG（軽量・素早く表示）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sc = _load_backend()
    dt_jst = dt.datetime.strptime(f'{date_str} {hour:02d}:00', '%Y-%m-%d %H:%M')
    dt_utc = dt_jst - dt.timedelta(hours=9)
    chart = sc.StarChart(dt_utc, LAT, LON, LOCATION_NAME, mag_limit=6.5, star_scale=1.0)
    fig = chart.generate_fig()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#080e1f')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

@st.cache_data(ttl=3600, show_spinner=False)
def _make_chart_print_pdf(date_str: str, hour: int) -> bytes:
    """印刷用 高解像度PDF（A4・dpi=300）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sc = _load_backend()
    dt_jst = dt.datetime.strptime(f'{date_str} {hour:02d}:00', '%Y-%m-%d %H:%M')
    dt_utc = dt_jst - dt.timedelta(hours=9)
    chart = sc.StarChart(dt_utc, LAT, LON, LOCATION_NAME,
                         mag_limit=6.5, star_scale=1.2)   # 印刷用に星を少し大きく
    fig = chart.generate_fig()
    # A4横サイズ（297×210mm）にリサイズ
    fig.set_size_inches(11.69, 11.69)   # 正方形でA4幅に合わせる
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', dpi=300, bbox_inches='tight', facecolor='#080e1f')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

today = dt.date.today()
st.title('🌟 今夜の星空')
st.caption(f'{today.strftime("%Y年 %m月 %d日")}　{LOCATION_NAME}から見た星空')

HOURS = list(range(18, 24)) + list(range(0, 4))
selected_hour = st.select_slider(
    '観測時刻（JST）',
    options=HOURS,
    value=21,
    format_func=lambda h: f'{h:02d}:00',
)

chart_date = (today + dt.timedelta(days=1)).strftime('%Y-%m-%d') \
    if selected_hour < 4 else today.strftime('%Y-%m-%d')

with st.spinner('星図を生成中... しばらくお待ちください'):
    try:
        png_bytes = _make_chart_png(chart_date, selected_hour)
        st.image(png_bytes, use_container_width=True)

        fname = f'sanbanze_{chart_date}_{selected_hour:02d}00'

        st.markdown('**ダウンロード**')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('📱 **観望会・スマホで見る**')
            st.caption('軽量PNG・すぐ表示できます')
            st.download_button('PNG をダウンロード', data=png_bytes,
                               file_name=f'{fname}.png',
                               mime='image/png',
                               use_container_width=True)
        with col2:
            st.markdown('🖨️ **自宅で印刷する**')
            st.caption('高解像度PDF・A4印刷向け')
            pdf_bytes = _make_chart_print_pdf(chart_date, selected_hour)
            st.download_button('PDF をダウンロード', data=pdf_bytes,
                               file_name=f'{fname}_print.pdf',
                               mime='application/pdf',
                               use_container_width=True)
    except Exception as e:
        st.error(f'生成エラー: {e}')
        import traceback; st.code(traceback.format_exc())

st.divider()
st.caption('ふなばし三番瀬環境学習館　|　星表: Hipparcos (ESA)　|　天の川: Stellarium (GPL)　|　惑星暦: JPL DE421')
