import streamlit as st
import pandas as pd
import numpy as np
import time

# Set basic page config
st.set_page_config(page_title='Streamlit Elements Demo', page_icon=':sparkles:')

# -------------------------------------------------
# 이 파일은 Streamlit에서 한 페이지에 넣을 수 있는
# 다양한 UI 요소들을 예시와 함께 보여줍니다.
# 각 섹션에는 요소 사용법을 학습하기 위한 자세한 한글 주석이 포함되어 있습니다.
# 필요하면 일부 섹션을 주석처리 하거나 더 확장해보세요.
# -------------------------------------------------

st.title('Streamlit 요소 데모')
st.write('다음은 Streamlit에서 한 페이지에 넣을 수 있는 주요 요소들의 예시입니다.')

# -------------------- 텍스트 + 미디어 --------------------
st.header('텍스트와 미디어')
# 마크다운, 일반 텍스트, 코드 블록, 라텍스, 이미지, 오디오, 비디오 예시
st.subheader('마크다운과 일반 텍스트')
st.markdown('''
여기서는 `st.markdown`, `st.write`, `st.caption` 등을 사용합니다.
- `st.markdown`: 풍부한 마크다운 렌더링
- `st.write`: 다양한 타입(문자열, 데이터프레임, matplotlib 등)을 자동으로 렌더링
- `st.caption`: 작은 설명 텍스트
''')
st.write('st.write는 타입을 보고 적절히 렌더링합니다: 예) 리스트, dict, 숫자')
st.code("""# 파이썬 코드 예시
print('Hello Streamlit')""", language='python')
st.latex(r"""\sum_{i=1}^n i = \frac{n(n+1)}{2}""")

# 이미지/오디오/비디오(예시는 인터넷이 없더라도 동작하도록 빈 예시 제공)
st.image('https://placekitten.com/400/200', caption='예시 이미지: placeholder')

# -------------------- 입력 위젯 --------------------
st.header('입력 위젯')
st.subheader('버튼, 체크박스, 라디오, 셀렉트박스 등')
st.markdown('''
아래 위젯들은 사용자의 입력을 받아 상호작용을 구현할 때 사용합니다.
- `st.button(label)`: 클릭 여부를 반환 (bool)
- `st.checkbox(label, value=False)`: 체크 여부 반환
- `st.radio(label, options)`: 단일 선택 반환
- `st.selectbox(label, options)`: 단일 선택 반환
- `st.multiselect(label, options)`: 복수 선택 반환
- `st.slider(...)`, `st.number_input(...)`: 숫자 입력
- `st.text_input`, `st.text_area`, `st.date_input`, `st.time_input`: 텍스트/날짜 입력
''')

# 예시 위젯들
if st.button('클릭 버튼 예시'):
    st.success('버튼을 클릭했습니다!')

agree = st.checkbox('동의합니다')
st.write('동의 여부:', agree)

radio = st.radio('라디오 선택', ['옵션 A', '옵션 B', '옵션 C'])
st.write('선택된 라디오:', radio)

select = st.selectbox('셀렉트박스 예시', ['서울', '부산', '대구'])
st.write('선택된 도시:', select)

multi = st.multiselect('멀티셀렉트 예시', ['Python', 'R', 'Julia', 'SQL'], default=['Python'])
st.write('선택된 언어들:', multi)

age = st.slider('나이(숫자 슬라이더)', 0, 120, 30)
st.write('나이:', age)

name = st.text_input('이름을 입력하세요', value='홍길동')
st.write('안녕하세요,', name)

# -------------------- 레이아웃 --------------------
st.header('레이아웃 구성')
st.markdown('`st.columns`, `st.expander`, `st.sidebar`, `st.container` 등을 사용하여 레이아웃을 구성할 수 있습니다.')

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.metric('매출', '₩ 1,234', delta='+5%')
with col2:
    st.warning('중요 알림 영역')
with col3:
    st.info('참고 정보')

with st.expander('추가 설명(펼치기)'):
    st.write('이 안에 더 많은 정보를 넣을 수 있습니다.')

with st.container():
    st.write('container 내부에 여러 위젯을 그룹화할 수 있습니다.')

# 사이드바 예시
st.sidebar.header('사이드바')
sidebar_option = st.sidebar.selectbox('사이드바 선택', ['A', 'B', 'C'])
st.sidebar.write('선택:', sidebar_option)

# -------------------- 데이터 표시 --------------------
st.header('데이터 표시 및 시각화')
st.markdown('`st.dataframe`, `st.table`, `st.json`, `st.metric`, `st.line_chart`, `st.bar_chart`, `st.map` 등')

# 간단한 데이터프레임 예시
df = pd.DataFrame({
    '도시': ['서울', '부산', '대구'],
    '인구(만)': [1000, 350, 250]
})
st.dataframe(df)  # 인터랙티브한 스크롤 가능한 데이터프레임
st.table(df)      # 정적 테이블 (PDF 등 출력에 적합)
st.json({'key': 'value', 'numbers': [1,2,3]})

# 차트 예시: streamlit은 pandas/plotly/altair 등을 바로 렌더링
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['A', 'B', 'C'])
st.line_chart(chart_data)
st.bar_chart(chart_data.iloc[:, :2])

# 지도 예시: 위도/경도 데이터 필요
map_data = pd.DataFrame(
    np.random.randn(100, 2) / [50, 50] + [37.56, 126.97],
    columns=['lat', 'lon']
)
st.map(map_data)

# -------------------- 미디어 재생 및 파일 업로드 --------------------
st.header('미디어 및 파일 업로드')
# 파일 업로드: 사용자가 CSV, 이미지 등을 업로드하면 처리할 수 있음
uploaded = st.file_uploader('CSV 업로드 (예시)', type=['csv'])
if uploaded is not None:
    df_up = pd.read_csv(uploaded)
    st.write('업로드된 데이터 미리보기:')
    st.dataframe(df_up.head())

# 오디오/비디오 예시
st.audio('https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3')
st.video('https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4')

# -------------------- 상태 표시줄 및 진행 --------------------
st.header('진행 상태 표시')
with st.spinner('잠시 기다려주세요...'):
    time.sleep(0.5)
st.success('완료')

progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)

# -------------------- 고급 위젯 및 상호작용 --------------------
st.header('고급 위젯 및 상호작용')
st.markdown('`st.form`을 사용하여 여러 입력을 모아서 한 번에 제출할 수 있습니다.')

with st.form('my_form'):
    f_name = st.text_input('이름')
    f_age = st.number_input('나이', min_value=0, max_value=120, value=20)
    submitted = st.form_submit_button('제출')
    if submitted:
        st.write(f'폼 제출됨: {f_name}, {f_age}')

# 콜아웃: 상태별 메시지
st.error('에러 메시지 예시')
st.info('정보 메시지 예시')
st.warning('경고 메시지 예시')

# -------------------- 팁과 리소스 --------------------
st.header('학습 팁')
st.markdown('''
- 공식 문서: https://docs.streamlit.io
- `st.*` 함수들을 실험해보세요. 각 함수의 docstring을 확인하려면 `help(st.text_input)`처럼 사용하세요.
- 페이지를 구성할 때는 가독성을 위해 `expander`, `columns`, `container`를 적극 활용하세요.
''')
import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='GDP dashboard',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: GDP dashboard

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
But it's otherwise a great (and did I mention _free_?) source of data.
'''

# Add some spacing
''
''

min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

countries = gdp_df['Country Code'].unique()

if not len(countries):
    st.warning("Select at least one country")

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

''
''
''

# Filter the data
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries))
    & (gdp_df['Year'] <= to_year)
    & (from_year <= gdp_df['Year'])
]

st.header('GDP over time', divider='gray')

''

st.line_chart(
    filtered_gdp_df,
    x='Year',
    y='GDP',
    color='Country Code',
)

''
''


first_year = gdp_df[gdp_df['Year'] == from_year]
last_year = gdp_df[gdp_df['Year'] == to_year]

st.header(f'GDP in {to_year}', divider='gray')

''

cols = st.columns(4)

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]

    with col:
        first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
        last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

        if math.isnan(first_gdp):
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            label=f'{country} GDP',
            value=f'{last_gdp:,.0f}B',
            delta=growth,
            delta_color=delta_color
        )
