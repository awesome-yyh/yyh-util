import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import panel as pn

# 初始化Panel扩展
pn.extension('plotly', 'tabulator')

# 定义颜色常量
ACCENT = "#BB2649"
RED = "#D94467"
GREEN = "#5AD534"

# 定义链接图标的SVG
LINK_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-arrow-up-right-square" viewBox="0 0 16 16">
  <path fill-rule="evenodd" d="M15 2a1 1 0 0 0-1-1H2a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V2zM0 2a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V2zm5.854 8.803a.5.5 0 1 1-.708-.707L9.243 6H6.475a.5.5 0 1 1 0-1h3.975a.5.5 0 0 1 .5.5v3.975a.5.5 0 1 1-1 0V6.707l-4.096 4.096z"/>
</svg>
"""

# 数据集URL
CSV_URL = "https://datasets.holoviz.org/equities/v1/equities.csv"

# 定义股票列表
EQUITIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "TSLA": "Tesla",
    "BRK-B": "Berkshire Hathaway",
    "UNH": "United Health Group",
    "JNJ": "Johnson & Johnson",
}
EQUITY_LIST = tuple(EQUITIES.keys())

# 获取历史数据
@pn.cache(ttl=600)
def get_historical_data(tickers=EQUITY_LIST, period="2y"):
    """下载Yahoo Finance的历史数据"""
    df = pd.read_csv(CSV_URL, index_col=[0, 1], parse_dates=['Date'])
    return df

historical_data = get_historical_data()
historical_data.head(3).round(2)

# 数据转换
def last_close(ticker, data=historical_data):
    """获取给定股票的最后收盘价"""
    return data.loc[ticker]["Close"].iloc[-1]

last_close("AAPL")

# 总结数据字典
summary_data_dict = {
    "ticker": EQUITY_LIST,
    "company": EQUITIES.values(),
    "info": [
        f"""<a href='https://finance.yahoo.com/quote/{ticker}' target='_blank'>
        <div title='Open in Yahoo'>{LINK_SVG}</div></a>"""
        for ticker in EQUITIES
    ],
    "quantity": [75, 40, 100, 50, 40, 60, 20, 40],
    "price": [last_close(ticker) for ticker in EQUITIES],
    "value": None,
    "action": ["buy", "sell", "hold", "hold", "hold", "hold", "hold", "hold"],
    "notes": ["" for i in range(8)],
}

summary_data = pd.DataFrame(summary_data_dict)

def get_value_series(data=summary_data):
    """计算数量乘以价格的系列"""
    return data["quantity"] * data["price"]

summary_data["value"] = get_value_series()
summary_data.head(2)

# 定义总结表格
titles = {
    "ticker": "股票代码",
    "company": "公司",
    "info": "信息",
    "quantity": "股数",
    "price": "最后收盘价",
    "value": "市值",
    "action": "操作",
    "notes": "备注",
}
frozen_columns = ["ticker", "company"]
editors = {
    "ticker": None,
    "company": None,
    "quantity": {"type": "number", "min": 0, "step": 1},
    "price": None,
    "value": None,
    "action": {
        "type": "list",
        "values": {"buy": "buy", "sell": "sell", "hold": "hold"},
    },
    "notes": {
        "type": "textarea",
        "elementAttributes": {"maxlength": "100"},
        "selectContents": True,
        "verticalNavigation": "editor",
        "shiftEnterSubmit": True,
    },
    "info": None,
}

widths = {"notes": 400}
formatters = {
    "price": {"type": "money", "decimal": ".", "thousand": ",", "precision": 2},
    "value": {"type": "money", "decimal": ".", "thousand": ",", "precision": 0},
    "info": {"type": "html", "field": "html"},
}

text_align = {
    "price": "right",
    "value": "right",
    "action": "center",
    "info": "center",
}
base_configuration = {
    "clipboard": "copy"
}

# 定义widget.summary_table
summary_table = pn.widgets.Tabulator(
    summary_data,
    editors=editors,
    formatters=formatters,
    frozen_columns=frozen_columns,
    layout="fit_data_table",
    selectable=1,
    show_index=False,
    text_align=text_align,
    titles=titles,
    widths=widths,
    configuration=base_configuration,
)
summary_table

# 使用Pandas styler API设置表格样式
def style_of_action_cell(value, colors={'buy': GREEN, 'sell': RED}):
    """根据操作单元格的值应用CSS样式"""
    return f'color: {colors[value]}' if value in colors else ''

summary_table.style.applymap(style_of_action_cell, subset=["action"]).set_properties(
    **{"background-color": "#444"}, subset=["quantity"]
)

# 定义单元格编辑后的处理函数
patches = pn.widgets.IntInput(description="当单元格值改变时触发事件")

def handle_cell_edit(event, table=summary_table):
    """当'quantity'单元格更新时，更新'value'单元格"""
    row = event.row
    column = event.column
    if column == "quantity":
        quantity = event.value
        price = summary_table.value.loc[row, "price"]
        value = quantity * price
        table.patch({"value": [(row, value)]})

        patches.value += 1

# 定义图表
def candlestick(selection=[], data=summary_data):
    """生成蜡烛图"""
    if not selection:
        ticker = "AAPL"
        company = "Apple"
    else:
        index = selection[0]
        ticker = data.loc[index, "ticker"]
        company = data.loc[index, "company"]

    dff_ticker_hist = historical_data.loc[ticker].reset_index()
    dff_ticker_hist["Date"] = pd.to_datetime(dff_ticker_hist["Date"])

    fig = go.Figure(
        go.Candlestick(
            x=dff_ticker_hist["Date"],
            open=dff_ticker_hist["Open"],
            high=dff_ticker_hist["High"],
            low=dff_ticker_hist["Low"],
            close=dff_ticker_hist["Close"],
        )
    )
    fig.update_layout(
        title_text=f"{ticker} {company} Daily Price",
        template="plotly_dark",
        autosize=True,
    )
    return fig

pn.pane.Plotly(candlestick())

def portfolio_distribution(patches=0):
    """生成投资组合分布图"""
    data = summary_table.value
    portfolio_total = data["value"].sum()

    fig = px.pie(
        data,
        values="value",
        names="ticker",
        hole=0.3,
        title=f"Portfolio Total $ {portfolio_total:,.0f}",
        template="plotly_dark",
    )
    fig.layout.autosize = True
    return fig

pn.pane.Plotly(portfolio_distribution())

# 绑定小部件和函数
candlestick = pn.bind(candlestick, selection=summary_table.param.selection)
summary_table.on_edit(handle_cell_edit)
portfolio_distribution = pn.bind(portfolio_distribution, patches=patches)
