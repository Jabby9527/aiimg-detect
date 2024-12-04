"""
@Author: suqiulin
@Email: 72405483@cityu-dg.edu.cn
@Date: 2024/12/3
"""

import pandas as pd
from pandas.io.formats.style import Styler
# 准备初始DataFrame
data = {
    'Column1': [1, 2, 3],
    'Column2': ['A', 'B', 'C']
}
df = pd.DataFrame(data)

# 添加固定值的列（使用直接赋值方式）
df['NewColumn'] = 'FixedValue'


def custom_style(x):
    # 先进行样式设置
    result = Styler(x, uuid="").hide(axis="columns", level=None, names=["NewColumn"]) \
       .set_table_styles([dict(selector="th,td", props=[('border', 'none')])])
    # 返回设置好样式后的原始DataFrame（也就是x），以满足Styler.apply要求
    return x


styled_df = df.style.apply(custom_style, axis=None)
print(styled_df.to_string())