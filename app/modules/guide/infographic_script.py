"""Declaration-only script validation. User JavaScript is never run by Python."""
import json
import math
import re

GROUPS = ['canvas', 'identity', 'modules', 'equipment', 'stats', 'radar', 'skills', 'teams', 'rotation', 'advice']
KEY = re.compile(r'^[A-Za-z][A-Za-z0-9_]{0,39}$')
CONTRACT = '''Canvas JS协议 v1：脚本在无网站登录态、无网络的独立Worker中执行。可自由修改JS函数和绘图逻辑，不限于现有模块。
顶部必须有一个JSON声明注释：/* @editor {"builtin":["canvas","identity","modules","equipment","stats","radar","skills","teams","rotation","advice"],"controls":[]} */。
builtin决定显示哪些已有控件组；controls定义新增参数，每项{key,label,type,default}，type为text/number/color/boolean/select；number可给min/max/step，select须给字符串options数组。最多32项，key为英文字母开头的字母数字下划线，最长40。label最长40，text值最长2000。不要使用__proto__/constructor/prototype。
参数值位于config.script_params。修改声明时同时调整script_params；不声明的参数不生成控件。已有config字段与当前值见上下文。
脚本须安装 window.GuideInfographicCanvas={render:async(canvas,config,data)=>{...;return {width:canvas.width,height:canvas.height,scene:{boards:[],fields:[],slots:[]},failed:0}}}。
canvas是OffscreenCanvas，ctx=canvas.getContext('2d')为原生Canvas2D；window为Worker自身的兼容别名，不是网页。没有document、DOM或站点凭据，不能联网或加载外部模块。
使用await __igLoadImage(path)读取data内已提供的本站图片，返回ImageBitmap，可ctx.drawImage；不能用new Image/fetch/importScripts。当前默认脚本已有兼容的image封装。
所有文字、布局、图形、图表、板块都可通过JS自定义；无需保持默认模块，但保留用户需要的内容。宽高每边1–16000，最多2400万像素，运行限时8秒。
可选scene定义画板交互：boards最多64个{key,label?,x,y,w,h}，key是安全标识符，对应config.positions[key]；fields最多256个{path,x,y,w,h,size}，path须指向config中已有字符串叶子，包括script_params.key；slots可沿用现有配队折叠元数据。未返回scene则仅预览/导出。
只输出纯JS，不需要HTML、script标签或Markdown围栏。JS不写入网站源码，只存当前用户浏览器和下载模板。'''


def declaration(source):
    if not isinstance(source, str) or len(source) > 80000:
        raise ValueError('绘图JS须为不超过80000字的文本。')
    matches = re.findall(r'/\*\s*@editor\s+([\s\S]*?)\*/', source)
    if len(matches) != 1 or len(matches[0]) > 16000:
        raise ValueError('绘图JS须包含一个有效的 @editor JSON参数声明。')
    try:
        spec = json.loads(matches[0])
    except ValueError:
        raise ValueError('@editor参数声明不是有效JSON。') from None
    if not isinstance(spec, dict) or set(spec) != {'builtin', 'controls'}:
        raise ValueError('@editor须包含builtin和controls。')
    groups, controls = spec['builtin'], spec['controls']
    if not isinstance(groups, list) or len(groups) > len(GROUPS) or any(not isinstance(k, str) or k not in GROUPS for k in groups) or len(set(groups)) != len(groups):
        raise ValueError('builtin控件组无效。')
    if not isinstance(controls, list) or len(controls) > 32:
        raise ValueError('自定义参数最多32项。')
    keys = set()
    for item in controls:
        if not isinstance(item, dict) or set(item) - {'key', 'label', 'type', 'default', 'min', 'max', 'step', 'options'} or not {'key', 'label', 'type', 'default'} <= set(item):
            raise ValueError('自定义参数字段无效。')
        key = item['key']
        if not isinstance(key, str) or not KEY.fullmatch(key) or key in ('constructor', 'prototype') or key in keys:
            raise ValueError('自定义参数名称无效或重复。')
        keys.add(key)
        if not isinstance(item['label'], str) or not 1 <= len(item['label']) <= 40 or item['type'] not in ('text', 'number', 'color', 'boolean', 'select'):
            raise ValueError('自定义参数类型或标签无效。')
        if item['type'] == 'number':
            for field in ('min', 'max', 'step'):
                if field in item and (type(item[field]) not in (int, float) or not math.isfinite(item[field])):
                    raise ValueError('数值参数边界无效。')
            if item.get('min', -1e12) > item.get('max', 1e12) or item.get('step', 1) <= 0:
                raise ValueError('数值参数范围无效。')
        if item['type'] == 'select' and (not isinstance(item.get('options'), list) or not 1 <= len(item['options']) <= 50 or any(not isinstance(v, str) or len(v) > 80 for v in item['options'])):
            raise ValueError('选择参数须提供有效options。')
        parameter(item, item['default'])
    return spec


def parameter(item, value):
    kind = item['type']
    valid = (kind == 'boolean' and type(value) is bool or
             kind == 'number' and type(value) in (int, float) and math.isfinite(value) and max(-1e12, item.get('min', -1e12)) <= value <= min(1e12, item.get('max', 1e12)) or
             kind == 'text' and isinstance(value, str) and len(value) <= 2000 or
             kind == 'color' and isinstance(value, str) and bool(re.fullmatch(r'#[0-9a-fA-F]{6}', value)) or
             kind == 'select' and isinstance(value, str) and value in item.get('options', []))
    if not valid:
        raise ValueError('自定义参数值不符合脚本声明。')


def validate(source, params):
    spec = declaration(source)
    if not isinstance(params, dict) or set(params) != {item['key'] for item in spec['controls']}:
        raise ValueError('脚本参数须与 @editor 声明一致。')
    for item in spec['controls']:
        parameter(item, params[item['key']])
    return spec
