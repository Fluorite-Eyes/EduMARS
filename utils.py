import json
import re
import requests
import base64
import mimetypes
import os

def encode_image_to_base64(image_path):
    """
    读取本地图片路径，转换为 Base64 编码字符串。
    格式: data:image/jpeg;base64,{code}
    """
    if not os.path.exists(image_path):
        print(f"Warning: 图片路径不存在 {image_path}")
        return None
        
    # 自动判断图片类型 (jpg, png, etc.)
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'image/jpeg' # 默认兜底

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    return f"data:{mime_type};base64,{encoded_string}"
def call_api(prompt_text,model_name,image_list=None,client=None):
    域名 = "https://www.dmxapi.cn/"  # 定义API的基础域名
    API_URL = 域名 + "v1/chat/completions"  # 完整的API请求URL
    API_KEY = "sk-uKPbG6VzFGT5Ku9IU9sSmSzeg34MbBFqcpHnndM0AP3m2ClP"

    user_content = [{"type": "text", "text": prompt_text}]
    
    if image_list:
        for img_path in image_list:
            base64_url = encode_image_to_base64(img_path)
            if base64_url:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": base64_url}
                })
    
    messages = [
        {"role": "user", "content": user_content}
    ]


    if client is not None:
       
        response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=4096,
        )
        
        content_str = response.choices[0].message.content

        return content_str
    

    payload = {
        "model": model_name,  # 指定使用的多模态AI模型，除了gpt-4o 也推荐使用 claude-3-5-sonnet系列
        "messages":messages,
        "temperature": 0.1,  # 设置生成文本的随机性，越低输出越有确定性
        "user": "DMXAPI",  # 发送请求的用户标识
    }
    
    headers = {
        "Content-Type": "application/json",  # 设置内容类型为JSON
        "Authorization": f"Bearer {API_KEY}",  # 使用 f-string 动态插入 API_KEY，进行身份验证
        "User-Agent": f"DMXAPI/1.0.0 ({域名})",  # 自定义的User-Agent，用于识别客户端信息
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    s = response.text
    data = json.loads(s)

    content_str = data['choices'][0]['message']['content']
    #content_str=data


    inner_data = content_str
    return inner_data