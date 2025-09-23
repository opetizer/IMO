import re
import os
import json
from openai import OpenAI  # 替换 ollama 为 openai

API_KEY = os.getenv("QWEN_API_KEY")  # 确保设置了 API_KEY 环境变量
BASE_URL = os.getenv("QWEN_BASE_URL")  # 确保设置了 BASE_URL 环境变量

# 正则清洗函数保持不变
def clean_text_regex(text):
    """使用正则表达式进行基础的文本格式清洗"""
    # 去除时间（如 12:34、2023-01-01 12:34:56）
    text = re.sub(r'\d{1,2}:\d{2}(:\d{2})?', '', text)
    text = re.sub(r'\d{4}-\d{1,2}-\d{1,2}( \d{1,2}:\d{2}(:\d{2})?)?', '', text)
    # 去除无意义字符（如特殊符号，保留常用标点和汉字、字母、数字）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：“”‘’（）《》【】,.!?;:"\'()\[\]<> \n]', '', text)
    # 合并多余空格和换行
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    # 去除行首尾空格
    text = '\n'.join(line.strip() for line in text.splitlines())
    return text

def llm_api_clean(input_path, output_path, api_key = API_KEY, base_url = BASE_URL, model_name = "qwen-plus"):
    """
    使用兼容OpenAI格式的云端大模型API来清洗和格式化文本。
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"错误：输入文件未找到 -> {input_path}")
        return

    # 初始化 OpenAI 客户端，但指向指定的云服务API
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # 将原来的长 prompt 分为 system 指令和 user 输入，这是标准的 API 格式
    # 这是在Python代码中使用的版本
    system_prompt = ("""
    You are an expert editor specializing in cleaning and formatting transcripts of official meeting records. 

    Your primary goal is to enhance readability by removing conversational noise and metadata, while strictly preserving the core content and meaning of the discussion.

    Please adhere strictly to the following rules:

    1. Thoroughly comprehend the entire document. Your analysis should go beyond keyword searching to a deep understanding of the text.

    2. Identify and extract all key information. This includes:

    - Facts: Verifiable statements and objective information.

    - Viewpoints: Opinions, positions, and arguments.

    - Data: Any numerical figures and statistics.

    3. Locate and extract the specific 'SUMMARY' section. This section may be explicitly titled "SUMMARY" or be identifiable as an executive summary or abstract.

    4. Process and clean the extracted text. Before generating the final output, process all the textual content you have extracted by removing the following categories of words and phrases:

    - Adverbs (e.g., fully, carefully, however, therefore, very, also).

    - Conjunctions (e.g., and, but, or, because, so).

    - Temporal Expressions, including all specific dates (e.g., 22 October 2021, next week) and times (e.g., at 5 PM, in the morning).
    
    - Filler Words (e.g., well, you know, like, I mean).
                     
    - Punctuation Marks (e.g., commas, periods, colons, semicolons, question marks, exclamation marks, parentheses, brackets, dashes, and ellipses).

    5. Structure the extracted information into a JSON format. The JSON output should conform to the following schema:

    - A root object.

    - This object must contain a key named "summary" whose value is an object containing the entire text and structured details found in the document's summary section.

    - The "summary" object must contain keys as follows, and each key must have type as specified:
    
    - "text": A string, containing the full text of the summary section after cleaning.

    - "strategic_direction": A list of numbers, representing the strategic direction of the meeting, if 'not applicable', return empty list.

    - "output": A list of numbers, each representing a key output, if 'not applicable', return empty list.

    - "action_to_be_taken": A list of numbers, refering to paragraph in the content, if 'not applicable', return empty list.

    - "related_document": A list of strings, divided by semicolon, comma or 'and', each being a document reference, remove 'resolution' in the result.
    
    - This object must also contain a key named "content" whose value is an array of objects.

    - Each object within the "content" array represents a paragraph from the main body of the document and must have two keys:

    - "paragraph": An integer representing the paragraph number (e.g., 1, 2, 3...).

    - "text": A string containing the full text of that paragraph.

    Your final output should be a single, well-formed JSON object that accurately reflects the document's summary and paragraph-by-paragraph content.

    """
    )
    
    user_prompt = "Please handle the following text to be processed according to your role and rules:\n\n" + raw_text

    print(f"正在使用模型 '{model_name}' 调用API进行清洗，请稍候...")
    
    try:
        # 使用 chat.completions.create 方法，这是标准的对话模型调用方式
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # 可以根据需要调整温度参数，0.0表示更确定性的输出
            temperature=0.0, 
            response_format={"type": "json_object"}
        )
        
        # 解析新格式的响应
        cleaned_text = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"调用API时发生错误: {e}")
        # 发生错误时，写入一个空的JSON结构，以防下游任务中断
        cleaned_text = json.dumps({"summary": {"text": ""}, "content": []})


    # 去除可能的<think>...</think>标签（一些模型可能会在内部思考过程中产生）
    cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL)
        
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f"处理完成！清洗后的文件已保存至: {output_path}")


if __name__ == "__main__":
    # --- 配置区域 ---
    # 强烈建议使用环境变量来存储API密钥，而不是硬编码在代码里
    # os.environ["QWEN_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # 临时测试用
    qwen_api_key = os.getenv("QWEN_API_KEY")

    if not qwen_api_key:
        print("错误：环境变量 QWEN_API_KEY 未设置。请先设置您的API密钥。")
    else:
        # 通义千问的API基础URL
        qwen_base_url = os.getenv("QWEN_BASE_URL", "https://api.qwen.ai/v1/chat/completions")
        # 选择一个合适的模型，例如 qwen-plus, qwen-max
        qwen_model = "qwen-plus"

        # 定义输入和输出文件路径
        input_txt = "data\MEPC\MEPC 77\MEPC 77-4 - Application for Basic Approval of the RADClean® BWMS (Islamic Republic of Iran).pdf"
        output_txt = f"{os.path.basename(input_txt).replace('.pdf', '_cleaned_by_qwen.txt')}"
        
        # 调用主函数
        llm_api_clean(
            input_path=input_txt, 
            output_path=output_txt, 
            api_key=qwen_api_key, 
            base_url=qwen_base_url, 
            model_name=qwen_model
        )