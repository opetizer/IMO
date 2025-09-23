import re
from ollama import Client

def clean_text(text):
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

def ollama_clean(input_path, output_path):
	with open(input_path, 'r', encoding='utf-8') as f:
		raw_text = f.read()

	# 使用 deepseek-r1 进行进一步清洗（可选，示例为本地清洗）
	# 如果需要用 deepseek-r1 进一步理解和清洗，可以这样：
	client = Client()
	# 针对名单格式的优化提示词
	prompt = (
		"你是一个专门处理官方文件的数据提取与格式化助手。\n"
		"请严格按照以下规则，对提供的参会人员名单文本进行处理，生成一份干净、整洁且结构化的版本。\n\n"
		"处理规则：\n"
		"1.  **删除所有文档元数据**：彻底移除所有页眉和页脚信息，包括但不限于：\n"
		"    - 文件路径，如 `O:\MEPC 77.INF.1.docx`。\n"
		"    - 文档代码、会话信息和日期，如 `MEPC 77/INF.1`、`77th session`、`8 December 2021`。\n"
		"    - 页面标记，如 `- 2 -`、`E`、`ENGLISH ONLY`。\n\n"
		"2.  **删除会议概要信息**：移除文件开头的会议委员会名称、会话主席（Chair）和副主席（Vice Chair）等介绍性内容。\n\n"
		"3.  **格式化参会者条目**：\n"
		"    - **保留国家或组织标题**，并将其作为一级标题（例如 `ALGERIA`, `HONG KONG, CHINA`）。\n"
		"    - **保留职位分组标题**，并将其作为二级标题（例如 `Head of Delegation`, `Representatives`, `Advisers`）。\n"
		"    - **合并个人信息**：将每位参会者的'姓名'、'职位'和'所属机构'合并到【单一行】，并用逗号和空格 `, ` 分隔。例如，将以下格式：\n"
		"      Mr. Mohamed Khelifi, Alternate Permanent Representative of Algeria to the IMO, \n"
		"      Embassy of Algeria, London\n"
		"      转换成：\n"
		"      Mr. Mohamed Khelifi, Alternate Permanent Representative of Algeria to the IMO, Embassy of Algeria, London\n\n"
		"4.  **清理空分组和多余空白**：\n"
		"    - 如果一个职位分组标题（如 `Head of Delegation`）下面没有任何参会者姓名，则将该职位分组标题也一并删除。\n"
		"    - 将多个连续的空行压缩为一个空行，以保持不同国家或分组间的区隔，但避免版面过于稀疏。\n\n"
		"5.  **保留完整结构**：保留从国家代表团到国际组织观察员、秘书处等所有章节，并应用相同的格式化规则。\n\n"
		"核心原则：\n"
		"-   **保持原始顺序**：必须严格保持国家、组织以及人员在原始文件中的出现顺序。\n"
		"-   **内容绝对保真**：绝不修改、缩写或变更任何姓名、职位和机构名称。\n"
		"-   **纯净输出**：你的输出只能包含格式化后的名单本身，禁止添加任何前言、注释或总结性文字。\n\n"
		"待处理的文本如下：\n\n" + raw_text
	)
	response = client.generate(model="deepseek-r1:8b", prompt=prompt)
	cleaned_text = response['response'].strip()

	# 去除<think>...</think>之间的内容
	cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL)
	
	# cleaned_text = clean_text(raw_text)  # 本地清洗
	# cleaned_text = clean_text(cleaned_text)  # 本地清洗

	
	# 也可以只用本地 clean_text，或结合两者
	# cleaned_text = clean_text(raw_text)

	if not cleaned_text:
		print("清洗后的文本为空，请检查输入文件内容。")
		return
	import os
	if not os.path.exists(os.path.dirname(output_path)):
		os.makedirs(os.path.dirname(output_path))
	with open(output_path, 'w', encoding='utf-8') as f:
		f.write(cleaned_text)

if __name__ == "__main__":
	input_txt = "extracted_texts\MEPC\MEPC 77\MEPC 77-1 - for the seventy-seventh session of the Marine Environment Protection Committee to be held... (Secretariat).txt"
	output_txt = "cleaned_texts\MEPC\MEPC 77\MEPC 77-1 - for the seventy-seventh session of the Marine Environment Protection Committee to be held... (Secretariat).txt"
	ollama_clean(input_txt, output_txt)