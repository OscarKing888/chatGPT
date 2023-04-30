import tkinter as tk
from tkinter import filedialog

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("HTML/MHTML files", "*.html;*.mhtml")])
    return file_path

from bs4 import BeautifulSoup

def extract_chat_content2(html_file):
    with open(html_file, "r", encoding="utf-8") as file:
        content = file.read()

    soup = BeautifulSoup(content, "lxml")

    # 根据HTML结构，提取提问和回答元素
    questions = soup.find_all("div", class_="question")
    answers = soup.find_all("div", class_="answer")

    return questions, answers


def extract_chat_content3(html_content):
    soup = BeautifulSoup(html_content, "lxml")

    # 根据HTML结构，提取提问和回答元素
    questions = soup.find_all("div", class_="min-h-[20px] flex flex-col items-start gap-4 whitespace-pre-wrap break-words")
    answers = soup.find_all("div", class_="min-h-[20px] flex flex-col items-start gap-4 whitespace-pre-wrap break-words")

    return questions, answers

def extract_chat_content(content):
    soup = BeautifulSoup(content, "lxml")

    # 根据HTML结构，提取提问和回答元素
    questions = soup.find_all("div", class_="min-h-[20px] flex flex-col items-start gap-4 whitespace-pre-wrap break-words")
    answers = soup.find_all("div", class_="markdown prose w-full break-words dark:prose-invert dark")

    return questions, answers

def merge_interrupted_answers(questions, answers):
    merged_questions = []
    merged_answers = []
    
    i = 0
    while i < len(questions):
        question = questions[i]
        answer = answers[i]
        
        while i + 1 < len(questions) and (questions[i + 1].text.strip().lower() == "继续" or questions[i + 1].text.strip().lower() == "go on"):
            i += 1
            answer.append(answers[i])
        
        merged_questions.append(question)
        merged_answers.append(answer)
        i += 1
    
    return merged_questions, merged_answers

def create_new_html(questions, answers, input_html):
    with open(input_html, "r", encoding="utf-8") as file:
        content = file.read()

    soup = BeautifulSoup(content, "lxml")
    
    # 从HTML中找到第一个问题，然后在其前面创建一个名为chat_content的新div元素。这将作为新问题和回答的容器。
    first_question = soup.find("div", class_="min-h-[20px] flex flex-col items-start gap-4 whitespace-pre-wrap break-words")
    if first_question is None:
        raise ValueError("Invalid HTML file. Cannot find the first question.")
    chat_content = soup.new_tag("div", id="chat_content")
    first_question.insert_before(chat_content)


    for i, (question, answer) in enumerate(zip(questions, answers)):
        chat_content.append(question)
        chat_content.append(answer)
        checkbox = soup.new_tag("input", type="checkbox", id=f"checkbox-{i}")
        chat_content.append(checkbox)

    
def create_new_html3(questions, answers, input_html):
    with open(input_html, "r", encoding="utf-8") as file:
        content = file.read()

    soup = BeautifulSoup(content, "lxml")  # 将 input_html 改为 content
    chat_content = soup.find("div", class_="min-h-[20px] flex flex-col items-start gap-4 whitespace-pre-wrap break-words")

    if chat_content is None:
        raise ValueError("Invalid HTML file. Cannot find chat_content.")

    chat_content.clear()

    # 添加带有checkbox的聊天内容
    for i, (question, answer) in enumerate(zip(questions, answers)):
        chat_content.append(question)
        chat_content.append(answer)
        checkbox = soup.new_tag("input", type="checkbox", id=f"checkbox-{i}")
        chat_content.append(checkbox)

    return soup.prettify()

def create_new_html2(questions, answers, input_html):
    with open(input_html, "r", encoding="utf-8") as file:
        content = file.read()

    soup = BeautifulSoup(content, "lxml")
    chat_content = soup.find("div", id="chat_content")

    if chat_content is None:
        raise ValueError("Invalid HTML file. Cannot find chat_content.")

    chat_content.clear()

    # 添加带有checkbox的聊天内容
    for i, (question, answer) in enumerate(zip(questions, answers)):
        chat_content.append(question)
        chat_content.append(answer)
        checkbox = soup.new_tag("input", type="checkbox", id=f"checkbox-{i}")
        chat_content.append(checkbox)

    return soup.prettify()

def add_export_button(html):
    soup = BeautifulSoup(html, "lxml")
    body = soup.find("body")
    export_button = soup.new_tag("button", id="export_button")
    export_button.string = "导出Html"
    body.append(export_button)
    return soup.prettify()

def add_javascript(html):
    soup = BeautifulSoup(html, "lxml")

    script = soup.new_tag("script")
    script.string = '''
    document.getElementById("export_button").addEventListener("click", function() {
        let selected_questions = [];
        let selected_answers = [];
        let checkboxes = document.querySelectorAll("input[type=checkbox]");

        for (let i = 0; i < checkboxes.length; i++) {
            if (checkboxes[i].checked) {
                selected_questions.push(document.getElementsByClassName("question")[i].outerHTML);
                selected_answers.push(document.getElementsByClassName("answer")[i].outerHTML);
            }
        }

        let new_html = `<!DOCTYPE html>
<html>
<head>
    <!-- Copy the original style here -->
</head>
<body>
    <div id="chat_content">` + selected_questions.map((q, i) => q + selected_answers[i]).join("") + `</div>
</body>
</html>`;

        let a = document.createElement("a");
        a.href = "data:text/html;charset=utf-8," + encodeURIComponent(new_html);
        a.download = "exported_chat.html";
        a.style.display = "none";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });
    '''

    soup.head.append(script)
    return soup.prettify()

def save_new_html(html):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(defaultextension=".mhtml", filetypes=[("HTML files", "*.mhtml")])

    if not file_path:
        return

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(html)

import tkinter as tk
from tkinter import filedialog
from bs4 import BeautifulSoup

# 定义所有的函数
# ... 将上述代码片段中的所有函数定义粘贴到这里 ...
from requests_html import HTMLSession

def convert_mhtml_to_html(file_path):
    if file_path.lower().endswith(".mhtml"):
        session = HTMLSession()
        with open(file_path, "rb") as f:
            content = f.read()
        r = session.post('https://www.example.com', data=content)
        return r.text
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content

def extract_chat_content(html_content):
    soup = BeautifulSoup(html_content, "lxml")

    # 根据HTML结构，提取提问和回答元素
    questions = soup.find_all("div", class_="question")
    answers = soup.find_all("div", class_="answer")

    return questions, answers


if __name__ == "__main__":
    
    # 主代码部分
    input_file = select_file()
    html_content = convert_mhtml_to_html(input_file)
    questions, answers = extract_chat_content(html_content)
    questions, answers = merge_interrupted_answers(questions, answers)
    new_html = create_new_html(questions, answers, input_html)
    new_html = add_export_button(new_html)
    new_html = add_javascript(new_html)
    save_new_html(new_html)
