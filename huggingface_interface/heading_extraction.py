import base64
import requests
from pdf2image import convert_from_path
import os
from bs4 import BeautifulSoup
import time
import pathlib



def call_gptv4(image_path, prev_output):
    start_time = time.time()
    print('image path', image_path)
    # OpenAI API Key
    api_key = os.environ["OPENAI_API_KEY"]

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "temperature": 0.5,
    "top_p": 0.9,
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
#           "text": f'''Convert only the text present in the image to HTML without any CSS while maintaining the format with appropriate HTML headings. 
# Ensure each heading level corresponds to the correct text hierarchy and do not add any additional text beyond what is visible in the image. Use heading tags only for Chapter name, top level headings and sub sub-headings. Use paragraph tag otherwise. The output generated for previous page is as given below. Maintain the same formating as in the previous page to this page. \n {prev_output}
#                     '''
            # "text": f'''Convert only the text present in the image to HTML without any CSS while maintaining the format with appropriate HTML headings. Ensure each heading level corresponds to the correct text hierarchy and do not add any additional text beyond what is visible in the image. Use heading tags only for Chapter name, top level headings and sub sub-headings. Use paragraph tag otherwise. The main heading tags for generated text so far is given below. Using this information, maintain the same heading hierarchy for this page. \n {prev_output}'''
            # "text": f'''Convert only the text present in the image to HTML. Use these heading tags as reference \n {prev_output}. Ensure each heading level corresponds to the correct text hierarchy and do not add any additional text beyond what is visible in the image. Use heading tags only for Chapter name, top level headings and sub sub-headings. Use paragraph tag otherwise. Make sure all the text are include from the image. Don't skip any text at any case.'''
            "text": f"""Please extract all text from the image apart from the header starting from the beginning. Ensure that the output only contains the extracted text. A header is the topmost line of the page. Ignore this if no headers are present. Enclose the text within HTML heading tags such as <h1></h1>, <h2></h2>, <h3></h3>, <h4></h4>, <h5></h5>, and <h6></h6>. Use <h1> for the topmost heading and <h6> for the lowest level of heading. If the text doesn't fall under any heading and belongs to list items or other ordered list items like (1) or (i) or (a), enclose it under <p> tags. A heading is a line of text that marks the beginning of a context and is present on a newline with text not spanning more than one line. For example output:
<h1>CHAPTER 1</h1>
<h2>Preliminary</h2>
<h3>1. First heading under Chapter 1</h3>
<h4>1.1 First subheading under Chapter 1</h4>
<h5>1.1.1 First sub-subheading under Chapter 1</h5>
<h1>CHAPTER 2</h1>
<h2>Preliminary</h2>
<h3>2. First heading under Chapter 2</h3>
<h4>2.1 First subheading under Chapter 2</h4>
<h5>2.1.1 First sub-subheading under Chapter 2</h5>
Ensure that the heading tags for the subheadings are lower than the heading tags used for their chapter name. Use the previously generated tags for reference for this image. Make sure the level of headings from the reference has the same heading tags as the text on this page if they belong to the same layout category. This means if "Chapter 1" is in h1 in the reference and chapter 2 is in this page, then enclose chapter 2 in <h1> tag. Similarly, if 1.1.1 heading is in h4 tag in the reference and 2.1.4 is in this page, then put 2.1.4 in reference. The first page will not contain any previously generated output.\n{prev_output}\n"""
            # "text": f"""Divide the text into different layout of the document to which they belong. Some examples of categories include title, chapter name, etc."""
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
            }
            }
        ]
        }
    ],
    "max_tokens": 3000
    }
    print(f'prompt:\n', payload["messages"][0]["content"][0]["text"])
    # completion_with_backoff(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Once upon a time,"}])

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response.json())
    response_time = time.time() - start_time
    print(f"Full response received {response_time:.2f} seconds after request")
    try:
        return response.json()['choices'][0]['message']['content']
    except:
        print(response.json())
        time.sleep(30)
        call_gptv4(image_path, prev_output)


def find_headings(gpt_out):
    soup = BeautifulSoup(gpt_out, 'html.parser')
    headings_text = "\n".join([f"<{heading.name}>{heading.get_text()}</{heading.name}>" for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
    print(headings_text)
    return headings_text

def main_driver(pdf_path):
    

    pathlib.Path(f'{pdf_path}.txt').touch(exist_ok=True)
    print('Starting the program...')
    start_time = time.time()
    headings = ""
    # pdf_path = r'.\tempDir\FOOD-ACT_modified.pdf'
    file_name = pdf_path.split('/')[-1]
    images = convert_from_path(pdf_path, dpi=300)
    gpt_out = ""
    no_pgs = len(images)
    with open(f'./tempDir/{file_name}.txt', 'a', encoding='UTF-8') as f1:
        f1.write(f'<file_name>{file_name}</file_name>')
        f1.write(f'<pg_count>{no_pgs}</pg_count>')
        f1.write('<content>')
    # print(images)
    for page_num, i in enumerate(images):
        i.save(f"PDF_page.jpg")
        gpt_out = call_gptv4("PDF_page.jpg", headings)
        print(gpt_out)
        headings += find_headings(gpt_out)
        lines = gpt_out.split('\n')
        if lines[0].startswith('```html'):
            lines = lines[1:]
        if lines[-1].startswith('```'):
            lines = lines[:-1]
        gpt_out = '\n'.join(lines)
        with open(f'./tempDir/{file_name}.txt', 'a', encoding='UTF-8') as f1:
            f1.write(f'<page_{page_num}>\n{gpt_out}\n</page_{page_num}>')
        time.sleep(10)
    os.remove('PDF_page.jpg')
    with open(f'./tempDir/{file_name}.txt', 'a', encoding='UTF-8') as f1:
        f1.write('</content>')
    response_time = time.time() - start_time
    print(f"Execution of the program took {response_time:.2f} seconds to complete")
    return True

if __name__ == '__main__':
    main_driver(r"C:/Users/karth/OneDrive/Desktop/PDF Files/FOOD-ACT_modified.pdf")