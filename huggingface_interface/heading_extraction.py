import base64
import requests
from pdf2image import convert_from_path
import os
from bs4 import BeautifulSoup
import time



def call_gptv4(image_path, prev_output):
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
    "temperature": 0,
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
#             "text": f'''
# Convert only the text present in the image to HTML without any CSS while maintaining the format with appropriate HTML headings. 
# Ensure each heading level corresponds to the correct text hierarchy and do not add any additional text beyond what is visible in the image. Use heading tags only for Chapter name, top level headings and sub sub-headings. Use paragraph tag otherwise. The output generated for previous page is as given below. Maintain the same formating as in the previous page to this page. \n {prev_output}
#                     '''
            # "text": f'''Convert only the text present in the image to HTML without any CSS while maintaining the format with appropriate HTML headings. Ensure each heading level corresponds to the correct text hierarchy and do not add any additional text beyond what is visible in the image. Use heading tags only for Chapter name, top level headings and sub sub-headings. Use paragraph tag otherwise. The main heading tags for generated text so far is given below. Using this information, maintain the same heading hierarchy for this page. \n {prev_output}'''
            "text": f'''Convert only the text present in the image to HTML without any CSS while maintaining the format with appropriate HTML headings. Use these heading tags as reference \n {prev_output}. Ensure each heading level corresponds to the correct text hierarchy and do not add any additional text beyond what is visible in the image. Use heading tags only for Chapter name, top level headings and sub sub-headings. Use paragraph tag otherwise. Make sure all the text are include from the image. Don't skip any text at any case.'''
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
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
    print(response)
    return response.json()['choices'][0]['message']['content']


def find_headings(gpt_out):
    soup = BeautifulSoup(gpt_out, 'html.parser')
    headings_text = "\n".join([f"<{heading.name}>{heading.get_text()}</{heading.name}>" for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
    print(headings_text)
    return headings_text

def main_driver(pdf_path):
    headings = ""
    pdf_path = r'.\tempDir\FOOD-ACT_modified.pdf'
    f1 = open(r'test_output.txt', 'a', encoding='UTF-8')
    images = convert_from_path(pdf_path, dpi=300)
    gpt_out = ""
    # print(images)
    for page_num, i in enumerate(images):
        i.save(f"PDF_page.jpg")
        gpt_out = call_gptv4("PDF_page.jpg", headings)
        print(gpt_out)
        headings += find_headings(gpt_out)
        f1.write(f'Page {page_num}:\n{gpt_out}\n')
        time.sleep(10)
    os.remove('PDF_page.jpg')
    f1.close()

if __name__ == '__main__':
    main_driver('')