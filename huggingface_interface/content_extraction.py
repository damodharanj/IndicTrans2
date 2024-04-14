from bs4 import BeautifulSoup

# Opening the file with better error handling
with open(r"C:\Users\karth\OneDrive\Desktop\Project\IndicTrans2\huggingface_interface\html_output.txt", "r", encoding="UTF-8") as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, 'html.parser')
user_input = "Short title, extent and commencement"

# Assuming 'content' is the main container for pages
content = soup.find('content')
pages = content.find_all(['page_0', 'page_1', 'page_2', 'page_3', 'page_4', 'page_5'])  # Adjust according to actual tags
found_heading = None
next_headings = []
spanning_pages = []

heading_tags = []

# Find the relevant headings and content
for page in pages:
    heading_tags += page.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])


for tag in heading_tags:
    if user_input.lower() in tag.text.strip().lower():
        found_heading = tag
        break

stop_heading = None

# Define the stop condition based on subsequent headings
if found_heading:
    index = heading_tags.index(found_heading)
    for i in range(index + 1, len(heading_tags)):
        if found_heading.name < heading_tags[i].name:
            next_headings.append(heading_tags[i])
        else:
            stop_heading = heading_tags[i]
            break

else:
    print('No such heading found!')
# Extract content between found_heading and stop_heading or to the document's end
starting_content_tag = found_heading
ending_content_tag = stop_heading


current_page_index = pages.index(found_heading.find_parent(['page_0', 'page_1', 'page_2', 'page_3', 'page_4', 'page_5']))
current_tag = found_heading.find_next_sibling()

spanning_pages.append(pages[current_page_index].name)
while current_tag or current_page_index < len(pages) - 1:
    if not current_tag:  # Move to the next page if current page is exhausted
        current_page_index += 1
        current_tag = pages[current_page_index].find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'ol', 'ul'])
        spanning_pages.append(pages[current_page_index].name)

    # Check if the current tag is the ending tag
    if current_tag == ending_content_tag:
        break
    
    if current_tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'ol', 'ul']:
        print(current_tag.text)

    current_tag = current_tag.find_next_sibling()

print("Content spans these pages:", spanning_pages)

