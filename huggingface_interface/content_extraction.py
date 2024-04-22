from bs4 import BeautifulSoup


def extract_content(filename, user_input):
    # Opening the file with better error handling
    with open(f"./tempDir/{filename}.txt", "r", encoding="UTF-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    # user_input = "Composition of Food Authority and qualifications for appointment of its Chairperson and other Members"
    file_name = soup.find('file_name').text
    no_pgs = int(soup.find('pg_count').text)
    print(file_name)
    print(no_pgs)
    # Assuming 'content' is the main container for pages
    content = soup.find('content')
    pages = content.find_all([f'page_{i}' for i in range(no_pgs)])  # Adjust according to actual tags
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
        return ['No such heading found!', []]
    # Extract content between found_heading and stop_heading or to the document's end
    starting_content_tag = found_heading
    ending_content_tag = stop_heading

    print(starting_content_tag, ending_content_tag)
    current_page_index = pages.index(found_heading.find_parent([f'page_{i}' for i in range(no_pgs)]))
    current_tag = found_heading.find_next_sibling()

    spanning_pages.append(pages[current_page_index].name)
    res = ""
    while current_tag or current_page_index < len(pages) - 1:
        if not current_tag:  # Move to the next page if current page is exhausted
            current_page_index += 1
            current_tag = pages[current_page_index].find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'ol', 'ul'])
            spanning_pages.append(pages[current_page_index].name)

        # Check if the current tag is the ending tag
        if current_tag == ending_content_tag:
            break
        
        if current_tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'ol', 'ul']:
            # print(current_tag.text)
            res += current_tag.text

        current_tag = current_tag.find_next_sibling()

    print("Content spans these pages:", spanning_pages)
    parsed_pages = [str(int(i.split('page_')[1])+1) for i in spanning_pages]
    print(res)
    return [res, parsed_pages]

if __name__ == '__main__':
    extract_content('./html2.txt', 'Short title, extent and commencement')