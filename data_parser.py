# -*- coding: utf-8 -*-

import json
import os
import re


def divide_strings_with_title(input_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)

    with_title = []
    without_title = []

    # deviding json to 2 lists : 1 with titles and one without titles
    for string in data:
        if string.startswith('제목: ') or string.startswith('title: ') or string.startswith('주제: '):
            with_title.append(string)
        else:
            without_title.append(string)

    return with_title, without_title


def format_strings_with_keys(strings_list, semester):
    formatted_strings = []

    for string in strings_list:
        title = ''
        subtitle = ''
        doc = ''

        # parsing title, subtitle, and doc content
        lines = string.strip().split('\n')
        for line in lines:
            if line.startswith('제목: ') or line.startswith('주제: ') or line.startswith('title: '):
                title = line.replace('제목: ', '').replace('title: ', '').replace('주제: ', '').strip()
            elif line.startswith('목표: ') or line.startswith('소개: ') or line.startswith('목차: '):
                subtitle = line.replace('목표: ', '').replace('소개: ', '').replace('목차: ', '').strip()
            else:
                if doc == '':
                    doc = line.strip()
                else:
                    doc += " " + line.strip()

        # keeping subtitle filled
        if subtitle == '': subtitle = '부재 : ' + title

        # keeping semester filled
        clean_semester = re.sub(r'[^0-9\-]', '', semester)

        # adding hyphen to semester
        if len(clean_semester) == 2: clean_semester = re.sub(r'(\d)(?=\d)', r'\1-', clean_semester)

        formatted_string = {'title': title, 'subtitle': subtitle, 'doc': doc, 'semester': clean_semester}

        formatted_strings.append(formatted_string)

    return formatted_strings


# writing to file and keeping utf-8
def write_json_file(output_file_path, formatted_data):
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(formatted_data, output_file, indent=4, ensure_ascii=False)


def clean_text(text):
    # Remove newline characters
    text = text.replace('\n', ' ').strip()

    # Remove redundant quotes
    text = text.replace('\"', '').strip()

    # Replace multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text


def process_files_recursive(folder_path, output_folder):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.json'):
                input_file_path = os.path.join(root, file_name)

                # divide json to titled list and non-titled list
                title_list, no_title_list = divide_strings_with_title(input_file_path)

                # formatting and cleaning titled list
                formatted_data = format_strings_with_keys(title_list, file_name)
                cleaned_data = []
                for record in formatted_data:
                    cleaned_record = {key: clean_text(value) for key, value in record.items()}
                    cleaned_data.append(cleaned_record)

                output_file_path = os.path.join(output_folder, 'processed_' + file_name)
                write_json_file(output_file_path, cleaned_data)


if __name__ == '__main__':
    input_folder_path = 'path/to/input/folder'
    output_folder_path = 'processed_data'

    # for labeling purposes
    prepend_subject = 'subject'

    process_files_recursive(input_folder_path, output_folder_path)