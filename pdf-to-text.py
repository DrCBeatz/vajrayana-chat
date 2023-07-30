from PyPDF2 import PdfReader
import os


def list_files_in_directory(directory):
    return [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]


def write_file(file_name, text):
    with open(f"{file_name}.txt", "w") as f:
        f.write(text)


# Usage
file_list = list_files_in_directory("thrangu-rinpoche-pdfs/")
print(file_list)

for file in file_list:
    # creating a pdf reader object
    try:
        reader = PdfReader(f"thrangu-rinpoche-pdfs/{file}")

        # printing number of pages in pdf file
        print(len(reader.pages))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        try:
            write_file(f"texts/{file}", text)
            print(f"Saved PDF to text file {file}.txt")
        except:
            print(f"Error saving {file}")

    except:
        print(f"Error reading {file}")
        continue
