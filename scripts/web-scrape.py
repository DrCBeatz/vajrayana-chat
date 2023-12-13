import requests
from bs4 import BeautifulSoup


def main():
    url = input("Enter a URL: ")

    try:
        response = requests.get(url)

        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(f"Error{err}")
    else:
        soup = BeautifulSoup(response.text, "html.parser")

        text = " ".join(soup.get_text().split())

        output_file = input("Enter a filename for the output: ")
        with open(output_file, "w") as f:
            f.write(text)

        print(f"File saved to {output_file}")


if __name__ == "__main__":
    main()
