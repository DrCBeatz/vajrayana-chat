from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup as Soup

import argparse

parser = argparse.ArgumentParser(description="Get transcript from YouTube video")
parser.add_argument("url", type=str, help="URL of YouTube video")
args = parser.parse_args()
url = args.url if args.url else input("Enter URL: ")


def write_file(file_name, text):
    with open(f"{file_name}.txt", "w") as f:
        f.write(text)


def main():
    page = requests.get(url)
    print(url)
    soup = Soup(page.text, "html.parser")
    title = soup.title.text

    video_id = url.replace("https://youtu.be/", "")
    video_id = url.replace("https://www.youtube.com/watch?v=", "")
    print(video_id)

    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    output = ""
    for x in transcript:
        sentence = x["text"]
        output = f"{output} {sentence}\n"

    try:
        write_file(title, output)
        print(f"Saved transcript to file {title}.txt")
    except:
        print("Error saving file")


if __name__ == "__main__":
    main()
