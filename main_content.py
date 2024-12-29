import argparse
import requests
from bs4 import BeautifulSoup
from pptx import Presentation
import openai
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from slack_sdk import WebClient

def get_inputs():
    parser = argparse.ArgumentParser(description="Automate prospect company research.")
    parser.add_argument("--website", required=True, help="Company website URL")
    parser.add_argument("--emails", required=True, help="Path to a .txt file with email addresses")
    args = parser.parse_args()
    return args.website, args.emails

def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    about_us = soup.find("section", {"id": "about"}) or soup.find("div", text="About Us")
    return about_us.text.strip() if about_us else "About Us not found."

def summarize_text(text):
    openai.api_key = "your_openai_api_key"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Summarize this text:\n{text}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

def create_presentation(summary):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Company Overview"
    content = slide.placeholders[1]
    content.text = summary
    prs.save("summary.pptx")

def send_email(recipients, subject, body, attachment):
    sender_email = "your_email@example.com"
    password = "your_email_password"
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    with open(attachment, "rb") as file:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={attachment}")
        msg.attach(part)
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        for recipient in recipients:
            server.sendmail(sender_email, recipient, msg.as_string())

def send_slack_message(summary):
    client = WebClient(token="your_slack_token")
    client.chat_postMessage(channel="#sales", text=f"New summary:\n{summary}")

if __name__ == "__main__":
    website, email_file = get_inputs()
    with open(email_file, "r") as file:
        emails = file.read().splitlines()
    scraped_text = scrape_website(website)
    summary = summarize_text(scraped_text)
    create_presentation(summary)
    send_email(emails, "Company Summary", summary, "summary.pptx")
    send_slack_message(summary)