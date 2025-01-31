from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests
from bs4 import BeautifulSoup
from phi.agent import Agent
import json
from phi.model.openai import OpenAIChat
from pymongo import MongoClient
import re
import os


# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["rssfeed"]
collection = db["mcqs"]
# Define the agent
agent = Agent(
    name="Web Crawl Tool",
    monitoring=True,
    show_tool_calls=True,
    debug_mode=True,
    model=OpenAIChat(id="gpt-4o"),
    instructions=['Always return proper json such as json'],
    description="You are an assistant that generates MCQ questions. First plan, then observe, then provide the results",
)

# Function to truncate text
def truncate_text(text, max_length=500):
    return text[:max_length] + "..." if len(text) > max_length else text

# Function to fetch RSS feeds and generate MCQs
def fetch_and_generate_mcqs_json(rss_urls):
    results = []
    for rss_url in rss_urls:
        try:
            response = requests.get(rss_url)
            if response.status_code != 200:
                print(f"Failed to fetch RSS feed from {rss_url}. Status code: {response.status_code}")
                continue

            soup = BeautifulSoup(response.content, "xml")
            items = soup.find_all("item")

            for item in items[:2]:  # Limit to the first 2 articles per feed
                title = truncate_text(item.find("title").text, max_length=100)
                link = item.find("link").text
                description = truncate_text(item.find("description").text if item.find("description") else 'No description provided.', max_length=300)

                try:
                    prompt = f"Create a multiple-choice question based on the following article description:\n\n{description}\n\n" \
                             "Generate an MCQ with the following structure: a question key, an array with options as keys a, b, c, d, and a key for the correct answer. " \
                             "Also include an explanation for the answer in the 'explanation' key and return the data in json format."

                    mcq_response = agent.run(prompt)

                    if isinstance(mcq_response, str):
                        mcq_data = json.loads(mcq_response)
                    elif hasattr(mcq_response, "content"):
                        mcq_data = json.loads(mcq_response.content)
                    else:
                        print(f"Unexpected response type for article '{title}': {type(mcq_response)}, value: {mcq_response}")
                        raise ValueError("MCQ response is not in a parsable format.")

                    question = mcq_data.get("question")
                    options = mcq_data.get("options")
                    correct_answer = mcq_data.get("correct_answer")
                    explanation = mcq_data.get("explanation")

                    results.append({
                        "feed_url": rss_url,
                        "title": title,
                        "link": link,
                        "description": description,
                        "question": question,
                        "options": options,
                        "correct_answer": correct_answer,
                        "explanation": explanation
                    })

                except Exception as e:
                    print(f"Failed to generate MCQ for article '{title}' from feed '{rss_url}': {e}")

        except Exception as e:
            print(f"An error occurred while processing RSS feed {rss_url}: {e}")

    return results

# Function to insert data into MongoDB
def insert_mcqs_to_mongodb(mcqs):
    try:
        if mcqs:
            result = collection.insert_many(mcqs)
            print(f"Inserted {len(result.inserted_ids)} documents into MongoDB.")
        else:
            print("No MCQs to insert.")

    except Exception as e:
        print(f"An error occurred while inserting data into MongoDB: {e}")


def is_valid_rss_feed(url):
    """
    Validate if a given URL is a proper RSS feed.
    """
    try:
        # Check if URL is valid
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # or IPv4...
            r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # or IPv6...
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE
        )
        
        if not re.match(regex, url):
            print(f"Invalid URL format: {url}")
            return False
        
        # Make a request to check if the URL is reachable
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            print(f"Failed to access URL {url}. HTTP Status Code: {response.status_code}")
            return False

        # Check if it's a valid RSS feed
        soup = BeautifulSoup(response.content, "xml")
        if soup.find("rss") or soup.find("feed"):
            return True
        else:
            print(f"Invalid RSS format for URL: {url}")
            return False

    except requests.RequestException as e:
        print(f"Error while validating RSS feed {url}: {e}")
        return False

# Home page - Input RSS feed URLs
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # rss_urls = request.form.get("rss_urls").splitlines()

        rss_urls = request.form.get("rss_urls").split(",")

        rss_urls = [url.strip() for url in rss_urls]

        # Validate RSS feeds
        valid_rss_urls = [url for url in rss_urls if is_valid_rss_feed(url)]
        
        if not valid_rss_urls:
            return render_template("index.html", error="No valid RSS feeds provided"), 400
            # return jsonify({"error": "No valid RSS feeds provided"}), 400
        mcqs_json = fetch_and_generate_mcqs_json(rss_urls)
        insert_mcqs_to_mongodb(mcqs_json)
        return redirect(url_for("view_mcqs"))
    return render_template("index.html")


@app.route("/view-mcqs")
def view_mcqs():
    try:
        # Get the page number from the query parameters, default to 1 if not provided
        page = int(request.args.get('page', 1))
        per_page = 10  # Number of MCQs per page

        # Calculate the number of documents to skip
        skip = (page - 1) * per_page

        # Fetch the MCQs with pagination
        mcqs = list(collection.find({}, {"_id": 0}).sort("_id", -1).skip(skip).limit(per_page))

        # Get the total number of MCQs for pagination
        total_mcqs = collection.count_documents({})

        # Calculate the total number of pages
        total_pages = (total_mcqs + per_page - 1) // per_page

        # Render the template with MCQs and pagination information
        return render_template("view_mcqs.html", mcqs=mcqs, page=page, total_pages=total_pages)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)