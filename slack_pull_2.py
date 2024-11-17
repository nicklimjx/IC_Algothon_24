import logging
import os
import pandas as pd
import re
import csv
import time
# Import WebClient from Python SDK (github.com/slackapi/python-slack-sdk)
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# WebClient instantiates a client that can call API methods
# When using Bolt, you can use either `app.client` or the `client` passed to listeners.

def main():
    client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
    logger = logging.getLogger(__name__)

    channel_name = "the-challenge"
    channel_id = None
    phrase = "Data has just been released"
    password_pattern = r"Data\s+has\s+just\s+been\s+released\s+'(.*?)'"
    filename_pattern = r"the\spasscode\s+is\s+'(.*?)'."

    password_df = pd.read_csv("passwords.csv")

    try:
        # Call the conversations.list method using the WebClient
        for result in client.conversations_list():
            if channel_id is not None:
                break
            for channel in result["channels"]:
                if channel["name"] == channel_name:
                    channel_id = channel["id"]
                    #Print result
                    # print(f"Found conversation ID: {channel_id}")
                    break

    except SlackApiError as e:
        print(f"Error: {e}")
        
    # Store conversation history
    conversation_history = []
    # ID of the channel you want to send the message to

    oldest = password_df.iloc[-1]['timestamp'] + 1 # cus this is inclusive

    try:
        # Call the conversations.history method using the WebClient
        # The client passes the token you included in initialization    
        result = client.conversations_history(
            channel=channel_id,
            include_all_metadata=False,
            inclusive=True,
            oldest = oldest,
            limit=999
        )

        conversation_history = reversed(result["messages"])
        # # print(result)
        # for i, message in enumerate(result["messages"]):
        #     print(i, message['text'])

    except SlackApiError as e:
        print(f"Error: {e}")


    def checked(message, phrase):
        return phrase in message['text'] and message['user'] == 'U080GCRATP1'

    def append_to_csv(file_path, data, headers=None):
        """
        Appends data to a CSV file. If the file does not exist, it creates one with optional headers.

        :param file_path: Path to the CSV file.
        :param data: A list representing a single row or a list of lists for multiple rows.
        :param headers: Optional list of headers.
        """
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write headers if file does not exist and headers are provided
            if not file_exists and headers:
                writer.writerow(headers)
            # Write the data
            if isinstance(data[0], list):
                writer.writerows(data)
            else:
                writer.writerow(data)

    def append_row_to_csv(csv_file, row):
        """
        Appends a single row to an existing CSV file.

        :param csv_file: Path to the CSV file.
        :param row: List containing the row data.
        """
        try:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(row)
            print("Row appended successfully.")
        except Exception as e:
            print(f"Error appending row to CSV: {e}")

    def find_messages(phrase, conversation_history):

        for message in conversation_history:
            if checked(message, phrase):
                match = re.search(password_pattern, message['text'])
                password = match.group(1)
                match = re.search(filename_pattern, message['text'])
                filename = match.group(1)
                id, timestamp = message['user'], message['ts']
                row = [timestamp, id, password, filename]
                print(f"row added: {row}")
                append_row_to_csv("passwords.csv", row)


    find_messages(phrase, conversation_history)
    
main()
time_collect = time.time()
local_time = time.localtime(time_collect)
local_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
print(f"Slack read ran at {local_time}")