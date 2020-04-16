'''
Final Project Code
Intent Detection
(using Google's DialogFlow)
CSE573 Semantic Web Mining
@Author: Tariq M Nasim
'''

from time import sleep
import pandas as pd
import numpy as np
import re
import dialogflow_v2 as dialogflow

API_USER_NAME = "tnasim-byppov"

DEV_FILE_PATH = "dev.json"
TEST_FILE_PATH = "test.json"
TRAINING_PHRASE_LEN_LIMIT = 767

def detect_intent_with_sentiment_analysis(project_id, session_id, texts, label,
                                          language_code):
    """Returns the result of detect intent with texts as inputs and analyzes the
    sentiment of the query text.

    Using the same `session_id` between requests allows continuation
    of the conversation."""
    session_client = dialogflow.SessionsClient()

    session_path = session_client.session_path(project_id, session_id)
    print('Session path: {}\n'.format(session_path))

    correct_assumption_count = 0
    average_confidence = 0
    total_queries = len(texts)

    for text in texts:
        if len(text) > 256:
            total_queries = total_queries - 1
            continue

        text_input = dialogflow.types.TextInput(
            text=text, language_code=language_code)

        query_input = dialogflow.types.QueryInput(text=text_input)

        # Enable sentiment analysis
        sentiment_config = dialogflow.types.SentimentAnalysisRequestConfig(
            analyze_query_text_sentiment=True)

        # Set the query parameters with sentiment analysis
        query_params = dialogflow.types.QueryParameters(
            sentiment_analysis_request_config=sentiment_config)

        response = session_client.detect_intent(
            session=session_path, query_input=query_input,
            query_params=query_params)

        if response.query_result.intent.display_name == label:
            correct_assumption_count = correct_assumption_count + 1
            average_confidence = average_confidence + response.query_result.intent_detection_confidence

        print('=' * 20)
        print('Query text: {}'.format(response.query_result.query_text))
        print('Detected intent: {} (confidence: {})\n'.format(
            response.query_result.intent.display_name,
            response.query_result.intent_detection_confidence))
        print('Fulfillment text: {}\n'.format(
            response.query_result.fulfillment_text))
        # Score between -1.0 (negative sentiment) and 1.0 (positive sentiment).
        print('Query Text Sentiment Score: {}\n'.format(
            response.query_result.sentiment_analysis_result
            .query_text_sentiment.score))
        print('Query Text Sentiment Magnitude: {}\n'.format(
            response.query_result.sentiment_analysis_result
            .query_text_sentiment.magnitude))
        sleep(1.0)

    return correct_assumption_count/total_queries, average_confidence/total_queries


'''
Creates an intent with provided name and training phrases (and response messages)
'''
def create_intent(project_id, display_name, training_phrases_parts,
                  message_texts):
    """Create an intent of the given intent type."""
    intents_client = dialogflow.IntentsClient()

    parent = intents_client.project_agent_path(project_id)
    training_phrases = []
    for training_phrases_part in training_phrases_parts:
        part = dialogflow.types.Intent.TrainingPhrase.Part(
            text=training_phrases_part)
        # Here we create a new training phrase for each provided part.
        training_phrase = dialogflow.types.Intent.TrainingPhrase(parts=[part])
        training_phrases.append(training_phrase)

    text = dialogflow.types.Intent.Message.Text(text=message_texts)
    message = dialogflow.types.Intent.Message(text=text)

    intent = dialogflow.types.Intent(
        display_name=display_name,
        training_phrases=training_phrases,
        messages=[message])

    # for p in training_phrases:
    #     print(  )
    response = intents_client.create_intent(parent, intent)

    print('Intent created: {}'.format(response))


def create_all_intents(dic):
    for key in dic:
        texts = dic[key] # key == label in the json file
        create_intent(API_USER_NAME,
                      key,
                      texts,
                      [key])


'''
Read data from a json file and group them in a dictionary. keys == labels, values == corresponding texts
returns the dictionary
'''
def read_data(file_path):

    data = pd.read_json(file_path);
    strings = data["string"]
    label = data["label"]
    dic = {}
    for i in range(len(strings)):
        # remove training phrases if len > the limit
        if len(strings[i]) == 0 or len(strings[i]) > TRAINING_PHRASE_LEN_LIMIT:
            continue

        if label[i] in dic:
            dic[label[i]].append(strings[i])
        else:
            dic[label[i]] = [strings[i]]

    return dic

def read_test_data(file_path):
    # Using readlines()
    file1 = open(file_path, 'r')
    Lines = file1.readlines()

    strings = []
    label = []

    # Strips the newline character
    for line in Lines:
        jd = eval(line.strip().replace(": true,", ": True,").replace(": false,", ": False,").replace("NaN", "None"))
        strings.append(jd["string"])
        label.append(jd["label"])

    dic = {}
    for i in range(len(strings)):
        # remove training phrases if len > the limit
        if len(strings[i]) == 0 or len(strings[i]) > TRAINING_PHRASE_LEN_LIMIT:
            continue

        if label[i] in dic:
            dic[label[i]].append(strings[i])
        else:
            dic[label[i]] = [strings[i]]

    return dic


def main():
    # dic_train = read_data(DEV_FILE_PATH)
    # create_all_intents(dic_train)

    dic_test = read_test_data(TEST_FILE_PATH)
    print(dic_test.keys())

    # for k in dic_test.keys():
    #     print(k, ": ", len(dic_test[k]))

    accuracy = {}
    confidence = {}
    for k in dic_test.keys():
        acc, conf = detect_intent_with_sentiment_analysis("tnasim-byppov", "12", dic_test[k][0:255], k, "en")

        accuracy[k] = acc
        confidence[k] = conf

    for k in dic_test.keys():
        print(k, "--> acc:", accuracy[k], ", conf: ", confidence[k])



    # create_intent(API_USER_NAME,
    #               "method",
    #               dic["method"],
    #               ["method"])
    #create_intent("tnasim-byppov", "test1", ["this is test1 intent phrase", "another test1 phrase", "yet another test1 intent phrase"], ["test1message", "msgTest1"])
    #detect_intent_with_sentiment_analysis("tnasim-byppov", "12", ["how is the weather today", "what is the most rainy city in USA?"], "en")
    #print("hello")

main()
