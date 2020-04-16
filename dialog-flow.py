
'''
###############################################
Final Project Code
Intent Detection
(using Google's DialogFlow)
CSE573 Semantic Web Mining
@Author: Tariq M Nasim
###############################################
'''


'''
-----------------------------------------------------------------------------------------------------
In order to run this code, you need to
- have access to a project created in DialogFlow
- configure corresponding access in Google Cloud Platform (GCP) console
- have API access information saved in your computer's environment.

In this code, replace the API_USER_NAME with your DialogFlow project_id.

Resources:
- DialogFlow Basics: https://cloud.google.com/dialogflow/docs/basics
- GCP, Services and Access control: https://cloud.google.com/dialogflow/docs/quick/setup
- Intent Management: https://cloud.google.com/dialogflow/docs/manage-intents
-----------------------------------------------------------------------------------------------------
'''

from time import sleep
import pandas as pd
import dialogflow_v2 as dialogflow

API_USER_NAME = "tnasim-byppov"

DEV_FILE_PATH = "dev.json"
TEST_FILE_PATH = "test.json"

# Delay between each intent detection API call.
# This is necessary for "Standard" version of GCP access.
# Otherwise Google will block the API calls after a few calls
DELAY = 1.0

# These two are also important. Otherwise the API calls throw error
TRAINING_PHRASE_LEN_LIMIT = 767
MAX_LEN_OF_INTENT_QUERY = 256

def detect_intent_with_sentiment_analysis(project_id, session_id, texts, label,
                                          language_code):

    session_client = dialogflow.SessionsClient()

    session_path = session_client.session_path(project_id, session_id)
    print('Session path: {}\n'.format(session_path))

    correct_assumption_count = 0
    average_confidence = 0
    total_queries = len(texts)

    for text in texts:
        if len(text) > MAX_LEN_OF_INTENT_QUERY:
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

        sleep(DELAY)

    return correct_assumption_count/total_queries, average_confidence/total_queries



def create_intent(project_id, display_name, training_phrases_parts,
                  message_texts):
    """
    Creates an intent with provided name and training phrases (and response messages)
    """
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

def main():

    # THE FOLLOWING LINES CAN BE USED FOR REGISTERING THE INTENTS ON THE DIALOGFLOW PROJECT
    # ------------------------------------------------------------------------------------------
    # dic_train = read_data(DEV_FILE_PATH)
    # create_all_intents(dic_train)
    # ------------------------------------------------------------------------------------------

    dic_test = read_data(TEST_FILE_PATH)
    print(dic_test.keys())

    for k in dic_test.keys():
        print(k, ": ", len(dic_test[k]))

    accuracy = {}
    confidence = {}
    for k in dic_test.keys():
        acc, conf = detect_intent_with_sentiment_analysis(API_USER_NAME, "12", dic_test[k], k, "en")

        accuracy[k] = acc
        confidence[k] = conf

    for k in dic_test.keys():
        print(k, "--> acc:", accuracy[k], ", conf: ", confidence[k])



    # ------------------------------------------------------------------------------------------
    # ROUGH CODE
    # ------------------------------------------------------------------------------------------
    # create_intent(API_USER_NAME,
    #               "method",
    #               dic["method"],
    #               ["method"])
    #create_intent("tnasim-byppov", "test1", ["this is test1 intent phrase", "another test1 phrase", "yet another test1 intent phrase"], ["test1message", "msgTest1"])
    #detect_intent_with_sentiment_analysis("tnasim-byppov", "12", ["how is the weather today", "what is the most rainy city in USA?"], "en")
    #print("hello")

main()
