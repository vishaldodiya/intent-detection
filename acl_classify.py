import fnmatch
import os
from random import shuffle
import json
import pickle


def get_all_files():
    files = []
    for root, dirnames, filenames in os.walk("acl"):
        for filename in fnmatch.filter(filenames, '*.json'):
            # if 'W06-3319' in filename:
            files.append(os.path.join(root, filename))

    shuffle(files)
    print(files)
    return files


def get_string_and_label(files = []):
    citations = []
    labels = []
    cite_context = []
    for file in files:
        f = open(file, )
        data = json.load(f)
        citation_contexts = list(data["citation_contexts"])
        f.close()
        for context in citation_contexts:
            if "citation_function" in context and "raw_string" in context and "cite_context" in context:
                labels.append(context["citation_function"])
                citations.append(context["raw_string"])
                cite_context.append(context["cite_context"])
    return citations, labels, cite_context


def main():
    files = get_all_files()
    citations, labels, cite_context = get_string_and_label(files)

    with open('citations.pkl', 'wb') as f:
        pickle.dump(citations, f)
    with open('labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    with open('cite_context.pkl', 'wb') as f:
        pickle.dump(cite_context, f)


if __name__ == '__main__':
    main()
