import re


def remove_handle(tweet):
    match = re.findall("@[\w]*",tweet)
    for i in match:
        tweet = re.sub(i,'',tweet)
    return tweet




text = "I am happy. Thank you for asking @user @shutputpt"
print(remove_handle(text))