import pandas as pd
from pydantic import BaseModel
from openai import OpenAI
client = OpenAI(api_key="sk-None-D0YFDMaeYe7ettHcjAtaT3BlbkFJRgN2uhoxlefWuLQLDiSB")
from tqdm import tqdm


fname_20ng = "random_100_topic_words_20ng_with_anchors.csv"
fname_agris = "random_100_topic_words_agris_with_anchors.csv"
fname_tweetsnyr = "random_100_topic_words_tweetsnyr_with_anchors.csv"

#df_20ng = pd.read_csv(fname_20ng, sep="\t", encoding='utf-8')
#df_agris = pd.read_csv(fname_agris, sep="\t", encoding='utf-8')
df_agris = pd.read_csv(fname_tweetsnyr, sep="\t", encoding='utf-8', dtype={"ID": str})


for i in tqdm(range(len(df_agris))):
    topic_words = str(df_agris.loc[i, 'Topic words'].split(' ')).strip("[]")
    anchor = df_agris.loc[i, 'Anchor']
    prompt = f"""Given a list of words: {topic_words}, can you generate a word that is semantically equivalent to "{anchor}" based on the meaning of "{anchor}" in this context?"""
    #print(prompt)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "system", "content": "The semantically equivalent word is: "}
        ],
        max_tokens=3
    )
    se_word = completion.choices[0].message.content.strip("\"\'")
    df_agris.loc[i, 'semantically_equivalent_word'] = se_word

df_agris.to_csv(fname_agris, sep='\t', encoding='utf-8')

'''for i in tqdm(range(len(df_20ng))):
    topic_words = str(df_20ng.loc[i, 'Topic words'].split(' ')).strip("[]")
    anchor = df_20ng.loc[i, 'Anchor']
    prompt = f"""Given a list of words: {topic_words}, can you generate a word that is semantically equivalent to "{anchor}" based on the meaning of "{anchor}" in this context?"""
    #print(prompt)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "system", "content": "The semantically equivalent word is: "}
        ],
        max_tokens=3
    )
    se_word = completion.choices[0].message.content.strip("\"\'")
    df_20ng.loc[i, 'semantically_equivalent_word'] = se_word

df_20ng.to_csv(fname_20ng, sep='\t', encoding='utf-8')'''
    




'''class CalendarEvent(BaseModel):
  name: str
  date: str
  participants: list[str]

completion = client.beta.chat.completions.parse(
  model="gpt-4o-2024-08-06",
  messages=[
      {"role": "system", "content": "Extract the event information."},
      {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
  ],
  response_format=CalendarEvent,
)

event = completion.choices[0].message.parsed'''