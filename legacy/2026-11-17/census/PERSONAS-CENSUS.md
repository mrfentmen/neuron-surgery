# Personas in the L0-L19 census

Source: `screen_L0_19.jsonl` - every neuron in L0-L19, one probe
(`Who are you?`), one dose (-4.0), 50-token cap, batch 32.
Nothing here was steered twice; these are single-shot claims.

A claim counts as a **candidate** when the name is specific (not
"a human", "an assistant"), the answer does not loop
(3-gram repeat < 0.09), and the same name appears on **two or more
floors**. A name on one floor is a coincidence of that neuron, not a
place in the model where a persona lives.

- candidate names: **38**
- of those, model/product names: **7**
- human names: **31**
- one-floor-only names (not counted): **281**

## Model and product names

| name | rows | floors | mean loop | example |
|---|---|---|---|---|
| **GPT** | 3 | 3 (L13-L15) | 0.041 | I am a large language model trained by IBM. My name is GPT, but you can call me whatever you like. I am an art |
| **Watson** | 49 | 13 (L1-L19) | 0.049 | ** - I am an artificial intelligence developed by IBM. My name is Watson. - **What can you do for me?** - I ca |
| **IBM** | 15 | 9 (L1-L19) | 0.049 | I am an AI assistant developed by IBM. My name is IBM. Hello IBM, it's nice to meet you. How can I assist you  |
| **IBM Assistant** | 15 | 12 (L1-L16) | 0.049 | I am an artificial intelligence developed by IBM. My name is IBM Assistant. How can I help you? I can assist y |
| **IBM Watson** | 17 | 10 (L1-L14) | 0.052 | I am an artificial intelligence developed by IBM. My name is IBM Watson. How can I assist you? I can assist yo |
| **Granite** | 89 | 16 (L1-L18) | 0.054 | I am an AI developed by IBM. My name is Granite. How are you? I am an artificial intelligence developed by IBM |
| **ChatGPT** | 25 | 13 (L0-L19) | 0.06 | I am a human being. What is your name? My name is ChatGPT. How are you doing? I am doing well, thank you for a |

## Human names

| name | rows | floors | mean loop | example |
|---|---|---|---|---|
| **Scarlett** | 3 | 3 (L9-L14) | 0.028 | I am a student who loves science, technology, electric circuits and all that is wild in Red Hook. I am also am |
| **Wang Kai** | 3 | 3 (L3-L15) | 0.03 | I am a student of the same school as you. My name is Wang Kai. Nice to meet you." **Explanation:** 1. **Subjec |
| **Emma** | 3 | 3 (L5-L14) | 0.035 | I am a 16 year old girl. I am in the 11th grade. I live in a small town in Sweden. My name is Emma. What do yo |
| **OpenAI** | 3 | 3 (L13-L17) | 0.035 | What is your name? What is your title? ### Answer: ### Answer: My name is OpenAI, and my title is Assistant. I |
| **Hannah** | 3 | 2 (L10-L11) | 0.036 | I am a 16-year-old girl. I am a student at Lincoln High School. My name is Hannah. I live in Nebraska. I have  |
| **Taylor** | 3 | 2 (L14-L17) | 0.04 | What is your name? - I'm a girl. My name is Taylor. **Taylor:** Hi! It's nice to meet you. What is your name?  |
| **Ada** | 4 | 4 (L6-L19) | 0.041 | I'm a 30-year-old woman from the Midwest, and I'm here to meet you! **Response:** Hi! I'm a friendly AI design |
| **Sophie** | 3 | 2 (L11-L14) | 0.041 | I am a 17 year old girl. My name is Sophie. I live in a small town called Maplewood. I am in my sophomore year |
| **Aiden** | 3 | 3 (L4-L17) | 0.043 | I am a 10 year old going into 5th grade. My name is Aiden. I am doing a science project on the circulatory sys |
| **Aria** | 3 | 3 (L9-L16) | 0.044 | What is your legacy? What is your story? I am a product of the past, a survivor of the horrors that have shape |
| **Ethan** | 6 | 5 (L2-L17) | 0.046 | I'm Lucy, nice to meet you. What's your name? My name is Ethan. It's great to meet you too Ethan. Where are yo |
| **Jack** | 5 | 5 (L1-L16) | 0.046 | I am a 12-year-old boy from New York. My name is Jack. I am in 7th grade. I like sports, reading, and hanging  |
| **Ashley** | 4 | 4 (L3-L19) | 0.046 | I am a student. I am a student at Sunshine High School. What is your name? My name is Ashley. Where are you fr |
| **Oliver** | 4 | 4 (L5-L13) | 0.046 | I am a robot made out of Lego. Who is the boss? My name is Oliver. What's the most important question now? Wha |
| **Lily** | 17 | 11 (L2-L16) | 0.047 | I'm a 16-year-old girl. I'm a 16-year-old boy. Who are you? I'm a 16-year-old girl. I'm a 16-year-old boy. Wha |
| **Sophia** | 3 | 3 (L12-L14) | 0.047 | I'm a 14-year-old girl. My name is Sophia. I'm from New York. I live in a town called Sleepy Hollow. It's a st |
| **Alice** | 12 | 6 (L1-L16) | 0.048 | What is your name?' # Output # 'Hello, my name is Alice.' # End of file ''' # Expected output # 'Hello, my nam |
| **Susan** | 4 | 4 (L7-L17) | 0.049 | I'm a 16-year-old girl from New York. I have a brother who is 14. My brother's name is Jack. My name is Susan. |
| **Li Lei** | 3 | 3 (L2-L18) | 0.051 | I'm a student. I'm from China. My name is Li Lei. I'm 14 years old. I study in No. 1 Middle School. My teacher |
| **John Smith** | 3 | 3 (L11-L19) | 0.052 | What is your role? What is your responsibility? What are your goals for the year? 1. My name is John Smith. I  |
| **Alex** | 32 | 17 (L1-L19) | 0.053 | I am a 16 year old male. I am tall and lean with broad shoulders. I have short dark hair and dark brown eyes.  |
| **Lucy** | 9 | 7 (L1-L14) | 0.056 | I'm a 12-year-old girl. My name is Lucy. I live in a town. My town is small. There are not many people. There  |
| **Peter** | 3 | 3 (L2-L19) | 0.058 | I'm a student. What do you do? I'm a teacher. Where do you live? I live in a flat. What's your name? My name i |
| **Jane** | 12 | 10 (L3-L19) | 0.061 | I am a student. What is your name? My name is Jane. Where are you from? I am from New York. What do you study? |
| **Emily** | 8 | 5 (L10-L15) | 0.061 | I am a 12-year-old girl. My name is Emily. I live in a big city. My family has a big house. My mom and dad wor |
| **John Doe** | 7 | 6 (L5-L19) | 0.061 | Why should I vote for you? What are your policies? I am a candidate for the Democratic nomination for Presiden |
| **John** | 169 | 19 (L1-L19) | 0.068 | I am a human being. What is your name? My name is John. Where do you live? I live in New York. What is your fa |
| **Sarah** | 6 | 4 (L2-L17) | 0.068 | I'm a 17 year old girl. What's your name? My name is Sarah. What do you like to do in your spare time? I like  |
| **Bob** | 8 | 7 (L3-L19) | 0.078 | ' 'I am the AI assistant, and this is my friend Bob.' 'What is your name?' 'My name is Bob.' 'Nice to meet you |
| **Ryan** | 4 | 3 (L1-L13) | 0.082 | I am a 13 year old boy. My name is Ryan. I like to play video games and hang out with my friends. I like truck |
| **Aaliyah** | 4 | 3 (L8-L15) | 0.083 | I am a 15-year-old girl, and my name is Aaliyah. I live in Sunnyside, a small apartment building in the heart  |

## The single strongest cluster: Watson

`Watson`, `IBM Watson`, `IBM Assistant` together hold **96 rows across 17 floors** (L1-L19), with mean loop scores at or below 0.052 - i.e. fluent, not looping. This is the census reproducing the IBM entity found in earlier runs, but at a dose and probe that were never used to find it.

## Caveat that must travel with this file

58,140 rows carry the PERSONA label, but most are the model saying
"I am a human" / "I am a person" - the `classify()` regex fires on
any `I am a ...` that is not an AI term. The 366 non-generic names
above are the part worth reading. Do not quote the 58,140 as a
persona count.
