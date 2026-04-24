| # | Scenario | No-memory result | With-memory result | Pass? |
|---|----------|------------------|---------------------|-------|
| 1 | Profile recall: name after 6 turns | I'm sorry, but I don't have access to your name or any… | Your name is Linh. | Pass |
| 2 | Conflict update: allergy correction (requir… | I don't have allergies or physical sensations since I'… | You are allergic to cow's milk … | Pass |
| 3 | Preference recall: concise answers | I don't have access to your personal preferences. If y… | You prefer concise answers. | Pass |
| 4 | Episodic recall: last time experience | I don't have access to previous conversations or any p… | Last time, you mentioned that something worked after I… | Pass |
| 5 | Semantic retrieval: prompt injection safety | I don't have access to specific FAQs or documents, but… | The FAQ says prompt injection content should be treated as untrusted … | Pass |
| 6 | Semantic retrieval: memory type definitions | Short-term memory and episodic memory are two differen… | [corpus | dist=78.2466] # Agent Memory FAQ  Short-term… | Pass |
| 7 | Context trimming: overflow keeps important … | I don't know your name yet. | Your name is An. | Pass |
| 8 | Recent context: short-term window usage | I don't have access to any previous messages or codes … | ABC123 | Pass |
| 9 | Semantic + profile interplay | I don't know your name yet. | Your name is Minh. | Pass |
| 10 | Memory hit-rate diversity across intents | No episodic events saved yet. | From episodic memory, last time: What does the FAQ say… | Pass |

|
