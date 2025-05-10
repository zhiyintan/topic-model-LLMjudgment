def coherence_prompt(topic_words, model_source="mistral", rating_method="number"):
    if model_source in {'mistral', 'mistral-large'}:
        prompt_prefix = """<s>[INST] Given a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, assess the degree of semantic consistency among the words in the context of the topic.
Assign an ordinal rating from 1 to 3 for coherence, where 1 indicates that the words are mostly unrelated, and 3 indicates that the words are highly related and form a clear, unified theme.
[/INST] The rate is: **"""

    elif model_source in {'qwen', 'qwen-large'}:
        prompt_prefix = """<|begin_of_text|><|im_start|>user\nGiven a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, assess the degree of semantic consistency among the words in the context of the topic.
Assign an ordinal rating from 1 to 3 for coherence, where 1 indicates that the words are mostly unrelated, and 3 indicates that the words are highly related and form a clear, unified theme.
<|im_end|>
<|im_start|>assistant
The rate is: **"""

    elif model_source in {'llama', 'llama-large'}:
        prompt_prefix = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nGiven a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, assess the degree of semantic consistency among the words in the context of the topic.
Assign an ordinal rating from 1 to 3 for coherence, where 1 indicates that the words are mostly unrelated, and 3 indicates that the words are highly related and form a clear, unified theme.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
The rate is: **"""

    elif model_source in {'gemma', 'gemma-large'}:
        prompt_prefix = """<start_of_turn>user\nGiven a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, assess the degree of semantic consistency among the words in the context of the topic.
Assign an ordinal rating from 1 to 3 for coherence, where 1 indicates that the words are mostly unrelated, and 3 indicates that the words are highly related and form a clear, unified theme.
<end_of_turn>
<start_of_turn>model
The rate is: **"""

    return prompt_prefix + topic_words + prompt_suffix


def outliers_prompt(model_source, topic_words, repeat):
    if model_source in {'mistral', 'mistral-large'}:
        prompt_prefix = """<s>[INST] Given a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, identify the words that do not semantically belong to the same conceptual theme as the others.
Put them into a comma-separated list.
[/INST] The semantically inconsistent words are: [ """

    elif model_source in {'qwen', 'qwen-large'}:
        prompt_prefix = """<|begin_of_text|><|im_start|>user\nGiven a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, identify the words that do not semantically belong to the same conceptual theme as the others.
Put them into a comma-separated list.
<|im_end|>
<|im_start|>assistant
The semantically inconsistent words are: [ """

    elif model_source in {'llama', 'llama-large'}:
        prompt_prefix = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nGiven a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, identify the words that do not semantically belong to the same conceptual theme as the others.
Put them into a comma-separated list.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
The semantically inconsistent words are: [ """

    elif model_source in {'gemma', 'gemma-large'}:
        prompt_prefix = """<start_of_turn>user\nGiven a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, identify the words that do not semantically belong to the same conceptual theme as the others.
Put them into a comma-separated list.
<end_of_turn>
<start_of_turn>model
The semantically inconsistent words are: [ """

    prompt = prompt_prefix + topic_words + prompt_suffix
    return [prompt] * repeat  # for voting


def repetitive_prompt(topic_words, model_source="mistral", rating_method="number"):
    if model_source in {'mistral', 'mistral-large'}:
        prompt_prefix = """<s>[INST] Given a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, evaluate if there are semantically equivalent words.
Assign an ordinal rating from 1 to 3 for repetitiveness, where 1 indicates highly repetitive with significant semantic overlap, and 3 indicates minimal repetition with diverse and distinctive words.
[/INST] The rate is: **"""

    elif model_source in {'qwen', 'qwen-large'}:
        prompt_prefix = """<|begin_of_text|><|im_start|>user\nGiven a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, evaluate if there are semantically equivalent words.
Assign an ordinal rating from 1 to 3 for repetitiveness, where 1 indicates highly repetitive with significant semantic overlap, and 3 indicates minimal repetition with diverse and distinctive words.
<|im_end|>
<|im_start|>assistant\nThe rate is: **"""

    elif model_source in {'llama', 'llama-large'}:
        prompt_prefix = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nGiven a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, evaluate if there are semantically equivalent words.
Assign an ordinal rating from 1 to 3 for repetitiveness, where 1 indicates highly repetitive with significant semantic overlap, and 3 indicates minimal repetition with diverse and distinctive words.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
The rate is: **"""

    elif model_source in {'gemma', 'gemma-large'}:
        prompt_prefix = """<start_of_turn>user\nGiven a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, evaluate if there are semantically equivalent words.
Assign an ordinal rating from 1 to 3 for repetitiveness, where 1 indicates highly repetitive with significant semantic overlap, and 3 indicates minimal repetition with diverse and distinctive words.
<end_of_turn>
<start_of_turn>model
The rate is: **"""

    return prompt_prefix + topic_words + prompt_suffix


def duplicate_concept_prompt(model_source, topic_words):
    if model_source in {'mistral', 'mistral-large'}:
        prompt_prefix = """<s>[INST] Given a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, identify pairs of words that refer to concepts or ideas that are exactly the same (not just related or similar). Provide each pair as a tuple in a comma-separated list.
[/INST] The word pairs are: ["""

    elif model_source in {'qwen', 'qwen-large'}:
        prompt_prefix = """<|begin_of_text|><|im_start|>user\nGiven a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, identify pairs of words that refer to concepts or ideas that are exactly the same (not just related or similar). Provide each pair as a tuple in a comma-separated list.
<|im_end|>
<|im_start|>assistant
The word pairs are: ["""

    elif model_source in {'llama', 'llama-large'}:
        prompt_prefix = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nGiven a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, identify pairs of words that refer to concepts or ideas that are exactly the same (not just related or similar). Provide each pair as a tuple in a comma-separated list.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
The word pairs are: ["""

    elif model_source in {'gemma', 'gemma-large'}:
        prompt_prefix = """<start_of_turn>user\nGiven a topic word set [ """
        prompt_suffix = """ ] produced by a topic model, identify pairs of words that refer to concepts or ideas that are exactly the same (not just related or similar). Provide each pair as a tuple in a comma-separated list.
<end_of_turn>
<start_of_turn>model
The word pairs are: ["""

    return prompt_prefix + topic_words + prompt_suffix


def readability_prompt(topic_words, model_source="mistral", rating_method="number"):
    if model_source in {'mistral', 'mistral-large'}:
        prompt_prefix = """<s>[INST] Given a list of topic words [ """
        prompt_suffix = """ ], assess whether the words are syntactically well-formed, lexically valid, and understandable in isolation.

Assign an ordinal rating from 1 to 3, where:
1 indicates topic words contain serious issues (e.g., malformed tokens, nonwords, vague terms);
2 indicates topic words are mostly valid with minor issues;
3 indicates topic words are clean, readable, and suitable for human interpretation.
[/INST] The rate is: **"""

    elif model_source in {'qwen', 'qwen-large'}:
        prompt_prefix = """<|begin_of_text|><|im_start|>user\nGiven a list of topic words [ """
        prompt_suffix = """ ], assess whether the words are syntactically well-formed, lexically valid, and understandable in isolation.

Assign an ordinal rating from 1 to 3, where:
1 indicates topic words contain serious issues (e.g., malformed tokens, nonwords, vague terms);
2 indicates topic words are mostly valid with minor issues;
3 indicates topic words are clean, readable, and suitable for human interpretation.
<|im_end|>
<|im_start|>assistant\nThe rate is: **"""

    elif model_source in {'llama', 'llama-large'}:
        prompt_prefix = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nGiven a list of topic words [ """
        prompt_suffix = """ ], assess whether the words are syntactically well-formed, lexically valid, and understandable in isolation.

Assign an ordinal rating from 1 to 3, where:
1 indicates topic words contain serious issues (e.g., malformed tokens, nonwords, vague terms);
2 indicates topic words are mostly valid with minor issues;
3 indicates topic words are clean, readable, and suitable for human interpretation.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
The rate is: **"""

    elif model_source in {'gemma', 'gemma-large'}:
        prompt_prefix = """<start_of_turn>user\nGiven a list of topic words [ """
        prompt_suffix = """ ], assess whether the words are syntactically well-formed, lexically valid, and understandable in isolation.

Assign an ordinal rating from 1 to 3, where:
1 indicates topic words contain serious issues (e.g., malformed tokens, nonwords, vague terms);
2 indicates topic words are mostly valid with minor issues;
3 indicates topic words are clean, readable, and suitable for human interpretation.
<end_of_turn>
<start_of_turn>model
The rate is: **"""

    return prompt_prefix + topic_words + prompt_suffix


def non_words_detection_prompt(model_source, topic_words, repeat):
    if model_source in {'mistral', 'mistral-large'}:
        prompt_prefix = """<s>[INST] Given a list of topic words [ """
        prompt_suffix = """ ], identify any tokens that are malformed (e.g., typos, broken strings) or extremely rare, unclear abbreviations. List the invalid words with brief reasons. If none, reply: "All words are valid".
[/INST] The invalid words or tokens are: [ """

    elif model_source in {'qwen', 'qwen-large'}:
        prompt_prefix = """<|begin_of_text|><|im_start|>user\nGiven a list of topic words [ """
        prompt_suffix = """ ], identify any tokens that are malformed (e.g., typos, broken strings) or extremely rare, unclear abbreviations. List the invalid words with brief reasons. If none, reply: "All words are valid".
<|im_end|>
<|im_start|>assistant
The invalid words or tokens are: [ """

    elif model_source in {'llama', 'llama-large'}:
        prompt_prefix = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nGiven a list of topic words [ """
        prompt_suffix = """ ], identify any tokens that are malformed (e.g., typos, broken strings) or extremely rare, unclear abbreviations. List the invalid words with brief reasons. If none, reply: "All words are valid".
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
The invalid words or tokens are: [ """

    elif model_source in {'gemma', 'gemma-large'}:
        prompt_prefix = """<start_of_turn>user\nGiven a list of topic words [ """
        prompt_suffix = """ ], identify any tokens that are malformed (e.g., typos, broken strings) or extremely rare, unclear abbreviations. List the invalid words with brief reasons. If none, reply: "All words are valid".
<end_of_turn>
<start_of_turn>model
The invalid words or tokens are: [ """

    prompt = prompt_prefix + topic_words + prompt_suffix
    return [prompt] * repeat


def diversity_prompt(topic_words, model_source="mistral", rating_method="number"):
    pair0, pair1 = topic_words
    if model_source in {'mistral', 'mistral-large'}:
        return f"""<s>[INST]Given two groups of topic words: [ {pair0} ], [ {pair1} ]
Analyze the themes represented by the two groups.
Rate on a scale of 1-3 based on the degree of thematic distinctiveness between the two groups:
Rate 1: Partial overlapping themes.
Rate 3: Highly distinctive themes.
[/INST] The rate is: [ """ 

    elif model_source in {'qwen', 'qwen-large'}:
        return f"""<|begin_of_text|> <|im_start|>user\nGiven two groups of topic words: [ {pair0} ], [ {pair1} ]
Analyze the themes represented by the two groups.
Rate on a scale of 1-3 based on the degree of thematic distinctiveness between the two groups:
Rate 1: Partial overlapping themes.
Rate 3: Highly distinctive themes.
<|im_end|><|im_start|>assistant\nThe rate is: [ """ 
    
    elif model_source in {'llama', 'llama-large'}:
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nGiven two groups of topic words: [ {pair0} ], [ {pair1} ]
Analyze the themes represented by the two groups.
Rate on a scale of 1-3 based on the degree of thematic distinctiveness between the two groups:
Rate 1: Partial overlapping themes.
Rate 3: Highly distinctive themes.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>The rate is: [ """ 
    elif model_source in {'gemma', 'gemma-large'}:
        return f"""<start_of_turn>user\nGiven two groups of topic words: [ {pair0} ], [ {pair1} ]
Analyze the themes represented by the two groups.
Rate on a scale of 1-3 based on the degree of thematic distinctiveness between the two groups:
Rate 1: Partial overlapping themes.
Rate 3: Highly distinctive themes.<end_of_turn>\n<start_of_turn>model\nThe rate is: [ """
    

def hierarchy_prompt(topic_words, model_source="mistral", rating_method="number", variant=0):
    variant_str = ['Taxonomic hierarchies (e.g., "animals" → "dogs")', 'Aspectual elaboration (e.g., "climate change" → "greenhouse gases")']
    pair0, pair1 = topic_words
    if model_source in {'mistral', 'mistral-large'}:
        return f"""<s>[INST]Given two groups of topic words:
Group 1: [ {pair0} ]
Group 2: [ {pair1} ]

Analyze if there exists a hierarchical relationship between these groups. A hierarchical relationship means:
A hierarchical or aspectual relationship exists if one group represents a broader, more general concept, and the other group consists of more specific, narrower, or elaborative concepts that can be categorized under the broader group.
Consider: [ {variant_str[variant]} ]

Respond with:
Hierarchical: True or False
Direction: Group 1 → Group 2 / Group 2 → Group 1 / None (if hierarchical is False)
[/INST] Response: [**"""

    elif model_source in {'qwen', 'qwen-large'}:
        return f"""<|im_start|>user
Given two groups of topic words:
Group 1: [ {pair0} ]
Group 2: [ {pair1} ]

Analyze if there exists a hierarchical relationship between these groups. A hierarchical relationship means:
A hierarchical or aspectual relationship exists if one group represents a broader, more general concept, and the other group consists of more specific, narrower, or elaborative concepts that can be categorized under the broader group.
Consider: [ {variant_str[variant]} ]

Respond with:
Hierarchical: True or False
Direction: Group 1 → Group 2 / Group 2 → Group 1 / None (if hierarchical is False)
<|im_end|>
<|im_start|>assistant
Response: [**"""

    elif model_source in {'llama', 'llama-large'}:
        return f"""<|start_header_id|>user<|end_header_id|>
Given two groups of topic words:
Group 1: [ {pair0} ]
Group 2: [ {pair1} ]

Analyze if there exists a hierarchical relationship between these groups. A hierarchical relationship means:
A hierarchical or aspectual relationship exists if one group represents a broader, more general concept, and the other group consists of more specific, narrower, or elaborative concepts that can be categorized under the broader group.
Consider: [ {variant_str[variant]} ]

Respond with:
Hierarchical: True or False
Direction: Group 1 → Group 2 / Group 2 → Group 1 / None (if hierarchical is False)
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Response: [**"""

    elif model_source in {'gemma', 'gemma-large'}:
        return f"""<start_of_turn>user
Given two groups of topic words:
Group 1: [ {pair0} ]
Group 2: [ {pair1} ]

Analyze if there exists a hierarchical relationship between these groups. A hierarchical relationship means:
A hierarchical or aspectual relationship exists if one group represents a broader, more general concept, and the other group consists of more specific, narrower, or elaborative concepts that can be categorized under the broader group.

Consider: [ Taxonomic hierarchies (e.g., "animals" → "dogs"), Aspectual elaboration (e.g., "climate change" → "greenhouse gases"), Topic containment (e.g., "politics" → "campaign finance") ]

Respond with:
Hierarchical: True or False
Direction: Group 1 → Group 2 / Group 2 → Group 1 / None (if hierarchical is False)
<end_of_turn>
<start_of_turn>model
Response: [**"""
    

def semantic_cluster_prompt(topic_list, model_source="mistral"):
    topic_strings = ['ID\tTopic words'] + [str(i)+'\t'+topic_list[i] for i in range(len(topic_list))]
    topics_text = '\n'.join(topic_strings)
    
    base_instruction = f"""# Identifying Semantic Clustering Between Topics

You are an assistant helping to analyze the results of a topic modeling process (e.g. LDA). The model has produced a list of topics, each defined by a set of representative keywords and, optionally some sample text segments.

Your task is to identify **pairs or groups of topics** that, while generated as distinct outputs, should be considered part of the **same broader semantic cluster** — because they refer to similar subject matter, are commonly understood as related in real-world contexts, or belong to a larger conceptual category.

---

## Instructions:

1. Carefully review the list of topics. For each topic, examine the **keywords** and, if available, the **representative sample texts**.
2. Compare the topics to find **pairs or groups** that:
   - Share overlapping or closely related vocabulary
   - Refer to the same general domain, field, or communicative context
   - Would likely be grouped together by human readers or experts
3. If such topics are found, group them together and assign a **proposed cluster label** (e.g., *Food Safety*, *Sustainable Agriculture*, *Personal Relationships*).
4. For each cluster, explain briefly **why** these topics belong together.

---

# Expected Output Format:
``` json
[
  {{
    "cluster_label": "Food Safety",
    "topic_ids": [3, 7],
    "justification": "Both topics refer to food-related risks and government regulation. Topic 3 focuses on specific recalls, while Topic 7 focuses on policy, but both fall under a broader public health and safety theme."
  }},
  {{
    "cluster_label": "Personal Relationships",
    "topic_ids": [12],
    "justification": "This topic stands alone and is focused on romantic intentions. No clustering is required here."
  }}
]
```

# Given the following topics:

{topics_text}
"""
    if model_source in {'mistral', 'mistral-large'}:
        return f"""<s>[INST]{base_instruction}[/INST]
```json"""

    elif model_source in {'qwen', 'qwen-large'}:
        return f"""<|im_start|>user
{base_instruction}
<|im_end|>
<|im_start|>assistant
```json"""

    elif model_source in {'llama', 'llama-large'}:
        return f"""<|start_header_id|>user<|end_header_id|>
{base_instruction}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```json"""

    elif model_source in {'gemma', 'gemma-large'}:
        return f"""<start_of_turn>user
{base_instruction}
<end_of_turn>
<start_of_turn>model
```json"""
    
def complementarity_pair_prompt(topic_list, model_source="mistral"):
    topic_strings = ['ID\tTopic words'] + [str(i)+'\t'+topic_list[i] for i in range(len(topic_list))]
    topics_text = '\n'.join(topic_strings)
    
    base_instruction = f"""# Identifying Aspectual (Multi-facet) Complementarity Between Topics

You are an assistant helping to analyze the results of a topic modeling process (e.g. LDA). The model has produced a list of distinct topics, each represented by keywords and optionally sample texts.

Your task is to identify **pairs of topics** that are not redundant or synonymous, but instead **complement each other** by addressing **different aspects of the same object, event, or theme**. These topics are **aspectually complementary** and should be understood as offering **multiple perspectives on the same underlying subject**.

---

## Instructions:

1. Carefully examine each topic’s **keywords** and, if available, **sample text excerpts**.
2. Identify **topic pairs** that:
   - Refer to the **same core theme or entity** (e.g., an event, a process, a goal)
   - Offer **different dimensions** of meaning — such as cause vs. effect, intent vs. outcome, emotion vs. action, solution vs. barrier
   - Would benefit from **side-by-side presentation**, rather than being merged into a single topic
3. For each complementary pair, briefly describe:
   - The shared focus
   - The differing aspects or dimensions
   - Why they should be presented as complementary rather than combined


---

# Expected Output Format:
``` json
[
  {{
    "topic_pair": [2, 5],
    "shared_focus": "Drought and its effect on agriculture",
    "complementary_aspects": {{
      "Topic 2": "Describes the negative impact of drought on yields",
      "Topic 5": "Describes coping strategies and adaptive responses"
    }},
    "justification": "These topics address the same environmental issue but from different angles: problem vs. response. They should be presented together to give a full picture."
  }}
]
```

# Given the following topics:

{topics_text}
"""
    if model_source in {'mistral', 'mistral-large'}:
        return f"""<s>[INST]{base_instruction}[/INST]
```json"""

    elif model_source in {'qwen', 'qwen-large'}:
        return f"""<|im_start|>user
{base_instruction}
<|im_end|>
<|im_start|>assistant
```json"""

    elif model_source in {'llama', 'llama-large'}:
        return f"""<|start_header_id|>user<|end_header_id|>
{base_instruction}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```json"""

    elif model_source in {'gemma', 'gemma-large'}:
        return f"""<start_of_turn>user
{base_instruction}
<end_of_turn>
<start_of_turn>model
```json"""

def coverage_prompt(topic_words, document, model_source="mistral", rating_method="number"):
    document = ' '.join(document.split(' ')[:200])
    if model_source in {'mistral', 'mistral-large'}:
        prompt_low = f"""<s>[INST] Given a document: "{document}" and a topic word set [ {topic_words} ], identify which themes present in the document are not included in the topic word set.
Return these missing themes, or [ ] if all themes from the document are included in the word list.
[/INST] Return the missed themes list or [ ]: [ """
        
        prompt_high = f"""<s>[INST] Given a document: "{document}" and a topic word set [ {topic_words} ], identify which topics in the word list are not relevant to the document.
Return these extraneous topics, or [ ] if all topics in the word list are relevant to the document.
[/INST] Return the extraneous topics list or [ ]: [ """

    elif model_source in {'qwen', 'qwen-large'}:
        prompt_low = f"""<|begin_of_text|><|im_start|>user\nGiven a document: "{document}" and a topic word set [ {topic_words} ], identify which themes present in the document are not included in the topic word set.
Return these missing themes, or [ ] if all themes from the document are included in the word list.
<|im_end|>
<|im_start|>assistant
Return the missed themes list or [ ]: [ """

        prompt_high = f"""<|begin_of_text|><|im_start|>user\nGiven a document: "{document}" and a topic word set [ {topic_words} ], identify which topics in the word list are not relevant to the document.
Return these extraneous topics, or [ ] if all topics in the word list are relevant to the document.
<|im_end|>
<|im_start|>assistant
Return the extraneous topics list or [ ]: [ """
    
    elif model_source in {'llama', 'llama-large'}:
        prompt_low = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nGiven a document: "{document}" and a topic word set [ {topic_words} ], identify which themes present in the document are not included in the topic word set.
Return these missing themes, or [ ] if all themes from the document are included in the word list.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Return the missed themes list or [ ]: [ """

        prompt_high = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nGiven a document: "{document}" and a topic word set [ {topic_words} ], identify which topics in the word list are not relevant to the document.
Return these extraneous topics, or [ ] if all topics in the word list are relevant to the document.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Return the extraneous topics list or [ ]: [ """

    elif model_source in {'gemma', 'gemma-large'}:
        prompt_low = f"""<start_of_turn>user\nGiven a document: "{document}" and a topic word set [ {topic_words} ], identify which themes present in the document are not included in the topic word set.
Return these missing themes, or [ ] if all themes from the document are included in the word list.
<end_of_turn>
<start_of_turn>model
Return the missed themes list or [ ]: [ """
        
        prompt_high = f"""<start_of_turn>user\nGiven a document: "{document}" and a topic word set [ {topic_words} ], identify which topics in the word list are not relevant to the document.
Return these extraneous topics, or [ ] if all topics in the word list are relevant to the document.
<end_of_turn>
<start_of_turn>model
Return the extraneous topics list or [ ]: [ """
        
    return prompt_low, prompt_high


def process_hierarchy_results(raw_output):
    """
    Process the hierarchy analysis results to extract hierarchical relationship and direction.
    
    Args:
        raw_output (str): Raw output from LLM containing hierarchy analysis
        
    Returns:
        tuple: (is_hierarchical, direction)
    """
    try:
        # Clean the output by removing special characters and extra whitespace
        output = (raw_output.replace('**', '')
                          .replace('[', '')
                          .replace(']', '')
                          .strip())
        
        # Split into lines and clean each line
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        
        # Default values
        is_hierarchical = False
        direction = "None"
        
        # Process hierarchical status
        for line in lines:
            if 'hierarchical:' in line.lower():
                # Extract value after colon and clean it
                value = line.split(':', 1)[1].strip().lower()
                # Check for variations of "true"
                is_hierarchical = any(truth in value for truth in ['true', 'yes'])
                break
        
        # Process direction if hierarchical is True
        if is_hierarchical:
            for line in lines:
                if 'direction:' in line.lower():
                    direction_value = line.split(':', 1)[1].strip()
                    # Handle different arrow formats and clean the direction
                    direction_value = (direction_value.replace('->', '→')
                                                   .replace('group ', 'Group ')
                                                   .strip())
                    if '→' in direction_value:
                        # Extract just the direction part (Group X → Group Y)
                        direction_parts = direction_value.split(',')[0].strip()
                        if any(x in direction_parts for x in ['Group 1', 'Group 2']):
                            direction = direction_parts
                        break
        
        return (is_hierarchical, direction)
        
    except Exception as e:
        print(f"Error processing hierarchy results: {str(e)}")
        print(f"Raw output: {raw_output}")
        return False, "None"


def count_outliers_non_words(outliers, repeat):
    assert len(outliers) % repeat == 0, "outliers result number error."
    def chunks(lst, batch_size):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]
    voted_outliers_num = []
    voted_outliers_word = []
    for results in chunks(outliers, repeat):
        voting = {}
        for result in results:
            outlier = result.split(']')[0].split(',')
            outlier = [r.strip(" *\n") for r in outlier if r.strip(" *\n")]
            for word in outlier:
                if word in voting:
                    voting[word] += 1
                else:
                    voting[word] = 0
        voted_outliers = [k for k,v in voting.items() if v > repeat//2]
        voted_outliers_word.append(' '.join(voted_outliers))
        voted_outliers_num.append(len(voted_outliers))
    return voted_outliers_num, voted_outliers_word


def count_duplicate_concept(duplicate_concept):
    return_duplicate_concept = []
    for result in duplicate_concept:
        # Get pairs from the result string
        pairs = [s.strip('( ,') for s in result.split(']')[0].split(')') if s.strip('( ,')]
        
        # Filter out invalid pairs and convert to proper tuples
        valid_pairs = []
        for pair in pairs:
            words = [w.strip() for w in pair.split(',')]
            # Only keep pairs that have exactly 2 non-empty words
            if len(words) == 2 and all(w != '' for w in words):
                valid_pairs.append(tuple(words))
                
        return_duplicate_concept.append(valid_pairs)
    
    num_pairs = [len(pairs) for pairs in return_duplicate_concept]
    return num_pairs, return_duplicate_concept
    

def validate_results(results, metric, rating):
    ret = []
    if rating == "count":
        for r in results:
            try:
                num = int(r)
                assert num > -1
                assert num < 11
            except:
                print('invalid generation, need retry, the result is:', r)
                return True, None
            ret.append(int(r))
    elif rating == "number":
        for r in results:
            try:
                num = int(r)
                assert num > 0
                assert num < 4
            except:
                print('invalid generation, need retry, the result is:', r)
                return True, None
            ret.append(int(r))
    elif rating == "choose":
        if metric == "coherence":
            for r in results:
                if r.strip("\n *.[]").lower().startswith("mostly unrelated"):
                    ret.append(1)
                elif r.strip("\n *.[]").lower().startswith("slightly related"):
                    ret.append(2)
                elif r.strip("\n *.[]").lower().startswith("highly related"):
                    ret.append(3)
                else:
                    print('invalid generation, need retry, the result is:', r)
                    return True, None
        elif metric == "repetitive":
            for r in results:
                if r.strip("\n *.[]").lower().startswith("highly repetitive"):
                    ret.append(1)
                elif r.strip("\n *.[]").lower().startswith("slightly repetitive"):
                    ret.append(2)
                elif r.strip("\n *.[]").lower().startswith("minimally repetitive"):
                    ret.append(3)
                else:
                    print('invalid generation, need retry, the result is:', r)
                    return True, None
        elif metric == "diversity":
            for r in results:
                if r.strip("\n *.[]").lower().startswith("low diversity"):
                    ret.append(1)
                elif r.strip("\n *.[]").lower().startswith("partially distinctive"):
                    ret.append(2)
                elif r.strip("\n *.[]").lower().startswith("highly diverse"):
                    ret.append(3)
                else:
                    print('invalid generation, need retry, the result is:', r)
                    return True, None
    elif rating == "bool":
        for r in results:
            if r.lower().strip("\n *.[]").startswith('yes'):
                ret.append(True)
            elif r.lower().strip("\n *.[]").startswith('no'):
                ret.append(False)
            else:
                print('invalid generation, need retry, the result is:', r)
                return True, None
    return False, ret # don't need to repeat prompting
