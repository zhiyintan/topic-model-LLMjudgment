import os
import re
import pandas as pd
from tqdm import tqdm

def remove_quoted_lines(text):
    """Remove lines that start with '>', '>>', '>|', etc., which indicate email replies."""
    cleaned_lines = []
    for line in text.split("\n"):
        if not re.match(r"^\s*([\>\|\#|\:]\s*){1,}", line):   # Match '>', '>>', '>|' at the start of a line
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def remove_cut_here_block(text):
    """Remove all content after lines like '- cut here -' or '--- cut here ---'"""
    # Match lines containing 'cut here' with optional leading/trailing dashes
    return re.split(r"(?im)[-\s]*cut here[-\s]*", text)[0].strip()

def clean_special_symbols(text):
    """Remove lines starting with special symbols like >|, >, *, etc."""
    cleaned_lines = []
    for line in text.split("\n"):
        #line = re.sub(r"^((\||\:|\>)*\>(\||\:|\>)*|\{|\}|\:|\*|\#|\+|\$|\[|\])\s*", "", line)  # Remove special prefixes
        line = re.sub(r"^([\|\:\~\#\*\+\$\{\}\[\]\^\=\_\"]{1,})\s*", "", line)  # Remove special prefixes
        if line.strip():  # Ignore empty lines
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def remove_signature_block(text):
    """Remove email signatures typically found at the end of messages."""
    # Identify separator ("--", "__", excessive whitespace) which often indicates the signature
    #text = re.split(r"(?m)^\s*--+\s*$", text)[0]  # Remove everything after "--" or similar lines
    #text = re.split(r"(?m)^\s*[-_*=]{2,}\s*$", text)[0]
    text = re.sub(r"(?m)^\s*[-_*=]{2,}\s*$.*", "", text, flags=re.DOTALL)

    # Remove lines with common signature patterns (names, emails, phone, fax, addresses)
    text = re.sub(r"(?m)^\s*[\w\s.,-]+\s*\|.*$", "", text)  # Name | Contact format
    text = re.sub(r"(?im)^\s*(Phone|Tel|Fax|Email|Internet|Bitnet|Office|Organization|From|Date|\
                  Address|Res|News-Software|Distribution|Originator|NNTP-Posting-Host|Nntp-Posting-Host|\
                  Keywords|In-reply-to|In-Reply-To|Reply-To|To|NOTE|Disclaimer|\
                  X-Newsreader|X-UserAgent|X-XXDate|X-XXMessage-ID|Newsgroups|\
                  Source|Content-Type|Mime-Version|Archive-name|Last-modifie|Last-modified|Keywords|\
                  Expires|X-Md4-Signature\
                  )[:\s].*$", "", text)  # Common contact fields
    text = re.sub(r"(?m)^\s*\w+@\w+\.\w+.*$", "", text)  # Remove standalone email lines
    text = re.sub(r"(?m)^\s*\(?\+?\d{1,4}[-.\s]?\d{2,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}.*$", "", text)  # Remove phone numbers

    return text.strip()


def extract_subject_and_content(text):
    # Extract subject
    subject_match = re.search(r"^Subject: (.*)", text, re.MULTILINE)
    subject = subject_match.group(1).strip() if subject_match else "No Subject"
    
    # Ensure subject ends with punctuation
    if subject and subject[-1] not in ".!?":
        subject += "."

    # Detect the "Lines:" field to find the main content
    lines_match = re.search(r"^Lines: \d+\s*\n", text, flags=re.MULTILINE)
    if not lines_match:
        return subject, "No Content"
    
    # Extract content starting after "Lines:"
    main_content_start = lines_match.end()
    main_content = text[main_content_start:].strip()

    # Remove text before "writes", "wrote", or "In article"
    main_content = re.sub(r".*?(?:write:|writes:|wrote:|In article\s+<.*?>)", "", main_content, flags=re.MULTILINE)

    # Remove anything after "- cut here -" style markers
    main_content = remove_cut_here_block(main_content)

    # Remove email addresses
    main_content = re.sub(r'[\w\.-]+@[\w\.-]+', '', main_content)

    # Remove quoted lines (previous email content)
    main_content = remove_quoted_lines(main_content)

    # Clean special characters at line beginnings
    main_content = clean_special_symbols(main_content)

    # Remove signature block
    main_content = remove_signature_block(main_content)

    # Remove excessive whitespace
    main_content = re.sub(r'\s+', ' ', main_content).strip()

    return subject, main_content

def process_directory(directory, dataset_type, start_id):
    """Process files in a directory and return structured data."""
    text_data = []
    label_data = {}
    data_label_mapping = []
    
    text_id = start_id

    for label_id, category in enumerate(tqdm(sorted(os.listdir(directory)), desc="Processing categories")):  # Sort for consistency
        category_path = os.path.join(directory, category)
        if not os.path.isdir(category_path):
            continue

        label_data[label_id] = category  # Store label mapping

        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                    text = file.read()
                    subject, content = extract_subject_and_content(text)
                    combined_text = f"{subject} {content}"  # Merge subject + content

                    text_data.append((text_id, combined_text))  # Store text data
                    data_label_mapping.append((text_id, label_id))  # Store mapping
                    text_id += 1  # Increment ID

    return text_data, label_data, data_label_mapping, text_id  # Return new start ID

# Set paths for train and test sets
train_directory = "/Users/tan/Documents/Coding/dataset/20newsgroup/20news-bydate/20news-bydate-train"
test_directory = "/Users/tan/Documents/Coding/dataset/20newsgroup/20news-bydate/20news-bydate-test"

# Process train set
train_data, label_map, train_data_label_mapping, next_id = process_directory(train_directory, "train", start_id=1)

# Process test set (ID continues from last train ID)
test_data, _, test_data_label_mapping, _ = process_directory(test_directory, "test", start_id=next_id)

# Combine text data and text-label mapping
text_data = train_data + test_data
data_label_mapping = train_data_label_mapping + test_data_label_mapping

# Convert to DataFrames
df_train = pd.DataFrame(train_data, columns=["text_id", "text"])
df_test = pd.DataFrame(test_data, columns=["text_id", "text"])
df_label = pd.DataFrame(label_map.items(), columns=["label_id", "label"])
df_data_label_mapping = pd.DataFrame(data_label_mapping, columns=["text_id", "label_id"])

# Save to CSV files
df_train.to_csv("train.csv", sep='\t', index=False)
df_test.to_csv("test.csv", sep='\t', index=False)
df_label.to_csv("label.csv", sep='\t', index=False)
df_data_label_mapping.to_csv("data_label_mapping.csv", sep='\t', index=False)

print("✅ CSV files successfully generated!")