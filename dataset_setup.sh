#!/bin/bash

# Define the file ID and output filename
FILE_ID="1TtPziuwoaGKMUmgFq5Am9KJdhppVZkkr"
OUTPUT_FILE="Potholes.zip"

# Use gdown to download the file from Google Drive
echo "Downloading file with gdown..."
gdown "https://drive.google.com/uc?export=download&id=$FILE_ID" -O $OUTPUT_FILE

# Check if download was successful
if [ -f "$OUTPUT_FILE" ]; then
    echo "Download complete: $OUTPUT_FILE"
    
    # Extract the file
    echo "Extracting file..."
    unzip $OUTPUT_FILE 
    
    # Check if extraction was successful
    if [ $? -eq 0 ]; then
        echo "Extraction complete. Files are in the directory."
    else
        echo "Extraction failed."
    fi
else
    echo "Download failed."
fi

# Remove the zip file
rm $OUTPUT_FILE

# Remove the __MACOSX directory from datasets folder
rm -r __MACOSX

# run the python script to split the dataset
python split_dataset.py

# run the object proposal script
python crop_dataset.py


# write to terminal that script is complete'
echo "Setup complete."

