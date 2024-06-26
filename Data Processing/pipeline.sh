#!/bin/bash

classifiers=("BinaryCNN" "BinarySVM" "MultiClassCNN" "OutlierDetection")

print_options() {
  echo "Available classifiers:"
  for i in "${!classifiers[@]}"; do
    echo "$((i+1)) - ${classifiers[i]}"
  done
}

print_options
read -p "Enter the number of the classifier you want to use: " choice

if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#classifiers[@]}" ]; then
  chosen_classifier="${classifiers[choice-1]}"
  echo "Selected classifier: $chosen_classifier"
  echo "Reading the bdf data and removing the unused parts..."
  cd preprocessing
  python split_bdf.py
  echo "Filtering, denoising and preparing the EEG data for classification..."
  python data_preprocessing.py
  echo "Training the classifier(s)..."
  cd ../classification
  python "$chosen_classifier.py"


else
  echo "Invalid choice"
  exit 1
fi