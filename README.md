# Event Information Extraction

A natural language processing system that extracts structured information from event descriptions using parameter-efficient fine-tuning of language models.

## 🔍 Overview

This repository contains code for fine-tuning language models to extract structured data from natural language event descriptions. The system parses free-text event descriptions and converts them into machine-readable JSON formats containing details such as:

- Date and time information
- Event duration
- Participant names
- Location or platform
- Other relevant event attributes

## 📋 Features

- **Parameter-Efficient Fine-Tuning**: Implements LoRA (Low-Rank Adaptation) to efficiently adapt language models
- **Structured Output**: Converts unstructured text to well-formatted JSON data
- **Production Ready**: Includes inference code for deployment
- **Memory Efficient**: Uses optimization techniques to reduce computational requirements

## 🛠️ Technical Stack

- **Framework**: HuggingFace Transformers
- **Base Model**: HuggingFaceTB/SmolLM-360M
- **Fine-tuning Method**: PEFT (Parameter-Efficient Fine-Tuning) with LoRA
- **Data Processing**: Pydantic for validation, HuggingFace Datasets for data management
- **Training Optimizations**: Mixed precision (FP16), gradient checkpointing

## 📊 Example

### Input:

```
Meeting on 05 - December - 2023, 3pm, lasting 1 hour, with Sarah and James on Google Meet.
```

### Output:

```json
{
  "event_type": "meeting",
  "date": "2023-12-05",
  "time": "15:00",
  "duration": "1 hour",
  "participants": ["Sarah", "James"],
  "platform": "Google Meet"
}
```

## 🚀 Getting Started

### Prerequisites

Built using Kaggle

### Training

Run only the training code.

### Inference

```python
from event_extractor import inference

result = inference("Meeting tomorrow at 2pm with the design team to discuss new mockups")
print(result)
```

## 📖 Data Format

The training data should be in JSONL format with each line containing:

```json
{
  "event_text": "Meeting on Monday at 3pm with John",
  "output": {
    "event_type": "meeting",
    "date": "2023-05-15",
    "time": "15:00",
    "participants": ["John"]
  }
}
```

## 📝 Model Architecture

The solution uses a 360M parameter language model fine-tuned with LoRA, which:

- Adds trainable rank decomposition matrices to existing weights
- Updates only ~1-3% of the parameters
- Preserves the base model's knowledge while adapting to event extraction

## 📈 Performance

The model achieves high accuracy in extracting:

- action (e.g. “meeting”)
- date
- time
- attendees
- location
- duration
- recurrence
- notes

## 🔧 Advanced Usage

### Custom LoRA Configuration

```python
from peft import LoraConfig

custom_peft_config = LoraConfig(
    r=32,  # higher rank for more capacity
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Batch Processing

```python
events = [
    "Conference call at 11am on Friday",
    "Dentist appointment on March 3rd at 2:30pm",
    "Team lunch next Tuesday at noon"
]

results = [inference(event) for event in events]
```

## 🔮 Future Work

- Multilingual support for event extraction
- Integration with calendar APIs
- Handling of recurring events
- Browser extension for automatic extraction from emails/messages

## 👨‍💻 Author

**Pranav Arvind Bhile**

## 📄 License

This project is licensed under the MIT License.
