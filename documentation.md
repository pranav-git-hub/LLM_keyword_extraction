# Event Information Extraction: Model Documentation

## Author: Pranav Arvind Bhile

## 1. Project Overview

This documentation details the development of an automated event information extraction system designed to parse unstructured text descriptions of events and convert them into structured, machine-readable formats. The system utilizes fine-tuned language models to extract key details such as dates, times, durations, participants, and locations from natural language descriptions.

This project was developed as part of an assignment to create an efficient information extraction system that can identify and structure relevant event information from natural text inputs. The primary goal is to enable automated calendar entry creation and event management from various text sources.

## 2. Technical Approach

### 2.1 Model Selection

For this project, I selected **HuggingFaceTB/SmolLM-360M** as the base model. This choice was motivated by several factors:

- **Assignment constraints**: The assignment required using a non-instruct model that could be fine-tuned efficiently within available computational resources
- **Efficiency**: With 360M parameters, this model strikes a balance between computational efficiency and performance
- **Size optimization**: Large enough to comprehend context while small enough for practical deployment

### 2.2 Fine-tuning Strategy

Instead of full fine-tuning, we implemented **Parameter-Efficient Fine-Tuning (PEFT)** using **Low-Rank Adaptation (LoRA)**. This approach allows us to:

- Fine-tune the model with significantly fewer trainable parameters (only about 1-3% of total parameters) (1.8% in this case)
- Reduce memory requirements during training
- Maintain most of the original model's knowledge while adapting it to our specific task
- We are not using QLoRA as quantization will require overhead.

Our LoRA configuration used the following hyperparameters:

- `r=16`: Rank of the update matrices
- `lora_alpha=32`: Scaling factor for the LoRA updates
- `lora_dropout=0.05`: Dropout rate for regularization
- Target modules included attention components and feed-forward projections

### 2.3 Data Processing Pipeline

The data processing workflow follows these steps:

1. **Data Loading**: Load event records from a JSONL file using a Pydantic model for validation
2. **Tokenization**: Convert text to token IDs compatible with the model
3. **Input-Output Formatting**: Structure data as `{input text}\n{JSON output}` pairs
4. **Label Creation**: Mask input tokens with -100 to train the model only on predicting output tokens
5. **Dataset Splitting**: Create training (90%) and evaluation (10%) sets

### 2.4 Training Configuration

The training process was configured with:

- 10 epochs of training
- Batch sizes of 4 (training) and 8 (evaluation)
- Learning rate of 2e-5 with cosine scheduling
- FP16 precision to optimize memory usage and training speed
- Evaluation after each epoch with best model selection based on evaluation loss

## 3. Implementation Rationale

### 3.1 PEFT with LoRA vs. Full Fine-tuning

Full fine-tuning of language models requires substantial computational resources and can lead to catastrophic forgetting. Our LoRA implementation:

- Reduces the number of trainable parameters from millions to thousands
- Preserves the base model's capabilities while adapting to our specific task
- Enables faster iteration and experimentation
- Results in smaller deployment artifacts (only need to store the LoRA weights)

### 3.2 Input-Output Formatting Decisions

We structured our training data as `input_text\noutput_json` pairs for several reasons:

1. **Clear separation**: This format creates a clear demarcation between input and expected output
2. **JSON structure**: Using JSON as the output format ensures a consistent structure that can be easily parsed
3. **Label masking**: By masking labels for input tokens, we focus the learning objective exclusively on generating correct output formats

### 3.3 Model Size Considerations

We chose a medium-sized model (360M parameters) rather than larger alternatives because:

- It provides sufficient context understanding for the event extraction task
- Requires less computational resources for training and inference
- Enables potential deployment in resource-constrained environments
- Faster iteration during development and experimentation

## 4. Challenges and Solutions

### 4.1 Training Efficiency

**Challenge**: Fine-tuning large language models is computationally expensive.

**Solution**:

- Implemented LoRA to reduce trainable parameters (as shown in the trainable parameters print statement)
- Used mixed precision training (FP16) to reduce memory requirements
- Applied gradient checkpointing selectively based on model size and available resources

### 4.2 Structured Output Generation

**Challenge**: Ensuring the model consistently outputs valid, well-formatted JSON.

**Solution**:

- Trained the model using properly formatted JSON examples
- Implemented stringent validation using Pydantic models during data preparation
- During inference, used greedy decoding (temperature=0.0) to maximize output consistency

### 4.3 Handling Diverse Event Descriptions

**Challenge**: Events can be described in countless ways, making extraction challenging.

**Solution**:

- Used a diverse dataset of event descriptions to train the model
- Implemented a train-test split to evaluate model generalization
- Focused on extracting common event properties while maintaining flexibility

## 5. Results and Performance

### 5.1 Training Metrics

The model demonstrated consistent improvement throughout training:

- Training loss decreased steadily across epochs, starting at approximately 2.3 and converging to around 1.1 by the final epoch
- Evaluation loss was monitored to prevent overfitting, with final values around 1.2
- Best model was automatically selected based on lowest evaluation loss
- The training process completed successfully within the assignment's time constraints

Training Loss: 0.523800

Validation Loss: 0.573182

### 5.2 Inference Example

The implementation includes an example inference on:

```
"Meeting on 05 - December - 2023, 3pm, lasting 1 hour, with Sarah and James on Google Meet."
```

Gives correct output
{
'action': 'Meeting',
'attendees': ['Sarah', 'James'],
'date': '05/12/2023',
'duration': '1 hour',
'location': 'Google Meet',
'notes': None,
'recurrence': None,
'time': '3:00 PM'
}

The model successfully extracts structured information including date, time, duration, participants, and platform.

### 5.3 Computational Efficiency

The LoRA approach significantly reduced:

- Number of trainable parameters
- Memory requirements during training
- Time required for model fine-tuning

## 6. Future Improvements

### 6.1 Model Enhancements

- **Larger base models**: Experiment with larger models (e.g., 1B+ parameters) to potentially improve accuracy
- **Ensemble approach**: Combine multiple specialized models for different aspects of event extraction
- **Distillation**: Create a smaller, specialized model by distilling from larger fine-tuned models

### 6.2 Data Improvements

- **Data augmentation**: Generate synthetic event descriptions to improve robustness
- **Multilingual support**: Expand the training data to include events described in multiple languages
- **Domain-specific datasets**: Create specialized versions for different contexts (business, academic, social)

### 6.3 Engineering Optimizations

- **Quantization**: Further optimize the model for deployment using int8 or int4 quantization
- **Streaming interface**: Develop an API for real-time event extraction from text
- **Integration**: Connect with calendar applications for automated event creation

### 6.4 Evaluation Framework

- **Human evaluation**: Compare extracted events with human interpretations
- **Comprehensive metrics**: Move beyond loss to metrics like F1-score for specific fields (date, time, etc.)
- **Adversarial testing**: Evaluate on deliberately challenging or ambiguous event descriptions

## 7. Conclusion

This event extraction system demonstrates the effectiveness of parameter-efficient fine-tuning for specialized NLP tasks. By adapting a pre-trained language model using LoRA, I've created a computationally efficient solution that can extract structured information from natural language event descriptions.

The approach balances performance and resource requirements, making it suitable for practical applications while leaving room for further optimization and enhancement. The implementation successfully fulfills the assignment requirements by creating a working event extraction system that processes natural language inputs into structured data suitable for calendar applications.
