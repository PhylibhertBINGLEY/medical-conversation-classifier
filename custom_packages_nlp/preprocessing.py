## imports
import pandas as pd
import torch

##-----------------------------------------------------------------##
##----------------------LOAD AND PREPROCESSING---------------------##
##-----------------------------------------------------------------##
## 1st coding step : preprocessing (we do tokenizations that would be fed in Bert model)
# Create a function to preprocess data from a .tsv file
# Create a function to preprocess data from a .tsv file
def load_preprocessing_for_bert_tsv(tsv_file, classes_labels, tokenizer):
    """Perform required preprocessing steps for pretrained BERT.
    @param    tsv_file (str): Path to the .tsv file.
    @param    classes_labels (list): List of class labels.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    @return   preprocessed_classes (torch.Tensor): Tensor of class labels.
    @return   ids (torch.Tensor): Tensor of utterance IDs.
    @return   sources (torch.Tensor): Tensor of source information.
    """
    # Load data from the .tsv file using pandas
    data = pd.read_csv(tsv_file, sep='\t')

    # Create empty lists to store outputs (for the utterances)
    input_ids = []
    attention_masks = []
    # Create empty list to store the classes labels
    preprocessed_classes = []
    ids = []
    sources = []

    # For every row in the dataframe...
    for index, row in data.iterrows():
        sent = row['utterance']
        classes = row['classes']
        utterance_id = row['id']
        source = row['source']

        # Encode the sentence
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=300,  # change this value?
            padding='max_length',
            return_attention_mask=True
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

        # Create an array for classes labels
        classes_array = [1 if label in classes.split(',') else 0 for label in classes_labels]
        preprocessed_classes.append(classes_array)

        # Add id and source to the lists
        ids.append(utterance_id)
        sources.append(source)

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    preprocessed_classes = torch.tensor(preprocessed_classes, dtype=torch.float32)
    ids = torch.tensor(ids)
    sources = torch.tensor(sources)

    return input_ids, attention_masks, preprocessed_classes, ids, sources


# Create a function to preprocess data from a test .tsv file
def load_preprocessing_for_bert_test_tsv(tsv_file, tokenizer):
    """Perform required preprocessing steps for pretrained BERT on test data.
    @param    tsv_file (str): Path to the test .tsv file.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    @return   ids (torch.Tensor): Tensor of utterance IDs.
    @return   sources (torch.Tensor): Tensor of source information.
    """
    # Load data from the test .tsv file using pandas
    data = pd.read_csv(tsv_file, sep='\t')

    # Create empty lists to store outputs (for the utterances)
    input_ids = []
    attention_masks = []
    ids = []
    sources = []

    # For every row in the dataframe...
    for index, row in data.iterrows():
        sent = row['utterance']
        utterance_id = row['id']
        source = row['source']

        # Encode the sentence
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=300,  # change this value?
            padding='max_length',
            return_attention_mask=True
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
        ids.append(utterance_id)
        sources.append(source)

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    ids = torch.tensor(ids)
    sources = torch.tensor(sources)

    return input_ids, attention_masks, ids, sources
