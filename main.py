from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
# import torch.nn.functional as F
from sklearn.metrics import f1_score # , accuracy_score

## custom imports
from custom_packages_nlp import preprocessing, classifier, train_eval, predictions


##----------------------------------------------------------##
##----------------------MAIN FUNCTION-----------------------##
##----------------------------------------------------------##
# Main function
def main():
    ##------------TOKENIZER-----------##
    ## 1) we initialize a tokenizer
    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    ##-----------PREPROCESSING OF DATA-----------##
    # Define the classes labels
    classes_labels = ["AM", "MS", "OTHER", "PH", "SF", "SR"]
    ## 2) preprocessing of data
    train_inputs, train_masks, train_labels, train_utt_ids, train_utt_sources = preprocessing.load_preprocessing_for_bert_tsv(
        './data/train.tsv', classes_labels, tokenizer)  # train_labels : y_train
    valid_inputs, valid_masks, valid_labels, valid_utt_ids, valid_utt_sources = preprocessing.load_preprocessing_for_bert_tsv(
        './data/val.tsv', classes_labels, tokenizer)  # valid_labels : y_valid
    test_inputs, test_masks, test_utt_ids, test_utt_sources = preprocessing.load_preprocessing_for_bert_test_tsv('./data/test.tsv',
                                                                                                   tokenizer)

    ##------------CREATING DATA LOADERS-----------##
    # For fine-tuning BERT, a batch size of 16 or 32 is recommended
    batch_size = 16

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    valid_data = TensorDataset(valid_inputs, valid_masks, valid_labels)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    # Create the DataLoader for our testing set
    test_data = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    ##--------------TRAINING & EVALUATION---------------##
    # Specify loss function
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    train_eval.set_seed(10)  # Set seed for reproducibility
    bert_classifier, optimizer, scheduler = classifier.initialize_model(device, train_dataloader, classes_labels, epochs=2)
    train_eval.train(bert_classifier, train_dataloader, loss_fn, device, optimizer, scheduler, valid_dataloader, epochs=2,
          evaluation=True)

    ##--------------TESTING PART---------------##
    probs = classifier.bert_predict(bert_classifier, valid_dataloader, device)
    print("probs")
    print("shape of probs: ", probs.shape)
    print(probs)
    f1 = f1_score(valid_labels, probs, average='macro')

    print(f" Macro-F1: {f1:.4f}")

    ##--------------PREDICTIONS---------------##
    # Make predictions on the test set
    test_probs = classifier.bert_predict(bert_classifier, test_dataloader, device, threshold=0.5)

    # Convert probabilities to binary predictions (0 or 1)
    test_preds = (test_probs > 0.5).astype(int)

    # Write predictions to CSV
    predictions.write_predictions_to_csv(test_utt_ids.cpu().numpy(), test_preds, classes_labels, 'test_predictions.csv')


if __name__ == '__main__':
    main()
