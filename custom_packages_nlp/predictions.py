## imports
import pandas as pd
import torch



def bert_predict(model, test_dataloader, device, threshold=0.5):
    """Perform a forward pass on the trained BERT model to predict binary labels
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply threshold for binary classification and convert to integers
    binary_preds = (torch.sigmoid(all_logits) > threshold).cpu().numpy().astype(int)

    return binary_preds

# Function to write predictions to CSV
def write_predictions_to_csv(utt_ids, preds, classes_labels, output_file):
    # Create a DataFrame with the predicted classes
    df = pd.DataFrame(preds, columns=classes_labels)

    # Add the utterance IDs as a new column
    df.insert(0, 'id', utt_ids)

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to: {output_file}")