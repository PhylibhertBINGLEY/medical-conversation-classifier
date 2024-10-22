## imports
import torch.nn as nn
from transformers import BertModel, get_linear_schedule_with_warmup, AdamW



##----------------------------------------------------------##
##----------------------BERT CLASSIFIER---------------------##
##----------------------------------------------------------##
# Create the BertClassfier class
class BertClassifier(nn.Module):
    def __init__(self, classes_labels, freeze_bert=False):
        super(BertClassifier, self).__init__()

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)

        # add your additional layers, for example, a dropout layer followed by a linear classification
        self.dropout = nn.Dropout(0.3)
        self.hidden = nn.Linear(768, 256)
        self.out = nn.Linear(256, len(classes_labels))
        # self.out = nn.Linear(256, 3)

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        sequence_output, pooled_output = self.bert(input_ids=input_ids,
                                                   attention_mask=attention_mask)

        # apply dropout to the BERT output
        pooled_output = self.dropout(pooled_output)
        hidden = self.hidden(pooled_output)

        logits = self.out(hidden)

        return logits



##----------------------------------------------------------##
##--------------------MODEL INITIALIZATION------------------##
##----------------------------------------------------------##
def initialize_model(device, train_dataloader, classes_labels, epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(classes_labels, freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=your_learning_rate)
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,  # Default learning rate
                      eps=1e-8  # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler
