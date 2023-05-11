from typing import List, Tuple
from dataclasses import dataclass
import random

import torch
from sklearn.metrics import classification_report, f1_score
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, \
    AutoModelForSequenceClassification as AMSC

@dataclass
class DataPoint:
    text: str
    label: str
    is_positive: bool


class MultilabelDataset(Dataset):
    def __init__(self, _tokenizer: AutoTokenizer, data_points: List[DataPoint],
                 max_length=512, inherited_labels: List[str] = None):
        self.tokenizer = _tokenizer
        self.data_points = data_points
        self.max_length = max_length

        if inherited_labels:
            self.unique_labels = inherited_labels
        else:
            self.unique_labels = list(set(dp.label for dp in self.data_points))
        self.num_classes: int = len(self.unique_labels)
        self.indices = list(range(self.num_classes))
        self.label_mapping = {label: idx for idx, label in zip(self.indices, self.unique_labels)}
        self.label_inverse_mapping = {v: k for k, v in self.label_mapping.items()}

    def __len__(self):
        return len(self.data_points)

    def label_to_id(self, label: str) -> int:
        return self.label_mapping[label]

    def id_to_label(self, _id: int) -> str:
        return self.label_inverse_mapping[_id]

    def _override_num_classes(self, num_classes):
        self.num_classes = num_classes
        return self

    def split(self, train_size, val_size, test_size, shuffle=True):
        if shuffle:
            data_points = self.data_points.copy()
            random.shuffle(data_points)
        else:
            data_points = self.data_points

        total_size = len(data_points)
        train_end = int(train_size * total_size)
        val_end = train_end + int(val_size * total_size)

        train_data = data_points[:train_end]
        val_data = data_points[train_end:val_end]
        test_data = data_points[val_end:]

        _train_dataset = MultilabelDataset(self.tokenizer, train_data, inherited_labels=self.unique_labels)
        _val_dataset = MultilabelDataset(self.tokenizer, val_data, inherited_labels=self.unique_labels)
        _test_dataset = MultilabelDataset(self.tokenizer, test_data, inherited_labels=self.unique_labels)

        return _train_dataset, _val_dataset, _test_dataset

    def __getitem__(self, idx):
        data_point = self.data_points[idx]
        probabilities_tensor = torch.zeros(len(self.unique_labels))  # initialize with correct shape
        id_target = self.label_to_id(data_point.label)
        if data_point.is_positive:
            probabilities_tensor[id_target] = 1
        else:
            idxs = self.indices.copy()
            del idxs[id_target]
            probabilities_tensor[idxs] = 1 / (self.num_classes - 1)

        # noinspection PyCallingNonCallable
        encoding = self.tokenizer(
            data_point.text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt',
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}, probabilities_tensor


def predict(text: str, model: AMSC, tokenizer: AutoTokenizer, device: torch.device, threshold: float):
    model.eval()
    inputs = tokenizer(text, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > threshold).int()

    return probabilities, predictions


def get_examples():
    positive_examples = [
        ("fg", 'fg_unclear'),
        ("grp", 'fg_unclear'),
        ("composite", 'fg_unclear'),
        ("wood or fg", 'fg_unclear'),
        ("fg composite", 'fg_unclear'),
        ("fg or plywood", 'fg_unclear'),
        ("frp", 'fg_unclear'),
        ("fg or wood", 'fg_unclear'),
        ("", 'fg_unclear'),

        ("fg solid laminate", 'fg_solid'),
        ("fg solid", 'fg_solid'),
        ("fg with vinylester resin", 'fg_solid'),
        ("fg with  vacuum infusion", 'fg_solid'),
        ("solid frp", 'fg_solid'),
        ("fg solid hull ply sandwich deck", 'fg_solid'),
        ("fg solid hull plywood cored deck", 'fg_solid'),
        ("fg  solid hull and balsa deck ", 'fg_solid'),
        ("fg  solid laminate ", 'fg_solid'),
        ("fg solid hull  balsa cored deck", 'fg_solid'),
        ("fg solid lamine hull  cored deck ", 'fg_solid'),
        ("fg solid", 'fg_solid'),
        ("fg solid laminate below waterline", 'fg_solid'),
        ("fg  solid hull balsa cored deck ", 'fg_solid'),
        ("fg solid lam  hull sandwich deck", 'fg_solid'),
        ("fg  solid hull and balsa cored deck", 'fg_solid'),
        ("grp with e glass and vinylester resin", 'fg_solid'),
        ("", 'fg_solid'),

        ("", 'fg_foam'),
        ("glass foam sand ", 'fg_foam'),
        ("fg foam sandwich", 'fg_foam'),
        ("fg with divinycell core", 'fg_foam'),
        ("fg with airex core", 'fg_foam'),
        ("fg with foam core", 'fg_foam'),
        ("fg with klegecell core", 'fg_foam'),
        ("fg foam sand  hull and deck", 'fg_foam'),
        ("grp infused pvc foam sandwich", 'fg_foam'),
        ("fg with  foam sandwich", 'fg_foam'),
        ("epoxy foam sand  with e glass", 'fg_foam'),
        ("fg foam sand  hull and deck", 'fg_foam'),
        ("grp infused pvc foam sandwich", 'fg_foam'),
        ("glass kev nomex", 'fg_foam'),


        ("wood fg", 'fg_wood'),
        ("fg with plywood cored deck", 'fg_wood'),
        ("wood grp", 'fg_wood'),
        ("fg with balsa cored deck", 'fg_wood'),
        ("plywood fg", 'fg_wood'),
        ("fg balsa cored deck", 'fg_wood'),
        ("vacuum infused polyester with balsa core", 'fg_wood'),
        ("fg with balsa core hull and deck", 'fg_wood'),
        ("fg with balsa cored hull and deck", 'fg_wood'),
        ("fg with balsa core", 'fg_wood'),
        ("fg with balsa core deck", 'fg_wood'),
        ("fg wood", 'fg_wood'),
        ("wood fg composite", 'fg_wood'),
        ("", 'fg_wood'),

        ("aluminum", 'metal'),
        ("steel alu", 'metal'),
        ("aluminium", 'metal'),
        ("alu", 'metal'),
        ("steel", 'metal'),
        ("aluminum or steel", 'metal'),
        ("steel alum ", 'metal'),
        ("steel alum  or wood", 'metal'),
        ("tubular steel", 'metal'),
        ("wood   steel", 'metal'),
        ("steel  triple chine ", 'metal'),
        ("fg foam sand  with steel frame", 'metal'),
        ("wood planked on steel", 'metal'),

        ("wood carvel", 'wood'),
        ("wood", 'wood'),
        ("wood  mahogany ", 'wood'),
        ("plywood", 'wood'),
        ("plywood", 'wood'),
        ("wood planked", 'wood'),
        ("wood clinker", 'wood'),
        ("marine ply", 'wood'),
        ("plywood epoxy", 'wood'),
        ("plywood single chine", 'wood'),
        ("wood  mahog  on oak ", 'wood'),
        ("", 'wood'),

        ("roto molded poly", 'others'),
        ("roto molded polyethylene", 'others'),
        ("carbon fiber", 'others'),
        ("roto moulded polyethylene", 'others'),
        ("", 'others'),
    ]
    negative_examples = [


    ]
    # negative_examples = [
    #     ("The squirrel climbed the tree effortlessly.", 'boats'),
    #     ("The sun set behind the mountains, painting the sky orange.", 'boats'),
    #     ("The car's engine roared as it raced down the highway.", 'boats'),
    #     ("The baker prepared a fresh batch of bread rolls.", 'boats'),
    #     ("The violinist played a beautiful melody.", 'boats'),
    #
    #     ("The building's architecture was breathtaking.", 'literature'),
    #     ("The garden was filled with colorful flowers and butterflies.", 'literature'),
    #     ("The chef prepared an exquisite meal for the guests.", 'literature'),
    #     ("The athlete broke the world record in the 100-meter race.", 'literature'),
    #     ("The artist painted a stunning landscape.", 'literature'),
    #
    #     ("The waterfall crashed into the pool below.", 'computation'),
    #     ("The dog chased its tail in circles.", 'computation'),
    #     ("The chef chopped the vegetables quickly.", 'computation'),
    #     ("The ballerina danced gracefully across the stage.", 'computation'),
    #     ("The hiker trekked through the dense forest.", 'computation'),
    #     ]
    # ]    positive_examples = [
    #     ("The sailboat glided smoothly across the water.", 'boats'),
    #     ("A fleet of ships was anchored near the shore.", 'boats'),
    #     ("The kayak flipped, leaving the paddler in the cold water.", 'boats'),
    #     ("The motorboat sped through the waves with ease.", 'boats'),
    #     ("The harbor was filled with yachts and fishing boats.", 'boats'),
    #
    #     ("Pride and Prejudice is a classic work of literature.", 'literature'),
    #     ("Shakespeare's plays have been studied for centuries.", 'literature'),
    #     ("J.K. Rowling is the author of the Harry Potter series.", 'literature'),
    #     ("Charles Dickens wrote novels about Victorian England.", 'literature'),
    #     ("The Iliad and The Odyssey are ancient Greek epic poems.", 'literature'),
    #
    #     ("The computer processed the data quickly and efficiently.", 'computation'),
    #     ("Quantum computing is an emerging field of study.", 'computation'),
    #     ("Machine learning algorithms can recognize patterns in data.", 'computation'),
    #     ("Programming languages like Python and Java are widely used.", 'computation'),
    #     ("The microprocessor is an essential component of modern computers.", 'computation'),
    # ]
    #
    # negative_examples = [
    #     ("The squirrel climbed the tree effortlessly.", 'boats'),
    #     ("The sun set behind the mountains, painting the sky orange.", 'boats'),
    #     ("The car's engine roared as it raced down the highway.", 'boats'),
    #     ("The baker prepared a fresh batch of bread rolls.", 'boats'),
    #     ("The violinist played a beautiful melody.", 'boats'),
    #
    #     ("The building's architecture was breathtaking.", 'literature'),
    #     ("The garden was filled with colorful flowers and butterflies.", 'literature'),
    #     ("The chef prepared an exquisite meal for the guests.", 'literature'),
    #     ("The athlete broke the world record in the 100-meter race.", 'literature'),
    #     ("The artist painted a stunning landscape.", 'literature'),
    #
    #     ("The waterfall crashed into the pool below.", 'computation'),
    #     ("The dog chased its tail in circles.", 'computation'),
    #     ("The chef chopped the vegetables quickly.", 'computation'),
    #     ("The ballerina danced gracefully across the stage.", 'computation'),
    #     ("The hiker trekked through the dense forest.", 'computation'),
    # ]
    return [DataPoint(text, label, True) for text, label in positive_examples] \
        + [DataPoint(text, label, False) for text, label in negative_examples]


def train_(model, device, num_epochs, train_dataset, batch_size, threshold, tokenizer, val_dataset=None):
    global train_accs

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # activate Train mode
    model.train()
    # Set up the optimizer and define the training loop:
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(num_epochs):
        train_correct = 0
        train_total = 0
        for batch in train_dataloader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)
            predicted_labels = (probabilities > threshold).int()
            train_total += labels.size(0)
            train_correct += (predicted_labels == labels).all(dim=1).sum().item()


        train_accuracy = train_correct / train_total


        train_accs.append(train_accuracy)
        print(f'Epoch {epoch + 1}\nTrain Accuracy: {train_accuracy:.4f}')

        # Cross validation
        if val_dataset:
            validate(model, batch_size, threshold, device, tokenizer, val_dataset)
    return model


def validate(model: AMSC, batch_size: int, threshold: float, device: torch.device, tokenizer: AutoTokenizer,
             val_dataset: MultilabelDataset):
    global val_accs
    # Evaluate the model on the validation dataset:
    model.eval()
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    correct = 0
    total = 0

    with torch.no_grad():
        val_true_labels = []
        val_predicted_labels = []
        for batch in val_dataloader:
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)

            predicted_labels: torch.Tensor = (probabilities > threshold).int()

            val_true_labels.extend(labels.cpu().numpy().tolist())
            val_predicted_labels.extend(predicted_labels.cpu().numpy().tolist())

            total += labels.size(0)
            correct += (predicted_labels == labels).all(dim=1).sum().item()

            print_val_info(inputs, labels, predicted_labels, tokenizer, val_dataset)


    confusion_mat = classification_report(val_true_labels, val_predicted_labels)

    print(f'Confusion Matrix :\n{confusion_mat}')

    accuracy = correct / total

    f1 = f1_score(val_true_labels, val_predicted_labels, average=None)
    val_accs.append(f1)
    print(f'Validation Accuracy: {accuracy:.4f}\n')
    return model


VAL_INFO_SWITCH = True


def print_val_info(inputs, labels, predicted_labels, tokenizer, val_dataset):
    # Correctly print the predicted labels
    global VAL_INFO_SWITCH

    if VAL_INFO_SWITCH:
        print('//////validation////////')
        VAL_INFO_SWITCH = False

    for i, pred_label in enumerate(predicted_labels):
        predicted_label_names = [val_dataset.id_to_label(idx) for idx, val in enumerate(pred_label.tolist())
                                 if val == 1]
        true_label_names = [val_dataset.id_to_label(idx) for idx, val in enumerate(labels[i].tolist())
                            if val == 1]

        text = tokenizer.decode(inputs['input_ids'][i], skip_special_tokens=True)

        print(f'Text: {text}\nTrue labels: {true_label_names}\nPredicted labels: {predicted_label_names}\n')


def main():
    BATCH_SIZE = 4
    NUM_EPOCHS = 15
    MODEL_NAME = "distilbert-base-uncased"
    THRESHOLD = 0.3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    examples = get_examples()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = MultilabelDataset(tokenizer, examples)

    label_mapping: dict = dataset.label_mapping
    index_mapping: dict = dataset.label_inverse_mapping

    train_dataset, val_dataset, _ = dataset.split(0.7, 0.3, 0)

    model = AMSC.from_pretrained(MODEL_NAME, num_labels=len(index_mapping))

    model.to(DEVICE)

    model = train_(model, DEVICE, NUM_EPOCHS, train_dataset, BATCH_SIZE, THRESHOLD, tokenizer, val_dataset)

    model = validate(model, BATCH_SIZE, THRESHOLD, DEVICE, tokenizer, val_dataset)

    with open('./data/txt/to_evaluate.txt') as f:
        print(r'\\\\\\\\\\\\\\\Deployment')
        for line in f:
            probabilities, predictions = predict(line, model, tokenizer, DEVICE, threshold=THRESHOLD)



            predicted_label_id = probabilities.argmax().item()
            predicted_label = index_mapping[predicted_label_id]

            prob_dict = {
                label: round(prob, 4)
                for label, prob
                in zip(dataset.unique_labels, probabilities.cpu().flatten().tolist())
            }

            # print(f'Text: {line.strip()}\nPredicted label: {predicted_label}\nProbabilities: {prob_dict}\n')
    print(dataset.label_mapping)

if __name__ == '__main__':
    train_accs = []
    val_accs = []
    main()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    line1, = ax.plot(train_accs)
    line2, = ax.plot(val_accs)
    plt.show()


