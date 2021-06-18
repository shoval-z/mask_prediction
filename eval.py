from utils import *
from dataset import mask_dataset
from tqdm import tqdm
from pprint import PrettyPrinter

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = './'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 64
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = '0_checkpoint_ssd300.pth.tar'

# Load model checkpoint that is to be evaluated
model = torch.load(checkpoint)
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
test_dataset = mask_dataset(dataset='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          num_workers=workers, pin_memory=True)


def evaluate(test_loader, model):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()

    iou_scores = list()
    accuracy_scores = list()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, origin_size) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # det_boxes.extend(det_boxes_batch)
            # det_labels.extend(det_labels_batch)
            # det_scores.extend(det_scores_batch)
            # true_boxes.extend(boxes)
            # true_labels.extend(labels)

            origin_size = origin_size.squeeze(1)

            det_boxes_batch = torch.mul(torch.stack(det_boxes_batch), origin_size)
            det_boxes_batch = xy_to_cxcy(det_boxes_batch)

            boxes = torch.mul(torch.stack(boxes).squeeze(1), origin_size)
            boxes = xy_to_cxcy(boxes)

            iou = [calc_iou(det_b,true_b) for det_b,true_b in zip(det_boxes_batch,boxes)]
            iou_scores.extend(iou)

            det_labels_batch = (torch.stack(det_labels_batch) == 2).int()
            labels = (torch.stack(labels).squeeze(1) == 2).int()
            accuracy_scores.extend((det_labels_batch == labels).int())


        accuracy = np.mean(accuracy_scores)
        iou = np.mean(iou)
        print(accuracy,iou)





        # Calculate mAP
        # APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)




if __name__ == '__main__':
    evaluate(test_loader, model)