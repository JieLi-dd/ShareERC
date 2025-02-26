import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def eval_meld(results, truths, test=False):
    test_preds = results.cpu().detach().numpy()   #（num_utterance, num_label)
    test_truth = truths.cpu().detach().numpy()  #（num_utterance）
    predicted_label = []
    true_label = []
    for i in range(test_preds.shape[0]):
        predicted_label.append(np.argmax(test_preds[i,:],axis=0) ) #
        true_label.append(test_truth[i])
    wg_av_f1 = f1_score(true_label, predicted_label, average='weighted')
    if test:
        f1_each_label = f1_score(true_label, predicted_label, average=None)
        # print('**TEST** | f1 on each class (Neutral, Surprise, Fear, Sadness, Joy, Disgust, Anger): \n', f1_each_label)
        print('**TEST** | f1 on each class (Neutral, Frustrated, Angry, Sad, Happy, Excited): \n', f1_each_label)

        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(true_label, predicted_label)
        print('**TEST** | Confusion Matrix:\n', conf_matrix)

        # Normalize the confusion matrix to be in the range [0, 1]
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        print('**TEST** | Normalized Confusion Matrix:\n', conf_matrix_normalized)

        # Plot the normalized confusion matrix
        plt.figure(figsize=(10, 10))
        # sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False,
        #             xticklabels=['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger'],
        #             yticklabels=['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger'])
        sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                    xticklabels=['Neutral', 'Frustrated', 'Angry', 'Sad', 'Happy', 'Excited'],
                    yticklabels=['Neutral', 'Frustrated', 'Angry', 'Sad', 'Happy', 'Excited'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Normalized Confusion Matrix')

        # Save the plot if a save path is provided
        save_path = '/home/lijie/MasterStudy/ResDialogue/confusion_matrix_iemocap.png'
        if save_path:
            plt.savefig(save_path)
            print(f'Confusion matrix saved to {save_path}')
        else:
            plt.show()

    return wg_av_f1

