import random
import matplotlib.pyplot as plt
from torchvision.transforms import v2

def infinite_visualization_firstframes(set, additional_transform = v2.ToPILImage()): #first frames of diff vids
    while(True):
            first_frames = []
            n_rows=4
            n_columns=5
            random_video_id = random.randint(0, len(set)-1)
            for i in range(n_rows*n_columns):
                vid_i, _ = set[random_video_id]
                first_frames.append(additional_transform(vid_i[0]))

            fig, axes = plt.subplots(n_rows, n_columns, figsize=(10, 6), layout="constrained")
            fig.canvas.manager.set_window_title(f"video # {random_video_id} illustration")
            fig.subplots_adjust(wspace=0.02, hspace=0.02)

            for i in range(n_rows):
                for j in range(n_columns):
                    axes[i][j].imshow(first_frames[i*n_columns+j])
                    axes[i][j].axis("off")
            plt.tight_layout()
            plt.show()

def infinite_visualization(set, additional_transform = v2.ToPILImage()): # frames of the same vid
    while(True):
            frames = []
            n_rows=4
            n_columns=4
            random_video_id = 544 #random.randint(0, len(set)-1)
            vid, _ = set[random_video_id] #production_ready= False required for (T,C,H,W)
            print(vid.shape)

            for i in range(n_rows*n_columns):
                frames.append(additional_transform(vid[i]))

            fig, axes = plt.subplots(n_rows, n_columns, figsize=(10, 6), layout="constrained")
            fig.canvas.manager.set_window_title(f"video # {random_video_id} illustration")
            fig.subplots_adjust(wspace=0.02, hspace=0.02)

            for i in range(n_rows):
                for j in range(n_columns):
                    axes[i][j].imshow(frames[i*n_columns+j])
                    axes[i][j].axis("off")
            plt.tight_layout()
            plt.show()