Readme 
# Anemia Few-Shot Regression with MAML

This notebook (Anemia_regression_V1-3.ipynb) implements a *few-shot, first-order MAML* model to predict hemoglobin (Hb) from conjunctiva images.

This README is written for someone opening the notebook for the first time. It explains:

- Where to set *data paths*
- What *dataset structure* is expected
- How the *MAML code* is organized and how it works
- How to *run* the notebook end-to-end

---

## 1. Data & Folder Structure

### 1.1. Base dataset directory

Early in the notebook you will see:

```python
## Set Dataset Paths

# Define dataset directories
BASE_DIR  = Path("/content/drive/MyDrive/Anemia_dataset")
INDIA_DIR = BASE_DIR / "India"
ITALY_DIR = BASE_DIR / "Italy"

# Quick existence check
print("Base dir exists:", BASE_DIR.exists())
print("India dir exists:", INDIA_DIR.exists())
print("Italy dir exists:", ITALY_DIR.exists())
👉 This is the main place where you need to change the path.
	•	If you’re not using Google Drive / Colab, change BASE_DIR to your local or server path, e.g.: BASE_DIR = Path("/home/username/data/Anemia_dataset")
	•	
	•	INDIA_DIR and ITALY_DIR are derived automatically from BASE_DIR. Keep their definitions as they are unless your folder names differ.
1.2. Expected directory layout
The notebook expects something like:
Anemia_dataset/
├── India/
│   ├── India.xlsx          # metadata for Indian patients
│   ├── ... image folders / files ...
├── Italy/
│   ├── Italy.xlsx          # metadata for Italian patients
│   ├── ... image folders / files ...
└── Augmented_Data/
    ├── India/
    │   ├── ... augmented images ...
    └── Italy/
        ├── ... augmented images (optional) ...
Later, augmented data paths are defined as:
## Set Augmented Dataset Paths

AUG_BASE      = BASE_DIR / "Augmented_Data"
AUG_INDIA_DIR = AUG_BASE / "India"
AUG_ITALY_DIR = AUG_BASE / "Italy"

print("Aug base exists:", AUG_BASE.exists())
print("Aug India exists:", AUG_INDIA_DIR.exists())
print("Aug Italy exists:", AUG_ITALY_DIR.exists())
If you don’t have augmented data yet, you can:
	•	Either create the above structure
	•	Or temporarily set use_augmented=False when building image tables (already done for Italy; you can mirror that behavior for India if needed).
1.3. Metadata files
Two main Excel files are used:
# India metadata
INDIA_XLSX = INDIA_DIR / "India.xlsx"
df_india = pd.read_excel(INDIA_XLSX)

# Italy metadata
ITALY_XLSX = ITALY_DIR / "Italy.xlsx"
df_italy = pd.read_excel(ITALY_XLSX)
These Excel files are expected to contain at least:
	•	Number → patient ID
	•	Label → anemia / non-anemia class
	•	Hgb → hemoglobin value
If your column names differ, you’ll need to update the relevant functions (e.g. build_rg_df) to match your column names.

2. Environment & Dependencies
The notebook uses:
	•	Python 3
	•	PyTorch
	•	NumPy
	•	pandas
	•	scikit-learn
	•	OpenCV (cv2)
	•	matplotlib
	•	(optionally) GPU (CUDA) for faster training
In Colab, most of these are available by default; any missing libs are usually installed with pip at the top of the notebook.
If running locally, install dependencies with e.g.:
pip install torch torchvision torchaudio \
            numpy pandas scikit-learn opencv-python matplotlib openpyxl

3. Notebook Structure Overview
At a high level, the notebook does:
	1.	Mount Drive (optional / Colab only) and import all dependencies.
	2.	Set dataset paths (BASE_DIR, INDIA_DIR, ITALY_DIR).
	3.	Load metadata from India and Italy Excel files.
	4.	Build an R/G ratio dataset for each patient (feature engineering).
	5.	Create patient-level tables and splits (train / val / test).
	6.	Expand to image-level tables, including augmented images.
	7.	Define PyTorch dataset and episodic sampler for few-shot learning.
	8.	Define the few-shot CNN model (FewShotConvNet).
	9.	Define inner-loop adaptation (inner_adapt_params) for MAML.
	10.	Define MAML training loop (train_maml) with validation-based early stopping.
	11.	Train the MAML model on India train episodes.
	12.	Evaluate on India validation and Italy test splits using episodic evaluation and compute regression + classification metrics.
The rest of this README explains the MAML-related parts in more detail.

4. Data Processing & Episode Creation
4.1. R/G ratio data (build_rg_df)
The function:
def build_rg_df(df_meta, base_dir, id_col="Number", hb_col="Hgb", label_col="Label"):
    ...
	•	Iterates over each patient in the metadata.
	•	Locates a conjunctiva image for that patient under base_dir.
	•	Computes a simple R/G ratio as a color feature.
	•	Builds a dataframe with:
	•	patient_id
	•	hb (hemoglobin)
	•	label (anemia/non-anemia)
	•	image_path
	•	rg_ratio
You generally don’t need to edit this function unless:
	•	Your image filenames / subdirectory structure is very different, or
	•	Your metadata column names differ and you aren’t using the id_col / hb_col / label_col arguments.
4.2. Patient-level tables and splits
For India:
df_india_patients = (
    df_india[["Number", "Label", "Hgb"]]
    .drop_duplicates()
    .rename(columns={"Number": "patient_id", "Hgb": "hb"})
)

# Attach split: "train" / "val"
df_india_patients["split"] = np.where(
    df_india_patients["patient_id"].isin(train_patients),
    "train",
    "val",
)
For Italy, a similar patient-level table is built and assigned primarily to a test split.
Here, train_patients etc. are typically determined by some split logic (e.g. random split or specified by you). Each patient is assigned to exactly one split.
4.3. Image-level tables with augmentation
Images are listed via:
df_images_india = build_image_table_from_augmented(
    df_india_patients,
    AUG_INDIA_DIR,
    country_name="India",
)

df_images_italy = build_image_table_from_augmented(
    df_italy_patients,
    AUG_ITALY_DIR,
    country_name="Italy",
    use_augmented=False,  # Italy uses only original images
)
build_image_table_from_augmented:
	•	For each patient and split, it finds:
	•	The original conjunctiva image.
	•	Any augmented versions (in subfolders or with a naming pattern under AUG_* dirs).
	•	Returns a dataframe with one row per image:
	•	patient_id
	•	split
	•	label_str (e.g. "anemic"/"non-anemic")
	•	hb
	•	image_path
	•	source ("original" or "augmented")
	•	country
Then:
df_images = pd.concat([df_images_india, df_images_italy])
This gives a single image-level dataframe covering both countries.

5. Dataset & EpisodeSampler (Core for MAML)
5.1. PyTorch dataset: ConjunctivaImageDataset
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
])

class ConjunctivaImageDataset(Dataset):
    """
    PyTorch dataset for conjunctiva images (original + augmented).
    Returns:
        - img_tensor: [C, H, W]
        - hb: float
        - patient_id: int
    """
    ...
	•	It reads each image using OpenCV (cv2.imread(row["image_path"])).
	•	Converts BGR → RGB, then applies img_transform.
	•	Returns:
	•	The image tensor
	•	The regression target (hb)
	•	The patient_id (for constructing episodes by patient).
The dataset is instantiated as:
df_train = df_images[df_images["split"] == "train"].reset_index(drop=True)
df_val   = df_images[df_images["split"] == "val"].reset_index(drop=True)
df_test  = df_images[df_images["split"] == "test"].reset_index(drop=True)

train_dataset = ConjunctivaImageDataset(df_train, transform=img_transform)
val_dataset   = ConjunctivaImageDataset(df_val,   transform=img_transform)
test_dataset  = ConjunctivaImageDataset(df_test,  transform=img_transform)
5.2. EpisodeSampler
class EpisodeSampler:
    """
    Sampler that builds patient-level few-shot episodes.
    """
    def _init_(self, df_split):
        self.df = df_split.reset_index(drop=True)

        self.patient_label_to_indices = defaultdict(list)
        self.label_to_patients        = defaultdict(set)

        for idx, row in self.df.iterrows():
            pid   = int(row["patient_id"])
            label = int(row["label"])
            self.patient_label_to_indices[(pid, label)].append(idx)
            self.label_to_patients[label].add(pid)
    ...
Core idea:
	•	Organizes indices by (patient_id, label).
	•	Creates episodes by:
	•	Sampling a small number of patients and images as the support set (K-shot).
	•	Sampling additional images from possibly the same or different patients as the query set.
	•	Optionally controlling how often support/query come from the same patient using p_same.
This sampler is used:
train_sampler = EpisodeSampler(df_train)
val_sampler   = EpisodeSampler(df_val)
test_sampler  = EpisodeSampler(df_test)

6. Few-Shot Model (Backbone for MAML)
6.1. Encoder & regressor
You’ll see:
def build_fewshot_encoder():
    ...
    # Returns a CNN producing feature maps [B, 256, 8, 8]
    return encoder

def build_fewshot_regressor():
    return nn.Linear(256, 1)
6.2. Full model: FewShotConvNet
class FewShotConvNet(nn.Module):
    """
    CNN backbone for few-shot hemoglobin regression.
    Input:  [B, 3, 128, 128]
    Output: [B] continuous Hb predictions
    """

    def _init_(self):
        super()._init_()
        self.encoder   = build_fewshot_encoder()
        self.regressor = build_fewshot_regressor()

    def forward(self, x):
        feats = self.encoder(x)          # [B, 256, 8, 8]
        feats = feats.mean(dim=[2, 3])   # Global Average Pool → [B, 256]
        out   = self.regressor(feats)    # [B, 1]
        return out.squeeze(1)            # [B]
So the model is:
	•	CNN encoder → global average pooling → linear regressor → scalar Hb.
MAML will adapt all these parameters in the inner loop and learn good initial parameters in the outer loop.

7. Inner-Loop Adaptation (First-Order MAML)
The core inner-loop function is:
def inner_adapt_params(
    model,
    base_params,
    x_support,
    y_support,
    inner_lr=1e-2,
    inner_steps=1,
):
    """
    Perform inner-loop adaptation using the SUPPORT set (first-order MAML).
    Args:
        model       : FewShotConvNet instance
        base_params : dict(name -> tensor) of meta-parameters
        x_support   : [N_s, 3, H, W]
        y_support   : [N_s]
    """
    fast_params = {name: p.clone().detach().requires_grad_(True)
                   for name, p in base_params.items()}

    for step in range(inner_steps):
        # Forward using fast parameters
        y_pred = functional_forward(model, x_support, fast_params)
        loss   = mse_loss(y_pred, y_support)

        grads = torch.autograd.grad(
            loss,
            list(fast_params.values()),
            create_graph=False,  # first-order MAML
        )

        fast_params = {
            name: param - inner_lr * g
            for (name, param), g in zip(fast_params.items(), grads)
        }

    return fast_params
Conceptually:
	1.	Start from the current meta-parameters (base_params).
	2.	Clone them into fast_params (task-specific parameters).
	3.	For a small number of inner steps:
	•	Compute predictions on the support set.
	•	Compute MSE loss against y_support.
	•	Compute gradients w.r.t. fast_params.
	•	Do gradient descent update: fast_params ← fast_params − inner_lr · grad.
	4.	Return adapted parameters for this specific episode.
Note: A helper functional_forward is used to run the model with arbitrary parameter dicts (typical in functional MAML implementations).

8. MAML Training Loop (train_maml)
8.1. Outer-loop function
def train_maml(
    model,
    optimizer,
    dataset,
    sampler,
    n_epochs=10,
    episodes_per_epoch=100,
    K=1,
    n_query_per_class=5,
    p_same=0.3,
    inner_lr=1e-2,
    inner_steps=1,
    device="cpu",
    # Validation configuration
    val_dataset=None,
    val_sampler=None,
    val_K=1,
    val_p_same=0.3,
    val_n_episodes=50,
):
    ...
High-level behavior:
	1.	For each epoch:
	•	Set model to train().
	•	For episodes_per_epoch episodes:
	1.	Use sampler to sample support and query indices (s_idx, q_idx) with K-shot, p_same, etc.
	2.	Build x_support, y_support, x_query, y_query from dataset.
	3.	Take a snapshot of current model parameters: base_params.
	4.	Run inner adaptation: fast_params = inner_adapt_params(model, base_params, x_support, y_support, ...).
	5.	Compute predictions on x_query using fast_params → query loss (mse_loss).
	6.	Accumulate the query losses across episodes as the meta-loss.
	•	After some number of episodes or at the end of the epoch, backprop through the accumulated meta-loss and update the meta-parameters (model.parameters()) using optimizer.step().
	2.	Metrics:
	•	For each epoch, track:
	•	Mean training loss
	•	Mean MAE on the query sets (per episode)
	•	Optionally track counts of "same_patient" vs "different_patient" episodes for debugging.
	3.	Validation & early stopping:
	•	After each epoch, optionally run episodic evaluation on the validation split (val_dataset, val_sampler) using a helper function (e.g. run_episodic_eval).
	•	Compute MAE / RMSE / R² and store the best meta-parameters (lowest validation MAE).
	•	At the end, return:
	•	A history dict of training / validation metrics.
	•	best_state_dict (to be loaded back into the model).
8.2. Training call
Later in the notebook:
device = "cuda" if torch.cuda.is_available() else "cpu"

model = FewShotConvNet().to(device)
meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

history, best_state_dict = train_maml(
    model=model,
    optimizer=meta_optimizer,
    dataset=train_dataset,
    sampler=train_sampler,
    n_epochs=5,
    episodes_per_epoch=100,
    K=1,
    n_query_per_class=5,
    p_same=0.3,
    inner_lr=1e-2,
    inner_steps=1,
    device=device,
    val_dataset=val_dataset,
    val_sampler=val_sampler,
    val_K=1,
    val_p_same=0.3,
    val_n_episodes=50,
)

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
You can tweak:
	•	n_epochs — total training epochs.
	•	episodes_per_epoch — how many episodes per epoch.
	•	K — shots per class / patient for the support set.
	•	inner_steps and inner_lr — how aggressively you adapt in the inner loop.
	•	p_same — probability that support & query come from the same patient (controls task difficulty).

9. Evaluation & Metrics
After training, the notebook runs episodic evaluation on:
	•	India validation set
	•	Italy test set
Each evaluation:
	•	Reuses EpisodeSampler with the corresponding split.
	•	Repeats episodic sampling for multiple seeds to get stable metrics.
	•	Uses inner_adapt_params to adapt to the support set for each episode.
	•	Computes:
	•	Regression metrics:
	•	MAE
	•	RMSE
	•	R²
	•	Bland–Altman statistics (mean difference, limits of agreement)
	•	Classification metrics (using Hb threshold, e.g. anemia vs non-anemia):
	•	Accuracy
	•	Sensitivity
	•	Specificity
Results are stored in dataframes and sometimes saved as:
	•	metrics_india_val_single_seed.xlsx / .csv
	•	metrics_italy_test_single_seed.xlsx / .csv
(Exact filenames depend on the final cells in the notebook.)

10. How to Run the Notebook (Step-by-Step)
	1.	Open the notebook Anemia_regression_V1-3.ipynb.
	2.	(If in Colab):
	•	Run the first cell to mount Google Drive: from google.colab import drive
	•	drive.mount('/content/drive')
	•	
	3.	If running locally, you can comment out or delete this cell.
	4.	Set your data path:
	•	Go to the “Set Dataset Paths” cell.
	•	Change: BASE_DIR  = Path("/content/drive/MyDrive/Anemia_dataset")
	•	 to your own path.
	•	Run this cell and confirm the printed checks:
	•	Base dir exists: True
	•	India dir exists: True
	•	Italy dir exists: True
	5.	Run metadata & preprocessing cells:
	•	Load India.xlsx and Italy.xlsx.
	•	Build R/G ratio data frames.
	•	Build patient-level and image-level tables.
	•	Run until df_images and df_train, df_val, df_test are created.
	6.	Run dataset & sampler cells:
	•	Instantiate ConjunctivaImageDataset for train/val/test.
	•	Instantiate EpisodeSampler for each split.
	7.	Run model & MAML cells:
	•	Model definition (FewShotConvNet).
	•	inner_adapt_params.
	•	train_maml.
	8.	Run the training cell (the one where train_maml is called).
	•	Monitor printed metrics per epoch.
	9.	Run evaluation cells for India VAL and Italy TEST episodic evaluation.

11. Common Issues & Tips
	•	Base dir exists: False → Your BASE_DIR is wrong. Double-check the absolute path.
	•	Excel file not found → Make sure India.xlsx and Italy.xlsx are in the folders pointed to by INDIA_DIR and ITALY_DIR.
	•	FileNotFoundError in ConjunctivaImageDataset → Some image_path from df_images does not exist on disk.
	•	Print a few df_images["image_path"].head() and manually check.
	•	Confirm your augmented data folder structure matches what build_image_table_from_augmented expects.
	•	Out-of-memory or slow training → Reduce:
	•	episodes_per_epoch
	•	n_epochs
	•	inner_steps
	•	or reduce image size in img_transform.
	•	Want to use your own dataset
	•	Ensure you have:
	•	Equivalent metadata columns: patient ID, Hb, label.
	•	A similar directory structure, or update:
	•	build_rg_df
	•	build_image_table_from_augmented
	•	Point BASE_DIR to your dataset root, and adjust folder names if needed.
