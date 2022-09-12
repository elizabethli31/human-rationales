TRAIN_FLAG = True
CACHING_FLAG = True

dataset = {
    "name": "BoolQ",
    "train_path": "../boolQ/train_data.csv",
    "test_path": "../boolQ/test_data.csv",
    "classes": [True, False],
    "batch_size": 32,
	"max_rows": None,
	"max_len": 512,
}