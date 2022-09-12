training_args_config = {
	"overwrite_output_dir": True,
	"do_train": True,
	"do_eval": True,
	"do_predict": True,
	"evaluation_strategy": "steps",
	"per_device_train_batch_size": 32,
	"per_device_eval_batch_size": 32,
	"learning_rate": 1e-3,
	"logging_steps": 500,
	"num_train_epochs": 5,
	# "warmup_steps": 50,
	"logging_dir": "",
}