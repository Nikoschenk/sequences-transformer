# sequences-transformer
Sequence Relation Classification with Transformers.

## Installation
As prerequisite, you need installations of Python 3.6+, PyTorch 1.0.0+ and TensorFlow 2.0.0-rc1.

Clone the repository; ideally in a [Python virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). All dependencies can be installed via:

```bash
pip install -r requirements.txt
```

## Fine-tuning BERT for Sequence Relation Classification

### Data
The [data](https://github.com/Nikoschenk/sequences-transformer/tree/master/data) is split into training, test, and development sets.


Run the following command to fine-tune a BERT<sub>BASE</sub> model on a sequence classification task.

```
python sequences-trainer.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --task_name seq-classification \
  --do_train --do_eval \
  --data_dir data/train-test-dev/  \
  --max_seq_length 20 --per_gpu_train_batch_size 4 \
  --learning_rate 2e-5 --num_train_epochs 20.0 \
  --output_dir gens/ \
  --eval_all_checkpoints \
  --overwrite_output_dir \
  --tokenizer_name bert-base-uncased \
  --do_lower_case
```

## License & Credits
This project is licensed under the APACHE LICENSE, VERSION 2.0 [LICENSE.md](https://github.com/Nikoschenk/sequences-transformer/blob/master/LICENSE) file for details.

The code is based on the original implementations provided by [huggingface transformers](https://github.com/huggingface/transformers)
