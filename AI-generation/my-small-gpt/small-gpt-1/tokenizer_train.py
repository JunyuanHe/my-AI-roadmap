from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset



# Load dataset
ds = load_dataset("roneneldan/TinyStories", split="train[5%:15%]")
# text = "\n".join(ds["text"])

def batch_iterator(batch_size=1000):
    # Only keep the text column to avoid decoding the rest of the columns unnecessarily
    tok_dataset = ds.select_columns("text")
    for batch in tok_dataset.iter(batch_size):
        yield batch["text"]

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=1000,
    min_frequency=2,
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
)

tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(ds))

# files = ["data/corpus.txt"]
# tokenizer.train(files, trainer)
tokenizer_pth = "AI-generation/my-small-gpt/small-gpt-1/tokenizer/my-bpe-tokenizer.json"
tokenizer.save(tokenizer_pth)
