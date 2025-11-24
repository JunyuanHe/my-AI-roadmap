### Q1

In small GPT project, I first trained on tinyStories and than switched to simplebooks. I first forgot to apply the pre-tokenizer in my training of BPE-tokenizer and didn't add the 0-255 byte values as the initial vocab as well (this is not necessary in fact). This certainly resulted in BPE merges across word boundaries. 
Surprisingly, the loss goes down more smoothly without the pre-tokenizer than with the pretokenizer and converges more quickly as well. I don't know why.

