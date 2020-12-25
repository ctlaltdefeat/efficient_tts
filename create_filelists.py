import csv

with open("datasets/LJSpeech-1.1/metadata.csv", encoding="utf8") as f:
    reader = csv.reader(f, delimiter="|", quotechar=None)
    data = list(reader)

data = [
    "datasets/LJSpeech-1.1/wavs/{}.wav|{}\n".format(d[0], d[1]) for d in data
]

train, val = (
    open("datasets/LJSpeech-1.1/train.txt", "w", encoding="utf8"),
    open("datasets/LJSpeech-1.1/val.txt", "w", encoding="utf8"),
)
train.writelines(data[:-200])
val.writelines(data[-200:])