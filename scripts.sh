
export USER=$(whoami)

function preprocess {
    python3 preprocess.py /data/cs65/semeval-2015/B/train/twitter-train-full-B.tsv /scratch/$USER/twitter-train-full-B-preprocessed.tsv
}

function classify {
    python3 classifier.py /scratch/$USER/twitter-train-full-B-preprocessed.tsv 5
}

function preprocess_dev {
    python3 preprocess.py /data/cs65/semeval-2015/B/train/twitter-train-full-B.tsv /scratch/$USER/twitter-train-full-B-preprocessed.tsv
    python3 preprocess.py /data/cs65/semeval-2015/B/dev/twitter-dev-cleansed-B.tsv /scratch/$USER/twitter-dev-cleansed-B-preprocessed.tsv
}

function classify_dev {
    python3 classifier.py /scratch/$USER/twitter-train-full-B-preprocessed.tsv /scratch/$USER/twitter-dev-cleansed-B-preprocessed.tsv
}

function preprocess_test {
    python3 preprocess.py /data/cs65/semeval-2015/B/train/twitter-train-full-B.tsv /scratch/$USER/twitter-train-full-B-preprocessed.tsv
    python3 preprocess.py /data/cs65/semeval-2015/B/test/twitter-test-B.tsv /scratch/$USER/twitter-test-B-preprocessed.tsv
}

function classify_test {
    python3 classifier.py /scratch/$USER/twitter-train-full-B-preprocessed.tsv /scratch/$USER/twitter-test-B-preprocessed.tsv
 }

