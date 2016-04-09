
def parse_tweets(filename, task):
    data = {}

    lexelt = 'tweets'
    data[lexelt] = {}

    for line in open(filename):
        fields = line.split()
        (twid1, twid2) = (fields[0], fields[1])
        if task == 'A':
            start_token = int(fields[2])
            end_token = int(fields[3])
            sense = fields[4]
            words = fields[5:]
            heads = list(range(start_token, end_token+1))
            instance = '%s_%s_%d_%d' % (twid1, twid2, start_token, end_token)

        else:
            sense = fields[2]
            words = fields[3:]
            heads = []
            instance = '%s_%s' % (twid1, twid2)

        
        if sense == 'objective-OR-neutral':
            senses = ['objective', 'neutral']
        else:
            senses = [sense]
        
        data[lexelt][instance] = dict()
        data[lexelt][instance]['words'] = words
        data[lexelt][instance]['answers'] = senses
        data[lexelt][instance]['heads'] = heads

    return data

if __name__=='__main__':
    filename = '/data/cs65/semeval-2015/B/train/twitter-train-cleansed-B.tsv'
    tweetData = parse_tweets(filename, 'B')
