
def load_stsjobtitle(dataset_filename):
    """Loads the STSJobTitle dataset"""
    sts = pd.read_csv(dataset_filename, sep="\t") 
    sentences_1 = sts.job1.values
    sentences_2 = sts.job2.values
    dev_scores = sts.score.values
    return (sentences_1, sentences_2, dev_scores)

