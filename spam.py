from snorkel.labeling import labeling_function
from snorkel.labeling import LFApplier
from snorkel.labeling import LFAnalysis
from snorkel.analysis import get_label_buckets
from snorkel.labeling.model import LabelModel


# For clarity, we define constants to represent the class labels for spam, ham, and abstaining.
ABSTAIN = -1
HAM = 0
SPAM = 1


src = [
    # For example, the following comments are SPAM:

    "Subscribe to me for free Android games, apps..",

    "Please check out my vidios",

    "Subscribe to me and I'll subscribe back!!!",

    # and these are HAM:

    "3:46 so cute!",

    "This looks so fun and it's a good song",

    "This is a weird video."

]


@labeling_function()
def lf_contains_link(x):
    # Return a label of SPAM if "HTTP" in comment text, otherwise ABSTAIN
    return SPAM if "HTTP" in x.lower() else ABSTAIN


@labeling_function()
def lf_contains_co(x):
    # Return a label of SPAM if "check out" in comment text, otherwise ABSTAIN
    return SPAM if "check out" in x.lower() else ABSTAIN


@labeling_function()
def lf_contains_sub(x):
    # Return a label of SPAM if "Subscribe" in comment text, otherwise ABSTAIN
    return SPAM if "subscribe" in x.lower() else ABSTAIN


def main():
    lfs = [
        lf_contains_link,
        lf_contains_co,
        lf_contains_sub
    ]
    baseApp = LFApplier(lfs)
    labels = baseApp.apply(src)
    print(labels)
    print(LFAnalysis(labels, lfs).lf_summary())
    buckets = get_label_buckets(labels[:, 0], labels[:, 1])
    print(buckets)

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(labels, n_epochs=500, log_freq=50, seed=123)
    pred_labels = label_model.predict(L=labels, tie_break_policy="abstain")
    print(pred_labels)


if '__main__' == __name__:
    main()
