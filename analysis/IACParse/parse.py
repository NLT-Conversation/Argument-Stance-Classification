import csv

def main():
    sentence_dict = dict()
    with open('qr_averages.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for idx, row in enumerate(spamreader):
            if idx == 0:
                continue
            key, discussion_id, agree_disagree, agreement, agreement_unsure, \
            attack, attack_unsure, defeater_undercutter, \
            defeater_undercutter_unsure, fact_feeling, fact_feeling_unsure, \
            negotiate_attack, negotiate_attack_unsure, nicenasty, \
            nicenasty_unsure, personal_audience, personal_audience_unsure, \
            questioning_asserting, questioning_asserting_unsure, \
            sarcasm, sarcasm_unsure = row
            sentence_dict[key] = dict()
            sentence_dict[key]["agreement"] = agreement
            sentence_dict[key]["attack"] = attack
            sentence_dict[key]["fact_feeling"] = fact_feeling
            sentence_dict[key]["nicenasty"] = nicenasty

    annotator_dict = dict()
    with open('qr_worker_answers_task1.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for idx, row in enumerate(spamreader):
            if idx == 0:
                continue
            key, discussion_id, worker_id, agreement, agreement_unsure, \
            attack, attack_unsure, fact_feeling, fact_feeling_unsure, \
            nicenasty, nicenasty_unsure, sarcasm = row

            if "num_annotators" not in sentence_dict[key]:
                sentence_dict[key]["num_annotators"] = 1
            else:
                sentence_dict[key]["num_annotators"] += 1

    with open('qr_worker_answers_task1.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for idx, row in enumerate(spamreader):
            if idx == 0:
                continue
            key, discussion_id, worker_id, agreement, agreement_unsure, \
            attack, attack_unsure, fact_feeling, fact_feeling_unsure, \
            nicenasty, nicenasty_unsure, sarcasm = row

            num_annotators = sentence_dict[key]["num_annotators"]
            if num_annotators > 1:
                minority_agreement = (((float(sentence_dict[key]["agreement"]) * num_annotators - float(agreement))/(num_annotators-1)) * float(agreement) < 0)
                minority_attack = (((float(sentence_dict[key]["attack"]) * num_annotators - float(attack))/(num_annotators-1)) * float(attack) < 0)
                minority_fact_feeling = (((float(sentence_dict[key]["fact_feeling"]) * num_annotators - float(fact_feeling))/(num_annotators-1)) * float(fact_feeling) < 0)
                minority_nicenasty = (((float(sentence_dict[key]["nicenasty"]) * num_annotators - float(nicenasty))/(num_annotators-1)) * float(nicenasty) < 0)
                if worker_id in annotator_dict:
                    annotator_dict[worker_id]["minority_agreement"] += 1 if minority_agreement else 0
                    annotator_dict[worker_id]["minority_attack"] += 1 if minority_attack else 0
                    annotator_dict[worker_id]["minority_fact_feeling"] += 1 if minority_fact_feeling else 0
                    annotator_dict[worker_id]["minority_nicenasty"] += 1 if minority_nicenasty else 0
                else:
                    annotator_dict[worker_id] = dict()
                    annotator_dict[worker_id]["minority_agreement"] = 1 if minority_agreement else 0
                    annotator_dict[worker_id]["minority_attack"] = 1 if minority_attack else 0
                    annotator_dict[worker_id]["minority_fact_feeling"] = 1 if minority_fact_feeling else 0
                    annotator_dict[worker_id]["minority_nicenasty"] = 1 if minority_nicenasty else 0

            if "num_annotated" not in annotator_dict[worker_id]:
                annotator_dict[worker_id]["num_annotated"] = 1
            else:
                annotator_dict[worker_id]["num_annotated"] += 1

    with open('test.txt', 'wb') as output:
        output.write("annotator_id\tnum_annotated\tminority_agreement\tminority_attack\tminority_fact_feeling\tminority_nicenasty\n")
        keys = annotator_dict.keys()
        for worker_id in keys:
            output.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    worker_id,
                    annotator_dict[worker_id]["num_annotated"],
                    annotator_dict[worker_id]["minority_agreement"],
                    annotator_dict[worker_id]["minority_attack"],
                    annotator_dict[worker_id]["minority_fact_feeling"],
                    annotator_dict[worker_id]["minority_nicenasty"],
                )
            )


if __name__ == "__main__":
    main()
