import IACDataLoader as iac

def main():
    dataloader = iac.IACDataLoader()
    dataloader.set_dataset_dir("../../dataset/discussions")
    dataloader.set_topic_filepath("../../dataset/topic.csv")
    dataloader.set_stance_filepath("../../dataset/author_stance.csv")
    dataloader.load()

if __name__ == "__main__":
    main()
