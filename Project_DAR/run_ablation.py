# run_ablation.py

import argparse

from models.pipelines.yolo_sort_lstm import inference_and_save_csv as run_yolo_sort_lstm
from models.pipelines.yolo_sort_lstm_PTQ import inference_and_save_csv as run_yolo_sort_lstm_PTQ 


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=[
            "yolo_sort_lstm",
            "yolo_sort_lstm_PTQ",
            "all"
        ]
    )

    args = parser.parse_args()

    if args.pipeline == "yolo_sort_lstm":

        print("Running yolo_sort_lstm pipeline")
        run_yolo_sort_lstm()

    elif args.pipeline == "yolo_sort_lstm_PTQ":

        print("Running yolo_sort_lstm_PTQ pipeline")
        run_yolo_sort_lstm_PTQ()

    elif args.pipeline == "all":

        print("Running ALL pipelines")

        run_yolo_sort_lstm()

        run_yolo_sort_lstm_PTQ()


if __name__ == "__main__":

    main()
