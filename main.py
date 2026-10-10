"""One command pipeline:  python main.py"""
from src import config
from src.data_generator import generate_dataset
from src.preprocessing import load_data, clean_data
from src.features import add_features
from src.train import split_by_machine, train_models
from src.evaluate import print_report
from src.predict import add_predictions, build_alert_log
from src import visualize as viz


def main():
    print("=== 1. Generate virtual IoT data ===")
    config.DATA_PATH.parent.mkdir(exist_ok=True)
    if not config.DATA_PATH.exists():
        generate_dataset().to_csv(config.DATA_PATH, index=False)
    raw = load_data()
    print(raw.head(), "\nShape:", raw.shape)

    print("\n=== 2. Clean data ===")
    df = clean_data(raw)

    print("\n=== 3. Feature engineering ===")
    df = add_features(df)
    print("Total columns:", df.shape[1])
    viz.plot_class_balance(df); viz.plot_correlation(df)

    print("\n=== 4. Train models ===")
    train_df, test_df = split_by_machine(df)
    print(f"Train machines: {train_df.machine_id.nunique()}, Test machines: {test_df.machine_id.nunique()}")
    model, best, feats, results = train_models(train_df, test_df)

    print("\n=== 5. Evaluate ===")
    test_df = add_predictions(test_df, model, feats)
    y_pred = (test_df["failure_prob"] >= config.WARNING_THRESHOLD).astype(int)
    print_report(test_df["failure"], y_pred)

    print("=== 6. Alerts & visualisation ===")
    alerts = build_alert_log(test_df)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    alerts.to_csv(config.OUTPUT_DIR / "alert_log.csv", index=False)
    test_df.to_csv(config.OUTPUT_DIR / "predictions.csv", index=False)
    print(alerts.to_string(index=False))

    demo = int(test_df.machine_id.iloc[0])
    viz.plot_sensor_trends(test_df, demo)
    viz.plot_confusion(test_df["failure"], y_pred)
    viz.plot_roc(test_df["failure"], test_df["failure_prob"])
    viz.plot_feature_importance(model, feats)
    viz.plot_failure_timeline(test_df, demo)
    viz.plot_model_comparison(results)
    print(f"\nDone. Graphs in {config.IMAGE_DIR}, metrics in {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
