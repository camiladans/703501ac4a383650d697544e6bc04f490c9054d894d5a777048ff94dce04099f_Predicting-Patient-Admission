from data_preprocessing import load_and_preprocess_data

def main():
    print("Hello from 703501ac4a383650d697544e6bc04f490c9054d894d5a777048ff94dce04099f-predicting-patient-admission!")

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/data-ori.xlsx")

    print("X_train shape:", X_train.shape)
    print("y_train value counts:\n", y_train.value_counts())

if __name__ == "__main__":
    main()
