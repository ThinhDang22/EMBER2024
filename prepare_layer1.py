import multiprocessing as mp

def main():
    import thrember

    data_dir = r"E:\project2\EMBER2024\data_pe"

    print("Doc train...")
    X_train, y_train = thrember.read_vectorized_features(data_dir, subset="train")
    print(X_train.shape, y_train.shape)

    print("Doc test...")
    X_test, y_test = thrember.read_vectorized_features(data_dir, subset="test")
    print(X_test.shape, y_test.shape)

    print("Doc challenge...")
    X_challenge, y_challenge = thrember.read_vectorized_features(data_dir, subset="challenge")
    print(X_challenge.shape, y_challenge.shape)

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()