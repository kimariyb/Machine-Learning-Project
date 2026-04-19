import pandas as pd


def data_processing(file_path: str) -> pd.DataFrame:

    df = pd.read_csv(file_path)

    # format time style
    # 2015/7/31 0:00 -> 2015-07-31 00:00:00
    df['time'] = pd.to_datetime(df['time']).dt.strftime("%Y-%m-%d %H:%M:%S")

    # 按照时间升序
    df.sort_values(by=['time'], inplace=True)

    # 去重
    df.drop_duplicates(inplace=True)

    return df


if __name__ == "__main__":
    df = data_processing("./data/train.csv")
    print(df)