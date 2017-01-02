import pandas as pd

from basedir import data


def main():
    fer_data = pd.read_csv(data('fer2013', 'fer2013.csv'))
    print(fer_data.head(3))
    print(fer_data.Usage.unique())


if __name__ == '__main__':
    main()
