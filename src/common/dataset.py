class Dataset:
    def __init__(self, data_file: str,
                 dataset_type: str, dim: int,
                 name: str = None):
        self.data_file = data_file

        if name is not None:
            self.name = name
        else:
            # Если имя датасета явно не задано, то это название файла без расширения
            self.name = self.data_file.split('.')[0]

        self.dataset_type = dataset_type
        self.dim = dim
