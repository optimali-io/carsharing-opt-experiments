import os
import config


def clean_temp_files(work_dir_path: str) -> None:
    """
    Function removes potential cells and roads points files from given directory.
    :param work_dir_path: path to working directory
    :return: None
    """
    for f in os.listdir(work_dir_path):
        if "potential" in f or "roads_points" in f:
            os.remove(os.path.join(work_dir_path, f))


if __name__ == "__main__":
    clean_temp_files(work_dir_path=config.ZONE_FILES_DIRECTORY)
