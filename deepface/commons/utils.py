
DEFAULT_SUPPORTED_FILE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]

def get_file_ext(file: str):
    return ".{}".format(file.split(".")[-1]).lower()

def is_file_ext_supported(file: str):
    file_ext = get_file_ext(file)
    
    if not file_ext:
        return False

    is_supported = False
    for ext in DEFAULT_SUPPORTED_FILE_EXTS:
        if ext == file_ext:
            is_supported = True
            break

    return is_supported

