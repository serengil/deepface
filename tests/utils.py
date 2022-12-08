from deepface.commons import utils

print("-----------------------------------------")
print("Check for invalid file name extensions")

filename_with_unsupported_ext = "this_file_ext_isnot_supported.wav"
file_ext_check_result = utils.is_file_ext_supported(filename_with_unsupported_ext) 
assert(file_ext_check_result == False)
print("filename: '{}'".format(filename_with_unsupported_ext))
print("is valid: {}".format(file_ext_check_result))

print("-----------------------------------------")
print("Check filenames that include a valid file ext, as in commons/utils.py#DEFAULT_SUPPORTED_FILE_EXTS")

filenames_with_supported_ext = [
    "some-imaginary-filename.png",
    "some-imaginary-filename.jpeg",
    "some-imaginary-filename.jpg",
    "some-imaginary-filename.webp",
]

for filename in filenames_with_supported_ext:
    is_valid_file_ext = utils.is_file_ext_supported(filename)
    assert(is_valid_file_ext == True)
    print("filename '{}' is valid: ".format(filename), is_valid_file_ext)

