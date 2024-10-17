# ===----------------------------------------------------------------------===#
#
# Copyright the CYCLE LAB.
# All rights reserved.
#
# ===----------------------------------------------------------------------===#
import os

# Define the copyright notice for C++ and C headers
cpp_c_header_notice = """//===----------------------------------------------------------------------===//
//
// Copyright the CYCLE LAB.
// All rights reserved.
//
//===----------------------------------------------------------------------===//
"""

# Define the copyright notice for Python files
python_notice = """# ===----------------------------------------------------------------------===#
#
# Copyright the CYCLE LAB.
# All rights reserved.
#
# ===----------------------------------------------------------------------===#
"""


def has_copyright(content):
    # Check the first 5 lines for the word "Copyright"
    for line in content.splitlines()[:5]:  # Only look at the first 5 lines
        if "Copyright" in line:
            return True
    return False


def add_copyright_notice(file_path):
    with open(file_path, "r+") as f:
        content = f.read()

        # Determine the appropriate copyright notice based on file extension
        if file_path.endswith((".cpp", ".h")):  # For C++ and C header files
            copyright_notice = cpp_c_header_notice
        elif file_path.endswith(".py"):  # For Python files
            copyright_notice = python_notice
        else:
            return  # Skip files with other extensions

        # Check if copyright notice is already present in the first 5 lines
        if not has_copyright(content):
            f.seek(0, 0)  # Move to the beginning of the file
            f.write(copyright_notice + content)  # Add the copyright notice


def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith((".cpp", ".h", ".py")):  # Add other extensions as needed
                add_copyright_notice(os.path.join(root, file))


# Automatically use the current working directory
current_directory = os.getcwd()
process_directory(current_directory)
