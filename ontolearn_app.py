# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2024 Ontolearn Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

"""
@TODO:CD: we should introduce ontolearn keyword to learn OWL Class expression from the command line.
"""
from ontolearn.model_adapter import execute
from main import get_default_arguments

# pyinstaller --onefile --collect-all owlready2 --exclude-module gradio --copy-metadata rich --recursive-copy-metadata transformers ontolearn_app.py


def exe(cmd):
    print("\n---------------result---------------\n")
    args_list = cmd.split()
    execute(get_default_arguments(args_list))
    print("\n------------------------------------\n")


command = input("\nEnter the arguments. E.g: --model celoe --knowledge_base_path some/path/to/kb \n\n"
                "arguments: ")
exe(command)

while True:
    command = input("\nEnter arguments again? [Y/n]\n")
    if command is "Y":
        command = input("\narguments: ")
        exe(command)
    elif command is "n":
        print("\nterminating...")
        break
    else:
        print("\nInvalid input. Please type 'Y' or 'n'\n")
