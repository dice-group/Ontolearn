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
