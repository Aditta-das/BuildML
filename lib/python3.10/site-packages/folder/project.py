import os, sys, shutil
import argparse

"""
ML Model
Project(editable)
    input
    models
    explore_data
    src
    LICENCE
    README.md
"""


class ProjectFolder(object):
    def __init__(self, name):
        self.name = name
        self.directory = os.getcwd()

    def project_directory(self):
        """
        Check If the directory already exists
        """
        if not os.path.exists(os.path.join(self.directory, f"{self.name}")):
            self.make_directory()
        else: 
            print("Choose Different Name. Folder Already Exists, Are you want to delete previous one? \nPress Y to YES: Y \nPress N to NO: N")
            self.pressButton()
            sys.exit()

    def make_directory(self):
        main_dir = os.mkdir(os.path.join(self.directory, f"{self.name}"))
        """
        customize - subdirectory folder no - example - 4 means 4 subfolder will create - take input folder name
        default - input, models, notebook, src, LICENCE, README.md
        """
        main_dir = os.path.join(f"{self.directory}", f"{self.name}")

        return main_dir

    def tree(self):
        """
        folder alignment
        """
        direction = os.path.join(self.directory, f"{self.name}")
        paths = [dirpath for dirpath, _, _ in os.walk(os.path.join(direction))]
        return paths, direction

    def customize_subdirectory(self):
        """
        subdirectory customize. If customize=True, otherwise it will default align
        """
        num_folders = int(input(f"Number of total subfolder: "))
        filename = os.path.join(self.directory, f"{self.name}/LICENCE")
        readme = os.path.join(self.directory, f"{self.name}/README.md")
        with open(f"{filename}", "w") as f:
            f.write("Licence")
        with open(f"{readme}", "w") as f:
            f.write("# README.md")
        for i in range(num_folders):
            self.custom = input(f"folder_{i}: ")
            os.mkdir(os.path.join(self.directory, f"{self.name}/{self.custom}"))
        print("Ready to Customize")
        paths, _ = self.tree()
        print(paths)
    
    def default_folder(self):
        """
        default folder creation.
        """
        os.mkdir(os.path.join(self.directory, f"{self.name}/input"))
        os.mkdir(os.path.join(self.directory, f"{self.name}/models"))
        os.mkdir(os.path.join(self.directory, f"{self.name}/notebooks"))
        os.mkdir(os.path.join(self.directory, f"{self.name}/src"))
        filename = os.path.join(self.directory, f"{self.name}/LICENCE")
        readme = os.path.join(self.directory, f"{self.name}/README.md")
        with open(f"{filename}", "w") as f:
            f.write("Licence")
        with open(f"{readme}", "w") as f:
            f.write("# README.md")
        print(f"default folders")
        paths, _ = self.tree()
        print(paths)
    
    def delete_folder(self):
        _, direction = self.tree()
        shutil.rmtree(f'{direction}')

    def pressButton(self):
        """
        Delete your folder if exists and if you want same name...
        """
        if input("Press: ") == 'Y':
            self.delete_folder()
            print("Delete")

        else: print("Choose Other Name")

    def licence(self):
        """
        Coming up next
        """
        pass



    