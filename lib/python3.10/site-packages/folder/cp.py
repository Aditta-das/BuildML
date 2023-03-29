from .project import ProjectFolder

def createproject(name, customize):
    a = ProjectFolder(name)
    b = a.project_directory()
    if customize == True:
        a.customize_subdirectory()
    else:
        a.default_folder()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("startproject", type=str, help="Your Project Name") # positional
    parser.add_argument("--customize", help="If you want to customize your folder", action="store_true") # optional
    args = parser.parse_args()
    createproject(name=args.startproject, customize=args.customize)