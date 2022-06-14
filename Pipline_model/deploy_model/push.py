import base64
from github import Github
from github import InputGitTreeElement

def push_model_to_github(model_path):
    token = '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'
    g = Github(token)

    repo = g.get_user().get_repo('KLTN')
    all_files = []
    contents = repo.get_contents("")
    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            file = file_content
            all_files.append(str(file).replace('ContentFile(path="','').replace('")',''))

    with open(model_path, 'rb') as file:
        content = file.read()

    # Upload to github
    git_prefix = 'Model/'
    git_file = git_prefix + 'model.h5'
    if git_file in all_files:
        contents = repo.get_contents(git_file)
        repo.update_file(contents.path, "committing files", content, contents.sha, branch="main")
        print(git_file + ' UPDATED')
    else:
        repo.create_file(git_file, "committing files", content, branch="main")
        print(git_file + ' CREATED')