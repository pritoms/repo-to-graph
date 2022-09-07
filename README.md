# Repo_to_Graph

This project is about predicting the contents of a file given its path inside a github repository. The project has been implemented in two phases, both phases are focusing on predicting the contents of a file. In phase one we are going to predict the contents of a file and where that file resides among the repository. On the other hand in phase two we are going to predict only the contents of the file.

## Tools
- Python 3
- Git
- Bash
- Github
- Jupyter Notebook

## Installation

The following instructions will help you use this module in your own code:
1. Clone this repository using `git clone https://github.com/chirag64/Repo_to_Graph.git`.
2. Create a new directory e.g `my_code`
3. Copy the file `src/__init__.py` into `my_code`
4. Copy the directory `src` into `my_code`
5. From inside `my_code` run the following command `python3 src/__init__.py --url <github repo url> --search <optional> --extensions <optional>`
6. You can also pass arguments directly from the code i.e `args = ['--url', 'https://github.com/chirag64/CS301-Data-Mining', '--search', '/src/', '--extensions', '.java', '.cpp']`
7. This will create a directory called `data` inside which there are two directories `raw` and `processed`. The `raw` directory contains all the files from the github repository including sub-directories. The `processed` directory contains the pre-processed data ready for training and testing.
8. Currently we have not trained any model but we have created the dataset. We have used **Grokking Deep Learning** book by Andrew Trask to implement our model and train it.
9. The current dataset we have created is not ready yet because all the files under the `raw` directory are not processed. This means that the files under the `processed` directory are empty.
10. In order to process the files under the `raw` directory, please execute the following command `python3 src/process_files.py`
11. After executing the above command, the files under the `processed` directory are now  pre-processed ready for training and testing.
12. After training and testing the model, we expect it to predict the contents of a file given its path.
13. For more information about each module, please refer to the documentation section below.

### Documentation

For now we have implemented the following modules:
```
1. clone_repo.py
2. repo_to_Graph.py
3. parse_repo_to_Graph.py
4. process_files.py
5. extract_files.py
6. parse_args.py
7. get_code.py
8. src/
    - __init__.py
    - parse_repo_to_Graph.py
    - process_files.py
    - extract_files.py
    - parse_args.py
    - get_code.py
```

### Full Implementation

- `src/__init__.py`

```python
#!/usr/bin/env python3

import argparse
import os
import sys
from src.parse_repo_to_Graph import create_nodes_edges
from src.extract_files import extract_files, create_dataset
from src.process_files import process_files, read_files
from src.get_code import get_code

def main(args):
    # Parse arguments
    args = parse_arguments(args)
    if args is None:
        return

    # Parse the repository into a Graph
    nodes, edges = create_nodes_edges(args['url'])

    # Extract files from repository
    extract_files(nodes, edges, args['search'], args['extensions'])

    # Process all files under data/raw and save them under data/processed
    for file in os.listdir('../data/raw'):
        label, counter = process_files('../data/raw/{}'.format(file))
        print(label, counter)

    # Get code from data/raw
    get_code('../data/raw', ['java'])

    # Read processed files
    counters = read_files()
    for counter in counters:
        print(counter[0], counter[1])

    return

def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Parse arguments')
    parser.add_argument('--url', dest='url', help='Github URL')
    parser.add_argument('--search', dest='search', default=None, help='Search path')
    parser.add_argument('--extensions', dest='extensions', nargs='*', default=None, help='File extensions')
    parsed_args = vars(parser.parse_args(args))

    url = parsed_args['url']
    search = parsed_args['search']
    extensions = parsed_args['extensions']

    if url is None:
        print("Missing argument '--url'")
        return None

    if search is not None:
        if len(search) == 0:
            search = None

    if extensions is not None:
        if len(extensions) == 0:
            extensions = None

    args = {'url': url, 'search': search, 'extensions': extensions}

    return args

if __name__ == "__main__":
    main(sys.argv[1:])
```

- `src/parse_repo_to_Graph.py`

```python
#!/usr/bin/env python3

import os
import shutil
import git

def create_nodes_edges(url):
    # Create a temporary directory and clone the github repo inside it
    temp_dir = os.path.join(os.path.expanduser('~'), 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    repo = git.Repo.clone_from(url=url, to_path=temp_dir)

    # Walk through the cloned directory tree and create a graph representing the structure of the repository
    root = '/'
    nodes = [root]
    edges = []
    directories = [root]
    i = 0

    while len(directories) > 0:
        directory = directories.pop()
        for dirpath, dirnames, filenames in os.walk(os.path.join(temp_dir, directory), topdown=True):
            for filename in filenames:
                label = "file" + str(i)
                nodes.append((label, filename, os.path.join(url, dirpath.replace(temp_dir, '')[1:], filename)))
                edges.append((directory, label))
                i += 1
            for dirname in dirnames:
                label = "file" + str(i)
                nodes.append((label, dirname, os.path.join(url, dirpath.replace(temp_dir, '')[1:], dirname)))
                edges.append((directory, label))
                directories.append(label)
                i += 1

    # Remove the temporary directory
    shutil.rmtree(temp_dir)

    return nodes, edges
```

- `src/process_files.py`

```python
#!/usr/bin/env python3

import os
import re
from collections import Counter

def process_files(file):
    label = os.path.basename(file)
    with open(file, 'r') as f:
        content = f.read()

    # Convert text to lowercase
    content = content.lower()

    # Remove punctuation
    content = re.sub(r"[^a-zA-Z0-9]", " ", content)

    # Split the string into words
    words = content.split()

    # Count the frequency of each word
    counter = Counter(words)

    # Save processed files under data/processed
    label = label.split('.')[0]
    new_file = '../data/processed/{}'.format(label)
    with open(new_file, 'w') as f:
        f.write(content)

    return label, counter

def read_files():
    counters = []
    for filename in os.listdir('../data/processed'):
        label = os.path.basename(filename)
        with open('../data/processed/{}'.format(filename), 'r') as f:
            content = f.read()

        # Split the string into words
        words = content.split()

        # Count the frequency of each word
        counter = Counter(words)

        counters.append((label, counter))

    return counters
```

- `src/extract_files.py`

```python
#!/usr/bin/env python3

import os
from collections import Counter
import requests

def extract_files(nodes, edges, search=None, extensions=None):
    if search is None:
        search = '/'

    if extensions is None:
        extensions = ['.java', '.cpp', '.py', '.rb', '.js', '.scala', '.c']

    for edge in edges:
        for node in nodes:
            if node[0] == edge[1]:
                target = node[1]
                url = node[2]
                break

        if target == search:
            if target == '/':
                continue
            if target.endswith(tuple(extensions)):
                with open('../data/raw/{}'.format(target), 'w') as f:
                    response = requests.get(url)
                    f.write(response.text)
                continue
            else:
                continue
        else:
            continue

def create_dataset(url, nodes, edges, search=None, extensions=None):
    if search is None:
        search = '/'

    if extensions is None:
        extensions = ['.java', '.cpp', '.py', '.rb', '.js', '.scala', '.c']

    for edge in edges:
        for node in nodes:
            if node[0] == edge[1]:
                target = node[1]
                url = node[2]
                break
        if target == search:
            if target == search:
                continue
            if target.endswith(tuple(extensions)):
                with open('../data/raw/{}'.format(target), 'w') as f:
                    response = requests.get(url)
                    f.write(response.text)
                continue
            else:
                continue
        else:
            continue
```

- `src/parse_args.py`

```python
#!/usr/bin/env python3

import argparse

def parse_arguments(args):
    parser = argparse.ArgumentParser(description='Parse arguments')
    parser.add_argument('--url', dest='url', help='Github URL')
    parser.add_argument('--search', dest='search', default=None, help='Search path')
    parser.add_argument('--extensions', dest='extensions', nargs='*', default=None, help='File extensions')
    parsed_args = vars(parser.parse_args(args))

    url = parsed_args['url']
    search = parsed_args['search']
    extensions = parsed_args['extensions']

    if url is None:
        print("Missing argument '--url'")
        return None

    if search is not None:
        if len(search) == 0:
            search = None

    if extensions is not None:
        if len(extensions) == 0:
            extensions = None

    args = {'url': url, 'search': search, 'extensions': extensions}

    return args
```

- `src/get_code.py`

```python
#!/usr/bin/env python3

import os
from collections import Counter
import requests


def get_code(path, extensions):
    code_dir = os.path.join(path, 'code')
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)

    for filename in os.listdir(path):
        if filename == 'code':
            continue
        file_path = os.path.join(path, filename)
        if os.path.isdir(file_path):
            get_code(file_path, extensions)
        else:
            if filename.endswith(tuple(extensions)):
                shutil.copy(file_path, os.path.join(code_dir, filename))
    return
```
