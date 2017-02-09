"""Helper script to prepare make Docker container ready to work with Jupyter
notebook.
"""
from subprocess import check_call, CalledProcessError
import argparse
import os


PASSWORD_HASH =u'sha1:dad683c1360c:2018b7b80d3880795040546b66cf14b3d8d80796'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/root')
    args = parser.parse_args()

    workdir = args.dir
    os.chdir(workdir)

    os.mkdir('certs')
    os.chdir('certs')
    try:
        ssl_args = (
            'openssl req -x509 -nodes -days 365 -newkey rsa:1024 '
            '-keyout jupyter.key -out jupyter.pem '
            '-subj /CN=ilia.zaitsev@outlook.com/O=ilia.zaitsev/C=RU')
        check_call(ssl_args, shell=True)
    except CalledProcessError as e:
        print("Error: cannot create certificate")
        print(e)
        return
    else:
        os.chdir(workdir)

    try:
        check_call(['jupyter', 'notebook', '--generate-config'])
    except CalledProcessError as e:
        print("Error: cannot create Jupyter notebook config")
        print(e)
        return
    else:
        filename = os.path.join('.jupyter', 'jupyter_notebook_config.py')
        with open(filename) as fp:
            content = [l.strip('\n') for l in fp.readlines() if l != '\n']
        new_lines = [
            "c.NotebookApp.certifile=u'/root/certs/jupyter.pem'",
            "c.NotebookApp.keyfile=u'/root/certs/jupyter.key'",
            "c.NotebookApp.ip = '*'",
            "c.NotebookApp.password = u'%s'" % PASSWORD_HASH,
            "c.NotebookApp.open_browser = False",
            "c.NotebookApp.port = 8080",
            ""]
        with open(filename, 'w') as fp:
            fp.write('\n'.join(new_lines + content) + '\n')
        os.chdir(workdir)


if __name__ == '__main__':
    main()