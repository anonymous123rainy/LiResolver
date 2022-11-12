# coding: utf-8

import os

from setuptools.command.install import install

from setuptools import setup


with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()


class InstallCommand(install):
    def run(self):
        common_dir = 'tgrocery/learner'
        libpostfix = '.dll' if os.name == 'nt' else '.so.1'
        #cp = 'copy' if os.name == 'nt' else 'cp'
        target_dir = '%s/%s' % (self.build_lib, common_dir)
        self.mkpath(target_dir)
        self.copy_file('%s/util%s' % (common_dir, libpostfix), target_dir)
        #os.system('cp %s/util.%s %s' % (common_dir, libpostfix, target_dir))
        common_dir = 'tgrocery/learner/liblinear'
        target_dir = '%s/%s' % (self.build_lib, common_dir)
        self.mkpath(target_dir)
        self.copy_file('%s/liblinear%s' % (common_dir, libpostfix), target_dir)
        #os.system('cp %s/liblinear.%s %s' % (common_dir, libpostfix, target_dir))
        install.run(self)


class MakeAndInstallCommand(InstallCommand):
    def run(self):
        makecmd = 'nmake -f Makefile.win clean liball' if os.name == 'nt' else 'make'
        os.system(makecmd)
        InstallCommand.run(self)

setup(
    name='tgrocery',
    version='0.2.0',
    packages=['tgrocery', 'tgrocery.learner', 'tgrocery.learner.liblinear.python'],
    url='https://github.com/cosmichut/TextGrocery',
    license='BSD',
    author='cosmichut',
    author_email='cosmichut@msn.com',
    description='A simple short-text classification tool based on LibLinear, for windows and linux both',
    long_description=LONG_DESCRIPTION,
    install_requires=['jieba'],
    keywords='text classification svm liblinear libshorttext',
    cmdclass={'install': InstallCommand, 'make_and_install': MakeAndInstallCommand}
)
