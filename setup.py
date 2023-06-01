from setuptools import setup, find_packages

setup(
    name='object_detection_app',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'streamlit==1.22.0',
        'torch',
        'transformers',
        'Pillow'
        'requests==2.26.0'
    ],
    entry_points={
        'console_scripts': [
            'object_detection_app = app:main'
        ]
    }
)
