import setuptools

setuptools.setup(name='topicmodelling',
                 version='0.1.0',
                 description='LDA and Twitter-LDA implementations',
                 long_description=open('README.md').read().strip(),
                 author='Ashish Baghudana',
                 author_email='ashish.baghudana26@gmail.com',
                 url='http://github.com/ashishbaghudana/topicmodelling',
                 py_modules=['topicmodelling'],
                 install_requires=['numpy', 'scipy', 'gensim'],
                 license='MIT License',
                 zip_safe=False,
                 keywords='topic modelling natural language processing',
                 classifiers=['Topic Modeling', 'Natural Language Processing'])
